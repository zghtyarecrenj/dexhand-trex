import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import numpy as np
import pytest
import torch
from transformers import BatchEncoding

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.mixed_data import (MixedBatchSampler, ExactValidationSampler,
    validate_split_rows, fixed_randomness, accumulate_validation)
from utils.flow_loss import masked_flow_mse
spec=importlib.util.spec_from_file_location('mixed_trainer_test', ROOT/'scripts/train.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


@pytest.mark.parametrize('world,batch',[(8,8),(4,16)])
def test_global_quota_disjoint_ranks_and_epoch_changes(world,batch):
    samplers=[MixedBatchSampler(list(range(1000)),list(range(1000,2000)),batch,r,world,10) for r in range(world)]
    draws=[list(s) for s in samplers]
    flat=[]
    for step in range(10):
        glob=[i for rank in draws for i in rank[step]]
        assert len(glob)==64 and len(set(glob))==64
        assert sum(i<1000 for i in glob)==48
        for rank in draws:assert sum(i<1000 for i in rank[step])==batch*3//4
        flat.extend(glob)
    assert len(flat)==len(set(flat))  # no repeats before pool exhaustion
    assert list(samplers[0])==draws[0]
    samplers[0].set_epoch(1);assert list(samplers[0])!=draws[0]


def test_cycles_keep_ratio_and_reject_invalid_configuration():
    draws=list(MixedBatchSampler([0,1,2],[3],8,0,1,20))
    assert all(sum(i<3 for i in b)==6 for b in draws)
    with pytest.raises(ValueError):MixedBatchSampler([1],[1],8,0,1,1)
    with pytest.raises(ValueError):MixedBatchSampler([1],[2],7,0,1,1)


def test_full_budget_covers_every_row_without_epoch_reset():
    from collections import Counter
    original=list(range(77373));dagger=list(range(77373,77373+59533))
    counts=Counter();first_pass=Counter()
    for rank in range(8):
        for step,batch in enumerate(MixedBatchSampler(original,dagger,8,rank,8,7442)):
            counts.update(batch)
            if step<3721:first_pass.update(batch)
    assert all(first_pass[i]>=1 for i in original+dagger)
    assert set(counts[i] for i in original)=={4,5}
    assert set(counts[i] for i in dagger)=={2,3}
    assert sum(counts[i] for i in original)==357216
    assert sum(counts[i] for i in dagger)==119072


@pytest.mark.parametrize('size',[1,7,63,64,65,6917,6277])
def test_validation_exact_partition_with_equal_collectives(size):
    ranks=[list(ExactValidationSampler(size,8,r,8)) for r in range(8)]
    assert len({len(x) for x in ranks})==1
    indices=[i for x in ranks for b in x for i in b if i!=-1]
    assert sorted(indices)==list(range(size))


def test_namespaced_source_split_guard():
    rows=[dict(dataset_id='original',source_episode_id='original:1',split='train'),
          dict(dataset_id='dagger',source_episode_id='dagger:1',split='val')]
    validate_split_rows(rows)
    with pytest.raises(ValueError):validate_split_rows(rows+[dict(rows[0],split='val')])
    with pytest.raises(ValueError):validate_split_rows([dict(rows[0],source_episode_id=1)])


class Processor:
    tokenizer=SimpleNamespace(pad_token_id=0)
    def apply_chat_template(self,*a,**k):return 'test'
    def __call__(self,*a,**k):return BatchEncoding(dict(input_ids=torch.ones(1,3,dtype=torch.long)))


def fixture_dataset():
    ds=object.__new__(trainer.SftDataset)
    ds.config=SimpleNamespace(action_chunk=16,action_dim=31,use_tactile_vec=1,use_tactile_deform=0,
        use_tactile_vqvae=0,use_robot_state=1,use_flare=0,flare_frame_stride=2,mixed_data=True)
    ds.processor=Processor();ds.training=True
    for kind,width in [('action',31),('state',31),('tacf6',30)]:
        setattr(ds,kind+'_mask',np.ones(width,bool));setattr(ds,kind+'_min',-10*np.ones(width));setattr(ds,kind+'_max',10*np.ones(width))
    ds.te_mean=np.zeros(28);ds.te_std=np.ones(28)
    row=dict(action=np.zeros((16,31)).tolist(),action_is_pad=[False]*3+[True]+[False]*12,
        action_supervision_mask=[True]*31,human_supervision_mask=np.zeros((16,31),bool).tolist(),
        state_fast=np.linspace(-.1,.1,31).tolist(),tactile_f6=np.zeros((5,6)).tolist(),
        input_image_slow=[],input_image_fast=[],dataset_id='dagger',mixed_row_id='dagger:7')
    return ds,row


def test_validation_clean_state_training_augmentation_and_repeatability():
    ds,row=fixture_dataset()
    with patch.object(trainer,'add_tracking_error_noise',side_effect=lambda state,*a:state+1) as noise:
        train=ds.collate_fn([row]);assert noise.call_count==1
        ds.training=False
        a=ds.collate_fn([row]);b=ds.collate_fn([row]);assert noise.call_count==1
    expected=ds._normalize(np.array(row['state_fast'],np.float32),ds.state_mask,ds.state_min,ds.state_max)
    torch.testing.assert_close(a['state_raw'][0],torch.tensor(expected,dtype=torch.bfloat16))
    assert not torch.equal(train['state_raw'],a['state_raw'])
    for key in ['noisy_actions','target','timesteps','eps_r','time_r','state_raw']:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert a['action_is_pad'][0,3] and not a['action_is_pad'][0,4]
    assert a['action_supervision_mask'].all()  # ownership false does not disable policy hand


def test_validation_rng_does_not_perturb_training():
    torch.manual_seed(44);np.random.seed(44)
    expected=torch.rand(5);expected_np=np.random.rand(5)
    torch.manual_seed(44);np.random.seed(44)
    with fixed_randomness(19):torch.rand(90);np.random.rand(90)
    torch.testing.assert_close(torch.rand(5),expected)
    np.testing.assert_array_equal(np.random.rand(5),expected_np)


def test_metrics_weight_valid_scalars_not_batch_size_or_padding():
    totals=torch.zeros(4,dtype=torch.float64)
    for steps,loss in [(1,1.),(2,9.)]:
        pad=torch.ones(1,16,dtype=torch.bool);pad[:,:steps]=False
        batch=dict(action_is_pad=pad,action_supervision_mask=torch.ones(1,16,31,dtype=torch.bool),metric_real_rows=torch.ones(1))
        accumulate_validation(totals,dict(action_loss=torch.tensor(loss),tactile_loss=torch.tensor(loss*2)),batch)
    assert totals[2]==93 and totals[3]==2
    assert float(totals[0]/totals[2])==pytest.approx(19/3)
    dummy=dict(action_is_pad=torch.ones(1,16,dtype=torch.bool),action_supervision_mask=torch.ones(1,16,31,dtype=torch.bool),metric_real_rows=torch.zeros(1))
    before=totals.clone();accumulate_validation(totals,dict(action_loss=torch.tensor(100.),tactile_loss=torch.tensor(100.)),dummy)
    torch.testing.assert_close(totals,before)


def test_both_experts_keep_g_hand_and_later_targets_with_same_masks():
    from utils.training_smoke import small_model
    from utils.training_runtime import PolicyLossModule
    from types import MethodType
    model=small_model()
    model.prepare_inputs_embeds=MethodType(lambda self,input_ids,**kw:self.model.embed_tokens(input_ids),model)
    model.get_rope_index=MethodType(lambda self,input_ids,**kw:(torch.arange(input_ids.shape[1])[None,None].expand(3,input_ids.shape[0],-1),None),model)
    args=trainer.build_parser().parse_args(['--model_path','unused','--action_dim','31','--action_chunk','16','--use_robot_state','1','--use_tactile_vec','1','--use_tactile_deform','0','--use_flare','0','--cascaded_total_steps','2','--cascaded_split_step','1'])
    ds,row=fixture_dataset();ds.training=False
    batch=ds.collate_fn([row,row])
    batch={k:v.float() if torch.is_tensor(v) and v.dtype==torch.bfloat16 else v for k,v in batch.items()}
    with patch('utils.training_runtime.masked_flow_mse',wraps=masked_flow_mse) as mse:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=PolicyLossModule(model,args)(batch)
        result['loss'].backward()
        assert mse.call_count==2
        for call in mse.call_args_list:
            assert torch.equal(call.args[2],batch['action_is_pad'])
            assert call.args[3][:,4:,9:].all()
        assert torch.isfinite(result['loss'])
        assert model.final_layer.mlp.fc2.weight.grad.abs().sum()>0
