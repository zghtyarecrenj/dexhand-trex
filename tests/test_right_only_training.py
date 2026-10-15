"""Real, small T-Rex models: loss gradients, single-hand VQ-VAE and JSON splits."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.flow_loss import action_padding_mask, masked_flow_mse

spec = importlib.util.spec_from_file_location('collection_train', ROOT/'scripts/train.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
torch.set_num_threads(2)


from utils.training_smoke import small_model, check_cascaded_backward


class RightOnlyTrainingTests(unittest.TestCase):
    def test_padding_has_zero_loss_and_gradient_even_for_invalid_tail_values(self):
        prediction = torch.ones(2,16,31,requires_grad=True)
        target = torch.zeros_like(prediction)
        pad = torch.ones(2,16,dtype=torch.bool); pad[0,:3]=False; pad[1,:1]=False
        target[pad] = float('nan')
        loss = masked_flow_mse(prediction,target,pad)
        self.assertEqual(loss.item(),1.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(torch.count_nonzero(prediction.grad[pad]),0)
        self.assertTrue(torch.all(prediction.grad[~pad]>0))

    def test_legacy_bimanual_loss_is_unchanged_without_padding(self):
        p=torch.randn(2,16,62); t=torch.randn_like(p)
        torch.testing.assert_close(masked_flow_mse(p,t),torch.nn.functional.mse_loss(p,t))

    def test_all_padding_returns_differentiable_zero(self):
        p=torch.randn(2,16,31,requires_grad=True)
        loss=masked_flow_mse(p,torch.zeros_like(p),torch.ones(2,16,dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(),0.)
        self.assertEqual(torch.count_nonzero(p.grad),0)

    def test_padding_contract_rejects_wrong_length_and_non_boolean_masks(self):
        for mask in ([False]*15,[0]*16):
            with self.assertRaisesRegex(ValueError,'boolean vector'):
                action_padding_mask([{'action_is_pad':mask}],16)
        self.assertFalse(action_padding_mask([{}],16).any())

    def test_actual_right_only_cascaded_model_backward(self):
        pad=torch.zeros(2,16,dtype=torch.bool); pad[1,1:]=True
        check_cascaded_backward(torch.randn(2,16,31),torch.randn(2,5,6),pad,
                               torch.rand(2,5,1,240,240))

    def test_actual_cascaded_model_masks_g_hand_in_both_heads(self):
        pad=torch.zeros(2,16,dtype=torch.bool); pad[:,8:]=True
        mask=torch.ones(2,16,31,dtype=torch.bool); mask[1,:,9:]=False
        check_cascaded_backward(torch.randn(2,16,31),torch.randn(2,5,6),pad, supervision=mask)

    def test_actual_cascaded_model_preserves_internal_handoff_holes(self):
        pad=torch.zeros(2,16,dtype=torch.bool); pad[:,[3,11]]=True
        check_cascaded_backward(torch.randn(2,16,31),torch.randn(2,5,6),pad,
                               supervision=torch.ones(2,16,31,dtype=torch.bool))

    def test_single_and_bimanual_vqvae_normalization_and_code_counts(self):
        for dim,nf in ((31,5),(62,10)):
            for granularity in ('hand','finger'):
                model=small_model(dim,vqvae=True,granularity=granularity)
                hist=torch.rand(2,16,nf,6)
                model.tacf6_vqvae_min.fill_(-1); model.tacf6_vqvae_max.fill_(3)
                actual=model.encode_tactile_f6_history(hist)
                normed=2*(hist+1)/4-1
                expected=[]
                for start in range(0,nf,5):
                    expected.append(model.tactile_vqvae.encode(normed[:,:,start:start+5]).reshape(2,-1))
                torch.testing.assert_close(actual,torch.cat(expected,dim=1))
                self.assertEqual(actual.shape,(2,nf if granularity=='finger' else nf//5))
                with self.assertRaisesRegex(ValueError,'fingertip count'):
                    model.encode_tactile_f6_history(torch.rand(2,16,6,6))

    def test_bimanual_vqvae_stats_require_explicit_hand_selection(self):
        model=small_model(vqvae=True)
        blob={'model_state':model.tactile_vqvae.state_dict(),
              'stats':{'tacf6_min':[-2]*30+[-4]*30,'tacf6_max':[2]*30+[4]*30,
                       'tacf6_mask':[True]*60}}
        with self.assertRaisesRegex(ValueError,'stats_hand'):
            model.load_tactile_vqvae(blob)
        model.load_tactile_vqvae(blob,stats_hand='right')
        self.assertTrue(torch.all(model.tacf6_vqvae_min==-4))
        self.assertTrue(torch.all(model.tacf6_vqvae_max==4))

    def test_json_collator_carries_mixed_axis_masks_to_loss_batch(self):
        from transformers import BatchEncoding
        class Processor:
            tokenizer=SimpleNamespace(pad_token_id=0)
            def apply_chat_template(self, *a, **k): return 'fixture'
            def __call__(self, **k): return BatchEncoding({'input_ids':torch.tensor([[1,2]])})
        ds=object.__new__(trainer.SftDataset)
        ds.config=SimpleNamespace(action_chunk=16,action_dim=31,use_tactile_vec=0,
            use_tactile_deform=0,use_robot_state=0,use_flare=0,flare_frame_stride=1)
        ds.action_mask=np.ones(31,bool);ds.action_min=np.zeros(31);ds.action_max=np.ones(31)
        ds.processor=Processor()
        rows=[{'action':np.zeros((16,31)).tolist(), 'action_supervision_mask':None},
              {'action':np.ones((16,31)).tolist(), 'action_supervision_mask':[True]*9+[False]*22}]
        batch=ds.collate_fn(rows)
        mask=batch['action_supervision_mask']
        self.assertEqual(tuple(mask.shape),(2,16,31))
        self.assertTrue(mask[0].all())
        self.assertTrue(mask[1,:,:9].all())
        self.assertFalse(mask[1,:,9:].any())

    def test_json_split_keeps_original_demonstrations_together(self):
        ds=object.__new__(trainer.SftDataset)
        ds.accelerator=SimpleNamespace(print=lambda *a:None)
        ds.hf_dataset=Dataset.from_dict({'source_episode_id':[0,0,1,2,2],
                                       'episode_id':[0,1,2,3,4]})
        val=ds.create_val_split(.5,42)
        self.assertFalse(set(ds.hf_dataset['source_episode_id']) & set(val.hf_dataset['source_episode_id']))
        ds.hf_dataset=Dataset.from_dict({'source_episode_id':[0,0]})
        with self.assertRaisesRegex(ValueError,'two independent'):
            ds.create_val_split(.5,42)

    def test_json_five_finger_history_pads_within_its_segment(self):
        ds=object.__new__(trainer.SftDataset)
        ds.vqvae_window=4; ds._ep_f6={}; ds._f6_loc={}
        ds.accelerator=SimpleNamespace(print=lambda *a:None)
        rows=[{'tactile_f6':np.full((5,6),value).tolist(),
               'tactile_image_deform':[f'media/episode_{ep}/image{i}_deform.png']}
              for ep,values in ((0,[1,2]),(1,[7,8])) for i,value in enumerate(values)]
        ds.hf_dataset=Dataset.from_list(rows)
        ds._build_f6_history_index()
        np.testing.assert_array_equal(ds._f6_history_for_sample(rows[2]),np.full((4,5,6),7))
        np.testing.assert_array_equal(ds._f6_history_for_sample(rows[3])[:,0,0],[7,7,7,8])


if __name__=='__main__':
    unittest.main()
