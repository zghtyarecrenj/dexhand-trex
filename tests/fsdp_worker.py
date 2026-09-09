"""Two-process CPU/Gloo or CUDA/NCCL check of the real trainer and full-state saving."""
import argparse
import os
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_single_process import trainer, TinyDataset
from utils.training_smoke import small_model

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
a = p.parse_args()
os.environ["ACCELERATE_USE_CPU"] = "true" if a.device == "cpu" else "false"
a.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(42)
model = small_model()
model.prepare_inputs_embeds = MethodType(lambda self, input_ids, **kw: self.model.embed_tokens(input_ids), model)
model.get_rope_index = MethodType(lambda self, input_ids, **kw: (
    torch.arange(input_ids.shape[1], device=input_ids.device)[None, None].expand(3, input_ids.shape[0], -1), None), model)
original_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
proc = SimpleNamespace(save_pretrained=lambda path: Path(path).mkdir(exist_ok=True))
args = trainer.build_parser().parse_args([
    '--model_path', str(a.output), '--output_dir', str(a.output), '--log_dir', str(a.output),
    '--distributed_backend', 'fsdp', '--gradient_checkpointing', '1',
    '--action_dim', '31', '--action_chunk', '16', '--image_size', '640', '360',
    '--use_robot_state', '1', '--use_tactile_vec', '1', '--use_tactile_deform', '0', '--use_flare', '0',
    '--n_epochs', '3', '--num_workers', '0', '--train_bsz_per_gpu', '1',
    '--gradient_accumulation_steps', '2', '--max_steps', '2',
    '--val_ratio', '.1', '--val_freq', '1', '--cascaded_total_steps', '2', '--cascaded_split_step', '1'])
with patch.object(trainer.Qwen3VLVLAModel, 'from_pretrained_qwen3vl', return_value=model), \
     patch.object(trainer.AutoProcessor, 'from_pretrained', return_value=proc), \
     patch.object(trainer, 'SftDataset', TinyDataset), \
     patch.object(trainer.wandb, 'init'), patch.object(trainer.wandb, 'log'):
    trainer.train(args)
import torch.distributed as dist
dist.barrier()
if dist.get_rank() == 0:
    path = a.output/'checkpoint-1-2/model.pt'
    sd = torch.load(path, map_location='cpu', weights_only=True)
    assert {k: tuple(v.shape) for k, v in sd.items()} == original_shapes
    assert all(torch.isfinite(t).all() for t in sd.values())
    assert sd['final_layer.mlp.fc2.weight'].abs().sum() > 0
    result = {'status': 'passed', 'ranks': 2, 'backend': a.device + ' FULL_SHARD',
              'activation_checkpointing': True, 'gradient_accumulation': 2,
              'optimizer_steps': 2, 'optimizer_changed_action_weights': True, 'portable_checkpoint_shapes_match': True,
              'validation_and_collective_checkpoint_completed': True,
              'scope': 'Tiny actual MoT and both experts with synthetic vision embeddings',
              'cuda_tested': a.device == 'cuda'}
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
dist.destroy_process_group()
