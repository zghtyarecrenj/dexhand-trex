"""Exercise the standalone trainer's real loss/optimizer/validation/save path."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.training_smoke import small_model

spec = importlib.util.spec_from_file_location('single_train', ROOT/'scripts/train.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
torch.set_num_threads(2)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.stats_data = {'test': {}}

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return i

    def create_val_split(self, **kwargs):
        return TinyDataset()

    def collate_fn(self, rows):
        b = len(rows)
        actions = torch.randn(b, 16, 31)
        noise = torch.randn_like(actions)
        f6 = torch.randn(b, 5, 6)
        return dict(input_ids=torch.ones(b, 4, dtype=torch.long),
                    attention_mask=torch.ones(b, 4, dtype=torch.long),
                    n_slow_images=1, state_raw=torch.randn(b, 31),
                    noisy_actions=(actions+noise)/2, norm_actions=actions,
                    target=noise-actions, timesteps=torch.full((b,), .5),
                    eps_r=noise, time_r=torch.full((b,), .4),
                    tactile_f6s=f6, tactile_f6s_delayed=f6,
                    action_is_pad=torch.zeros(b, 16, dtype=torch.bool))


class SingleProcessTests(unittest.TestCase):
    def test_local_metrics_without_distributed_initialization(self):
        m = trainer.TrainingMetrics('cpu')
        m.update(3., 1., 2.)
        m.update(5., 2., 3.)
        self.assertEqual(m.get_metric()['total_loss'], 4.)
        self.assertEqual(m.n_step, 0)

    def test_frozen_deform_is_strict_and_stays_in_eval(self):
        model = small_model(deform=True)
        with self.assertRaises(FileNotFoundError):
            model.load_deform_encoder_weights('/nonexistent/encoder.pth')
        model.deform_encoder.requires_grad_(False)
        model.train()
        self.assertFalse(model.deform_encoder.training)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'encoder.pth'
            torch.save({}, path)
            with self.assertRaises(RuntimeError):
                model.load_deform_encoder_weights(str(path))
            torch.save(model.deform_encoder.state_dict(), path)
            model.load_deform_encoder_weights(str(path))

    def test_two_optimizer_steps_validation_and_checkpoint_on_cpu(self):
        self._exercise_training(accumulation=1, batch_size=2)

    def test_accumulation_flushes_short_final_group_with_activation_checkpointing(self):
        self._exercise_training(accumulation=2, batch_size=1)

    def _exercise_training(self, accumulation, batch_size):
        model = small_model()
        # Synthetic vision embeddings; the MoT and both expert heads are real.
        model.prepare_inputs_embeds = MethodType(
            lambda self, input_ids, **kw: self.model.embed_tokens(input_ids), model)
        model.get_rope_index = MethodType(
            lambda self, input_ids, **kw: (
                torch.arange(input_ids.shape[1])[None, None].expand(3, input_ids.shape[0], -1), None), model)
        proc = SimpleNamespace(save_pretrained=lambda path: Path(path).mkdir())
        with tempfile.TemporaryDirectory() as tmp:
            args = trainer.build_parser().parse_args([
                '--model_path', tmp, '--output_dir', tmp, '--log_dir', tmp,
                '--action_dim', '31', '--action_chunk', '16', '--image_size', '640', '360',
                '--use_robot_state', '1', '--use_tactile_vec', '1',
                '--use_tactile_deform', '0', '--use_flare', '0',
                '--n_epochs', '3', '--num_workers', '0', '--train_bsz_per_gpu', str(batch_size),
                '--gradient_accumulation_steps', str(accumulation), '--gradient_checkpointing', '1',
                '--max_steps', '2', '--val_ratio', '.1', '--val_freq', '2',
                '--cascaded_total_steps', '2', '--cascaded_split_step', '1'])
            before = model.final_layer.mlp.fc2.weight.detach().clone()
            with patch.object(trainer, 'Accelerator', side_effect=lambda **kw: Accelerator(cpu=True, **kw)), \
                 patch.object(trainer.Qwen3VLVLAModel, 'from_pretrained_qwen3vl', return_value=model), \
                 patch.object(trainer.AutoProcessor, 'from_pretrained', return_value=proc), \
                 patch.object(trainer, 'SftDataset', TinyDataset), \
                 patch.object(trainer.wandb, 'init'), patch.object(trainer.wandb, 'log') as log:
                trainer.train(args)
            ckpt = Path(tmp)/'checkpoint-0-2'
            self.assertTrue((ckpt/'model.pt').is_file())
            self.assertEqual(json.loads((ckpt/'training_args.json').read_text())['image_size'], [640, 360])
            self.assertFalse(torch.equal(before, model.final_layer.mlp.fc2.weight))
            self.assertTrue(any('val/action_loss' in c.args[0] for c in log.call_args_list))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
