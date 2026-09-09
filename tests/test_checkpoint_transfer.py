import copy
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.training_smoke import small_model
from utils.checkpoint_transfer import transfer_state, RESET_PREFIXES


class TransferTests(unittest.TestCase):
    def test_shared_weights_transfer_but_robot_heads_remain_new(self):
        source = small_model(62, vqvae=True)
        target = small_model(31, vqvae=True)
        before = {k: v.clone() for k, v in target.state_dict().items() if k.startswith(RESET_PREFIXES)}
        source.tacf6_vqvae_min.copy_(torch.arange(60))
        source.tacf6_vqvae_max.copy_(torch.arange(60)+100)
        report = transfer_state(target, source.state_dict(), {'action_dim': 62, 'action_chunk': 16})
        for key, value in before.items():
            torch.testing.assert_close(target.state_dict()[key], value)
        torch.testing.assert_close(target.model.layers[0].mlp_action.gate_proj.weight,
                                   source.model.layers[0].mlp_action.gate_proj.weight)
        torch.testing.assert_close(target.model.layers[0].mlp_tactile.gate_proj.weight,
                                   source.model.layers[0].mlp_tactile.gate_proj.weight)
        torch.testing.assert_close(target.tacf6_vqvae_min, torch.arange(30, 60).float())
        self.assertFalse(report['action_heads_sliced'])
        self.assertFalse(report['source_dataset_normalization_loaded'])

    def test_missing_or_incompatible_shared_expert_is_rejected(self):
        source = small_model(62).state_dict()
        target = small_model(31)
        config = {'action_dim': 62, 'action_chunk': 16}
        del source['model.norm_tactile.weight']
        with self.assertRaisesRegex(ValueError, 'Missing shared'):
            transfer_state(target, source, config)
        source['model.norm_tactile.weight'] = torch.zeros(3)
        with self.assertRaisesRegex(ValueError, 'Incompatible shared'):
            transfer_state(target, source, config)

    def test_disabled_optional_encoders_and_flare_are_explicitly_omitted(self):
        source = small_model(62, vqvae=True, deform=True).state_dict()
        source['flare_queries'] = torch.rand(1, 32, 64)
        target = small_model(31)
        report = transfer_state(target, source, {'action_dim': 62, 'action_chunk': 16})
        self.assertIn('flare_queries', report['omitted_source_tensors'])
        self.assertIn('tactile_code_embedder.weight', report['omitted_source_tensors'])


if __name__ == '__main__':
    unittest.main()
