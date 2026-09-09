"""Offline collection adaptation regressions; no model weights or devices."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('collection_loader', ROOT/'qwen_vla/lerobot_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CollectionLoaderTests(unittest.TestCase):
    def make_loader(self, directory, right_only, vqvae=False):
        dim = 31 if right_only else 62
        fingers = 5 if right_only else 10
        keys = module.DEFORM_KEYS[5:] if right_only else module.DEFORM_KEYS
        features = {module.KEY_STATE: {'shape': [dim]}, module.KEY_ACTION: {'shape': [16, dim]},
                    module.KEY_TACF6: {'shape': [fingers, 6]}, module.KEY_WRIST_R: {}}
        if not right_only:
            features[module.KEY_WRIST_L] = {}
        features.update({key: {} for key in keys})
        meta = Path(directory)/'meta'
        meta.mkdir(exist_ok=True)
        (meta/'info.json').write_text(json.dumps({'fps': 30, 'features': features}))
        stats = {key: {'mask': [True]*size, 'q01': [0]*size, 'q99': [1]*size}
                 for key, size in [('state', dim), ('action', dim), ('tactile_f6', fingers*6)]}
        cfg = SimpleNamespace(lerobot_root=directory, action_dim=dim, use_tactile_vqvae=vqvae)
        return module.TRexLeRobotDataset(cfg, None, SimpleNamespace(print=lambda *a: None),
                                        _ds=[], _stats={'test': stats})

    def test_single_hand_features_are_used_without_left_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = self.make_loader(directory, True)
            self.assertEqual(loader.wrist_keys, [module.KEY_WRIST_R])
            self.assertEqual(loader.deform_keys, module.DEFORM_KEYS[5:])

    def test_bimanual_camera_and_finger_order_stays_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = self.make_loader(directory, False)
            self.assertEqual(loader.wrist_keys, [module.KEY_WRIST_R, module.KEY_WRIST_L])
            self.assertEqual(loader.deform_keys, module.DEFORM_KEYS)

    def test_single_hand_rejects_stock_vqvae_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'ten fingers'):
                self.make_loader(directory, True, vqvae=True)

    def test_one_demonstration_cannot_leak_across_train_and_validation(self):
        segments = [{'segment_id': 0, 'source_index': 0}, {'segment_id': 1, 'source_index': 0}]
        with self.assertRaisesRegex(ValueError, 'two independent'):
            module.split_episode_groups(2, .5, 42, segments)

    def test_fragments_of_a_source_always_stay_together(self):
        segments = [{'segment_id': i, 'source_index': source} for i, source in enumerate([0, 0, 1, 2, 2])]
        for seed in range(10):
            train, val = module.split_episode_groups(5, .5, seed, segments)
            train_sources = {segments[i]['source_index'] for i in train}
            val_sources = {segments[i]['source_index'] for i in val}
            self.assertFalse(train_sources & val_sources)
            self.assertEqual(sorted(train+val), list(range(5)))

    def test_incomplete_group_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cover all'):
            module.split_episode_groups(3, .5, 42, [{'segment_id': 0, 'source_index': 0}])


if __name__ == '__main__':
    unittest.main()
