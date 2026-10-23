"""The dated experiment keeps held-out sources and images out of augmentation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset
from PIL import Image


@pytest.fixture(params=['train.py', 'train_restart_20261011.py'])
def trainer(request):
    path = Path(__file__).resolve().parents[1] / 'scripts' / request.param
    spec = importlib.util.spec_from_file_location('chalk_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_split_keeps_all_source_rows(trainer, monkeypatch):
    monkeypatch.setenv('CHALK_VALIDATION_SOURCES', '[2]')
    ds = object.__new__(trainer.SftDataset)
    ds.is_validation = False
    ds.accelerator = SimpleNamespace(print=lambda *args: None)
    ds.hf_dataset = Dataset.from_dict({'source_episode_id': [0, 1, 2, 2]})
    val = ds.create_val_split()
    assert list(ds.hf_dataset['source_episode_id']) == [0, 1]
    assert list(val.hf_dataset['source_episode_id']) == [2, 2]
    assert val.is_validation and not ds.is_validation


def test_missing_heldout_source_is_rejected(trainer, monkeypatch):
    monkeypatch.setenv('CHALK_VALIDATION_SOURCES', '[9]')
    ds = object.__new__(trainer.SftDataset)
    ds.hf_dataset = Dataset.from_dict({'source_episode_id': [0, 1]})
    with pytest.raises(AssertionError):
        ds.create_val_split()


def test_color_jitter_preserves_geometry_and_skips_validation(trainer, monkeypatch, tmp_path):
    monkeypatch.setenv('CHALK_AUGMENT', 'color')
    path = tmp_path / 'image.png'
    Image.new('RGB', (8, 6), (100, 120, 140)).save(path)
    ds = object.__new__(trainer.SftDataset)
    ds.img_dir, ds.image_size, ds.is_validation = str(tmp_path), None, False
    with patch.object(trainer.np.random, 'uniform', return_value=1.15) as draw:
        augmented = ds._open(path.name)
        assert draw.call_count == 3
        ds.is_validation = True
        clean = ds._open(path.name)
        assert draw.call_count == 3
    assert augmented.size == clean.size == (8, 6)
    assert not np.array_equal(augmented, clean)
    np.testing.assert_array_equal(np.array(clean), np.array(Image.open(path)))
