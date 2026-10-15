import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.flow_loss import action_supervision_mask, masked_flow_mse
from utils.lerobot_common import supervised_action_stats


def test_mixed_f_g_loss_and_gradients_ignore_policy_hand_and_padding():
    mask = action_supervision_mask([{}, {'action_supervision_mask': [True]*9+[False]*22}], 16, 31)
    prediction = torch.ones(2,16,31, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[1,:,9:] = float('nan')
    padding = torch.zeros(2,16,dtype=torch.bool); padding[:,8:] = True
    target[:,8:] = float('nan')
    loss = masked_flow_mse(prediction, target, padding, mask)
    assert loss.item() == 1
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[1,:,9:].count_nonzero() == 0
    assert prediction.grad[:,8:].count_nonzero() == 0
    assert torch.all(prediction.grad[0,:8] > 0)
    assert torch.all(prediction.grad[1,:8,:9] > 0)


def test_legacy_loss_and_bad_masks():
    p=torch.randn(2,16,31); t=torch.randn_like(p)
    torch.testing.assert_close(masked_flow_mse(p,t), torch.nn.functional.mse_loss(p,t))
    with pytest.raises(ValueError):
        action_supervision_mask([{'action_supervision_mask':[1]*31}],16,31)


def test_action_stats_exclude_g_hand_and_padding():
    a=np.ones((2,16,31)); a[1,:,:9]=3; a[1,:,9:]=999
    pad=np.zeros((2,16),bool);pad[:,8:]=True;a[:,8:]=888
    axes=np.ones((2,31),bool);axes[1,9:]=False
    s=supervised_action_stats(a,pad,axes)
    np.testing.assert_allclose(s['mean'][:9],2)
    np.testing.assert_allclose(s['mean'][9:],1)
    assert max(s['max'])==3


def test_internal_handoff_holes_keep_later_policy_supervision():
    prediction = torch.zeros(1, 16, 31, requires_grad=True)
    target = torch.ones_like(prediction)
    padding = torch.zeros(1, 16, dtype=torch.bool)
    padding[:, [3, 11]] = True
    target[padding] = 1000
    # Successful-continuous exports supervise G's policy hand too.
    axes = action_supervision_mask([{'action_supervision_mask': [True]*31}], 16, 31)
    loss = masked_flow_mse(prediction, target, padding, axes)
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad[padding].count_nonzero() == 0
    assert torch.all(prediction.grad[:, 12:, 9:] != 0)
    stats = supervised_action_stats(target.numpy(), padding.numpy(), np.ones((1, 31), bool))
    np.testing.assert_allclose(stats['mean'], 1)
