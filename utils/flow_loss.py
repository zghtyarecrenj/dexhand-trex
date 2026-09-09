"""Flow supervision for any action width, excluding artificial chunk tails."""
import torch


def action_padding_mask(batch, chunk):
    """Read optional per-step padding; legacy datasets supervise every step."""
    masks = []
    for row in batch:
        mask = torch.as_tensor(row.get('action_is_pad', torch.zeros(chunk, dtype=torch.bool)))
        if mask.dtype != torch.bool or mask.shape != (chunk,):
            raise ValueError('action_is_pad must be a boolean vector of length action_chunk')
        masks.append(mask)
    return torch.stack(masks)


def masked_flow_mse(prediction, target, action_is_pad=None):
    """Mean squared velocity error over recorded steps and actual action axes.

    Both experts use this same objective. A right-only model supplies 31 axes;
    there are no absent left-hand targets. Sum/denominator are float32 even in
    mixed precision. An all-padding batch returns a differentiable zero.
    """
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError('Flow prediction and target must have matching [B,T,D] shapes')
    valid = torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
    if action_is_pad is not None:
        if action_is_pad.dtype != torch.bool or action_is_pad.shape != prediction.shape[:2]:
            raise ValueError('action_is_pad must have boolean [B,T] shape')
        valid = ~action_is_pad.to(prediction.device)
    error = torch.where(valid.unsqueeze(-1), prediction.float() - target.float(), 0.)
    count = valid.sum().float() * prediction.shape[-1]
    return error.square().sum() / count.clamp_min(1.)
