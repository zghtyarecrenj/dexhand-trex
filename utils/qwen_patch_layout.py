"""Avoid the contiguous BF16 Conv3d fallback in Qwen3-VL patch embedding."""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed


def _channels_last_input(module, inputs):
    patches, = inputs
    if patches.is_cuda:
        patches = patches.contiguous(memory_format=torch.channels_last_3d)
    return (patches,)


def enable_qwen_patch_channels_last(model):
    """Preserve parameters/state-dict keys; change only Conv3d input storage."""
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3VLVisionPatchEmbed):
            if not getattr(module.proj, '_trex_channels_last_input', False):
                module.proj.register_forward_pre_hook(_channels_last_input)
                module.proj._trex_channels_last_input = True
            count += 1
    if count == 0:
        raise RuntimeError('No Qwen3-VL vision patch embedding found')
    return count
