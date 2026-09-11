"""The imported performance hook preserves checkpoint structure and results."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.qwen_patch_layout import enable_qwen_patch_channels_last


class PatchLayoutTest(unittest.TestCase):
    def test_cpu_identity_and_idempotence(self):
        module=Qwen3VLVisionPatchEmbed(SimpleNamespace(patch_size=4,temporal_patch_size=2,in_channels=3,hidden_size=8))
        x=torch.randn(5,3*2*4*4)
        expected=module(x); before=set(module.state_dict())
        self.assertEqual(enable_qwen_patch_channels_last(module),1)
        enable_qwen_patch_channels_last(module)
        self.assertEqual(len(module.proj._forward_pre_hooks),1)
        self.assertEqual(set(module.state_dict()),before)
        torch.testing.assert_close(module(x),expected,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_layout_and_numerics(self):
        module=Qwen3VLVisionPatchEmbed(SimpleNamespace(patch_size=4,temporal_patch_size=2,in_channels=3,hidden_size=8)).cuda().bfloat16()
        x=torch.randn(5,3*2*4*4,device='cuda',dtype=torch.bfloat16)
        expected=module(x)
        enable_qwen_patch_channels_last(module)
        seen=[]
        module.proj.register_forward_pre_hook(lambda m,args:seen.append(args[0].is_contiguous(memory_format=torch.channels_last_3d)))
        torch.testing.assert_close(module(x),expected,rtol=.03,atol=.03)
        self.assertEqual(seen,[True])


if __name__=='__main__':unittest.main()
