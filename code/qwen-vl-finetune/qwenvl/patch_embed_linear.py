"""Work around PyTorch 2.9.0's Conv3d regression in Qwen3-VL training.

The nonoverlapping patch projection is exactly a flattened linear operation.
Keep the Conv3d module and its parameters so checkpoint keys/optimizers do not
change. Per-patch GEMM retains the native projection shape and rounding; a single
large GEMM was rejected because its BF16 rounding changed vision features.
See vllm-project/vllm#27406 and pytorch/pytorch#166122.
"""
import logging
import os
from types import MethodType
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def linear_patch_forward(self, hidden_states):
    proj = self.proj
    kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
    if (tuple(proj.kernel_size) != kernel or tuple(proj.stride) != kernel
            or tuple(proj.padding) != (0, 0, 0) or tuple(proj.dilation) != (1, 1, 1)
            or proj.groups != 1 or proj.padding_mode != 'zeros'):
        raise ValueError('Linear patch projection requires the original nonoverlapping Qwen3-VL Conv3d')
    width = self.in_channels * self.temporal_patch_size * self.patch_size ** 2
    pixels = hidden_states.reshape(-1, width).to(dtype=proj.weight.dtype)
    weight = proj.weight.reshape(self.embed_dim, width)
    # Preserve the native one-patch GEMM shape and rounding. A single large
    # BF16 GEMM (or FP32 accumulation) changes vision features measurably.
    # These lightweight projections avoid the costly Conv3d fallback while
    # retaining its per-patch numerical behavior and ordinary autograd.
    if pixels.shape[0] == 0:
        return F.linear(pixels, weight, proj.bias)
    return torch.cat([F.linear(patch, weight, proj.bias)
                      for patch in pixels.split(1, dim=0)], dim=0)

def enable_patch_embed_linear(model):
    """Enable only for the verified torch version; return patched module count."""
    if os.getenv('QWEN3VL_PATCH_EMBED_LINEAR', '1') == '0':
        logger.warning('Qwen3-VL linear patch embedding disabled by environment')
        return 0
    if torch.__version__.split('+')[0] != '2.9.0':
        logger.info('Keeping native patch embedding on torch %s', torch.__version__)
        return 0
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed
    count = 0
    for module in model.modules():
        if isinstance(module, Qwen3VLVisionPatchEmbed):
            module.forward = MethodType(linear_patch_forward, module)
            count += 1
    logger.warning('Qwen3-VL patch embedding: linear projection enabled for %d module(s); weights and input pixels preserved', count)
    return count
