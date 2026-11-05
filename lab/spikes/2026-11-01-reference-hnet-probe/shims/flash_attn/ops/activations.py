"""flash_attn.ops.activations.swiglu(x, y) = silu(x) * y (the fused kernel's math, in torch)."""
import torch.nn.functional as F


def swiglu(x, y):
    return F.silu(x.float()).to(x.dtype) * y
