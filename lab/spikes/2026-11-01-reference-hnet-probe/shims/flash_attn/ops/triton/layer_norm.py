"""flash_attn.ops.triton.layer_norm.RMSNorm with the fused add + norm semantics, in torch.

    forward(x, residual=None, prenorm=False, residual_in_fp32=False):
        s   = x + residual   (in fp32)            # or x if residual is None
        res = s in fp32 if residual_in_fp32 else s in residual's (or x's) dtype
        y   = s * rsqrt(mean(s^2) + eps) * weight, cast to x.dtype
        return (y, res) if prenorm else y
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, zero_centered_weight=False, device=None, dtype=None):
        super().__init__()
        assert dropout_p == 0.0 and not zero_centered_weight
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        s = x.float() if residual is None else x.float() + residual.float()
        res_dtype = torch.float32 if residual_in_fp32 else (residual.dtype if residual is not None else x.dtype)
        y = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)
        return (y, s.to(res_dtype)) if prenorm else y
