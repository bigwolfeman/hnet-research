"""flash_attn.ops.triton.rotary.apply_rotary in torch (same arguments and semantics as the Triton kernel).

    x (B,S,H,D) or packed (T,H,D) with cu_seqlens; cos/sin (S_ro, rotary_dim/2)
    position of token = its index in its own sequence + seqlen_offsets (int or per-batch tensor)
    first rotary_dim dims rotated (GPT-NeoX halves, or interleaved pairs), the rest passed through; conjugate: sin -> -sin
"""
from __future__ import annotations

import torch


def _rotate(x, cos, sin, interleaved: bool):
    """x (..., rd) float32; cos/sin (..., rd/2) broadcastable."""
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        o1, o2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
        return torch.stack((o1, o2), dim=-1).flatten(-2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def apply_rotary(x, cos, sin, seqlen_offsets=0, cu_seqlens=None, max_seqlen=None, interleaved=False, inplace=False,
                 conjugate=False):
    rd = cos.shape[-1] * 2
    assert rd <= x.shape[-1]
    if cu_seqlens is None:
        B, S = x.shape[:2]
        pos = torch.arange(S, device=x.device)[None, :]
        pos = pos + (seqlen_offsets[:, None] if torch.is_tensor(seqlen_offsets) else seqlen_offsets)  # (B or 1, S)
        c, s_ = cos[pos][:, :, None, :], sin[pos][:, :, None, :]            # (B,S,1,rd/2)
    else:
        T = x.shape[0]
        starts = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff())
        pos = torch.arange(T, device=x.device) - starts
        assert not torch.is_tensor(seqlen_offsets) and seqlen_offsets == 0, "varlen with offsets is not used"
        c, s_ = cos[pos][:, None, :], sin[pos][:, None, :]                  # (T,1,rd/2)
    c, s_ = c.float(), s_.float()
    if conjugate:
        s_ = -s_
    rot = _rotate(x[..., :rd].float(), c, s_, interleaved).to(x.dtype)
    if inplace:
        x[..., :rd] = rot
        return x
    return torch.cat((rot, x[..., rd:]), dim=-1)
