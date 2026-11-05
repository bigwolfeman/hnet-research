"""Building blocks: RMSNorm, RoPE, SwiGLU, attention (SWA or global), Mamba-2, Stack.

Pseudocode (one pre-norm block):
    x = x + Mixer(RMSNorm(x))     # attention: SWA via flex_attention, global via SDPA(is_causal); or Mamba-2 (hnet/ssm.py)
    x = x + SwiGLU(RMSNorm(x))    # only in blocks with an MLP
A Stack is a layout of blocks plus a final RMSNorm (as in the 2025 Isotropic module).
Layout strings follow the reference H-Net (goombalab/hnet): "T2" = 2 attention blocks with MLP, "m4" = 4 Mamba-2
blocks without MLP, "T1m4" = one then the other. t/T = attention, m/M = Mamba-2; upper case adds the SwiGLU MLP.
Right padding is safe: attention and Mamba-2 are causal, so a real position never reads a pad position.
Stack.forward(x, pack): with a Pack (hnet/packing.py) the blocks run on the valid positions only, as one sequence:
Mamba-2 gets seq_idx, attention runs varlen (torch.nn.attention.varlen) with cu_seqlens; same math per valid position.
Stack.forward runs in segments, the code between two Mamba-2 calls (Mamba-2's Triton kernels cannot be traced):
    x, h = seg_0(x)                      # [pack] + attention blocks + norm1 of the first Mamba-2 block
    for each Mamba-2 block k:
        mix  = mamba_k(h)                # eager, mamba_ssm kernels
        x, h = seg_k+1(x, mix)           # x + mix [+ MLP of block k] + attention blocks + norm1 of the next Mamba-2 block
    return seg_last(x, mix)              # ... + final norm [+ unpack]
Each segment compiles as one graph (run_segment). The residual add fuses with the next norm, the residual stream x
flows from graph to graph (no eager gradient adds), and h leaves the graph already cast to the autocast dtype (the
cast autocast would do in Mamba-2's in_proj; same values). A stack without Mamba-2 is one segment.
Note: .agents/notes/implemented/feature/2026-11-05-throughput-compile-and-packing.md
Experiment that motivated Mamba-2 here: lab/experiments/refuted/2026-11-01-reference-hnet-difference-ablation.md
"""
from __future__ import annotations

import math
import re
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import Pack, pack_rows, unpack_rows

_FLEX = None  # compiled flex_attention, created lazily
_BLOCK_MASKS: dict = {}
_DENSE_MASKS: dict = {}
_ROPE: dict = {}
# run_segment is one code object shared by every Stack: one cache entry per (segment structure, shapes, grad mode).
# The 2-stage reference layout needs about 14 in training, more with eval and probe shapes; past the limit Dynamo
# silently runs the frame eagerly.
SEGMENT_RECOMPILE_LIMIT = 64


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_cos_sin(L: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (L, head_dim, theta, str(device))
    if key not in _ROPE:
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        ang = torch.arange(L, device=device, dtype=torch.float32)[:, None] * inv[None]
        _ROPE[key] = (ang.cos(), ang.sin())  # (L, hd/2)
    return _ROPE[key]


def apply_rope(x, cos, sin):
    """x: (B,H,L,hd). Rotates the first 2 * cos.shape[-1] dims of each head in interleaved pairs (x0,x1), (x2,x3), ...
    and passes the rest through (partial RoPE when rope_dim < head_dim). Computed in fp32.
    The reference rotates halves (x_i, x_{i+r/2}) instead; that is a fixed permutation of q/k dims, so the two
    conventions give the same model family and the same init distribution."""
    r = 2 * cos.shape[-1]
    xf = x.float()
    rot, rest = xf[..., :r], xf[..., r:]
    x1, x2 = rot[..., ::2], rot[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    if rest.shape[-1]:
        out = torch.cat([out, rest], dim=-1)
    return out.to(x.dtype)


def _swa_block_mask(L: int, window: int, device):
    from torch.nn.attention.flex_attention import create_block_mask

    key = (L, window, str(device))
    if key not in _BLOCK_MASKS:
        def mask_mod(b, h, q, kv):
            return (q >= kv) & (q - kv < window)

        _BLOCK_MASKS[key] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device)
    return _BLOCK_MASKS[key]


def _swa_dense_mask(L: int, window: int, device):
    key = (L, window, str(device))
    if key not in _DENSE_MASKS:
        i = torch.arange(L, device=device)
        d = i[:, None] - i[None, :]
        _DENSE_MASKS[key] = (d >= 0) & (d < window)
    return _DENSE_MASKS[key]


def _flex():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


class Attention(nn.Module):
    """Causal MHA. window=None -> global (SDPA is_causal). window=w -> sliding window of w keys."""

    def __init__(self, d: int, head_dim: int, window: int | None, impl: str):
        super().__init__()
        assert d % head_dim == 0
        self.d, self.head_dim, self.n_heads = d, head_dim, d // head_dim
        self.window, self.impl = window, impl
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def attn_mask(self, L: int, device):
        """Mask object built outside any compiled region: None (global/causal), BlockMask (flex) or dense bool."""
        if self.window is None or self.window >= L:
            return None
        if self.impl == "flex":
            return _swa_block_mask(L, self.window, device)
        if self.impl == "sdpa_mask":
            return _swa_dense_mask(L, self.window, device)
        raise ValueError(f"attn_impl {self.impl!r}")

    def forward(self, x, cos, sin, mask):
        B, L, _ = x.shape
        if isinstance(mask, Pack):
            return self._varlen(x, cos, sin, mask)
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.impl == "flex":
            o = _flex()(q, k, v, block_mask=mask)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(o.transpose(1, 2).reshape(B, L, self.d))

    def _varlen(self, x, cos, sin, pack: Pack):
        """Packed (1, T, d) input; cos/sin are (T, rope/2), gathered at each position's in-row index."""
        from torch.nn.attention.varlen import varlen_attn

        T = x.shape[1]
        q, k, v = self.qkv(x[0]).view(T, 3, self.n_heads, self.head_dim).unbind(1)   # (T, H, hd)
        cs, sn = cos[:, None], sin[:, None]
        q, k = apply_rope(q, cs, sn), apply_rope(k, cs, sn)
        # window w keys = i-w+1..i -> flash's (left, right) = (w - 1, 0); global causal = (-1, 0)
        left = -1 if self.window is None else self.window - 1
        o = varlen_attn(q, k, v, pack.cu, pack.cu, pack.max_len, pack.max_len, window_size=(left, 0))
        return self.out_proj(o.reshape(1, T, self.d))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.fc2(F.silu(g) * u)


_LAYOUT_PART = re.compile(r"([mMtT])(\d+)")


def parse_layout(layout: str) -> list[tuple[str, bool]]:
    """'T1m4' -> [('attn', True), ('mamba', False) x 4]. The one parser for model building and FLOPs counting."""
    parts = _LAYOUT_PART.findall(layout)
    if "".join(a + n for a, n in parts) != layout:
        raise ValueError(f"layout {layout!r}: expected letters t/T/m/M each followed by a count, e.g. 'T1m4'")
    return [("mamba" if a in "mM" else "attn", a.isupper()) for a, n in parts for _ in range(int(n))]


class Block(nn.Module):
    """Pre-norm block. mixer 'attn' -> self.attn (Attention); 'mamba' -> self.mamba (Mamba-2). mlp=False: no SwiGLU."""

    def __init__(self, d, head_dim, hidden, window, impl, mixer: str = "attn", mlp: bool = True, ssm: dict | None = None):
        super().__init__()
        self.mixer = mixer
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d) if mlp else None   # registered before the mixer: the state_dict order of older checkpoints
        if mixer == "attn":
            self.attn = Attention(d, head_dim, window, impl)
        elif mixer == "mamba":
            from .ssm import mamba2  # imported only when a layout has Mamba-2 blocks (needs mamba_ssm)
            if not ssm:
                raise ValueError("a Mamba-2 block needs model.ssm (d_state, d_conv, expand, headdim, chunk_size)")
            self.mamba = mamba2(d, ssm)
        else:
            raise ValueError(f"mixer {mixer!r}")
        self.mlp = SwiGLU(d, hidden) if mlp else None

    def forward(self, x, cos, sin, mask, seq_idx=None):
        h = self.norm1(x)
        return self.finish(x, self.attn(h, cos, sin, mask) if self.mixer == "attn" else _run_mamba(self.mamba, h, seq_idx))

    def finish(self, x, mix):
        """The block after its mixer: residual add, then the MLP half (if any)."""
        x = x + mix
        return x if self.mlp is None else x + self.mlp(self.norm2(x))


@torch.compiler.disable
def _run_mamba(mamba, h, seq_idx):
    """Mamba-2 runs its own fused Triton kernels; Dynamo cannot trace them (it fails inside mamba_ssm's autotuner)."""
    return mamba(h, seq_idx=seq_idx)


class Segment(NamedTuple):
    """One compile unit of a Stack: the code between two Mamba-2 calls (see run_segment). Holds references to blocks
    that Stack.blocks registers; a NamedTuple, not a module, so no parameter is registered twice."""
    tail: Block | None     # Mamba-2 block whose mixer output this segment adds first; None: the first segment
    blocks: tuple          # attention blocks, run whole
    head: Block | None     # Mamba-2 block whose mixer input this segment returns; None: the last segment


def split_segments(blocks) -> list[Segment]:
    """'T1m4' -> [(None, (T,), m0), (m0, (), m1), (m1, (), m2), (m2, (), m3), (m3, (), None)]."""
    segs, tail, run = [], None, []
    for b in blocks:
        if b.mixer == "mamba":
            segs.append(Segment(tail, tuple(run), b))
            tail, run = b, []
        else:
            run.append(b)
    segs.append(Segment(tail, tuple(run), None))
    return segs


def run_segment(seg: Segment, x, mix, cos, sin, mask, pack: Pack | None, final_norm: nn.Module, cast):
    """first segment: pack the input; else finish the previous Mamba-2 block. Run the attention blocks. Return
    (x, mixer input of the next Mamba-2 block, cast to the autocast dtype here, not by autocast in its in_proj), or
    after the last block the final norm, unpacked."""
    if seg.tail is None:
        if pack is not None:
            x = pack_rows(x, pack)
    else:
        x = seg.tail.finish(x, mix)
    if seg.blocks:
        if pack is not None:
            cos, sin = cos[pack.pos], sin[pack.pos]
        for blk in seg.blocks:
            x = blk(x, cos, sin, mask)
    if seg.head is not None:
        h = seg.head.norm1(x)
        return x, (h if cast is None else h.to(cast))
    x = final_norm(x)
    return x if pack is None else unpack_rows(x, pack)


def swiglu_hidden(d: int, ratio: float, multiple_of: int) -> int:
    return int(math.ceil(d * ratio / multiple_of) * multiple_of)


class Stack(nn.Module):
    """Pre-norm blocks in `layout` order + final RMSNorm (final_norm=False: none, the Parcae core). window=None means
    global attention."""

    def __init__(self, d, layout: str, head_dim, window, mlp_ratio, mlp_multiple_of, rope_theta, impl,
                 ssm: dict | None = None, rope_dim: int | None = None, final_norm: bool = True):
        super().__init__()
        spec = parse_layout(layout)
        self.d, self.layout, self.layers, self.head_dim, self.window = d, layout, len(spec), head_dim, window
        # residual-stream additions (the 2025 Isotropic.height): 1 per block without an MLP, 2 per block with one.
        # The init scales output projections by 1/sqrt(the sum over enclosing stacks) (hnet/stage.py init_stack).
        self.height = sum(2 if mlp else 1 for _, mlp in spec)
        self.rope_theta = rope_theta
        self.rope_dim = head_dim if rope_dim is None else int(rope_dim)
        if not (0 < self.rope_dim <= head_dim and self.rope_dim % 2 == 0):
            raise ValueError(f"rope_dim {rope_dim}: must be even and in (0, head_dim={head_dim}]")
        self.hidden = swiglu_hidden(d, mlp_ratio, mlp_multiple_of)
        self.blocks = nn.ModuleList(Block(d, head_dim, self.hidden, window, impl, mixer, mlp, ssm) for mixer, mlp in spec)
        # index of the first attention block (it builds the shared mask); an int, so no second module registration
        self._attn_idx = next((i for i, b in enumerate(self.blocks) if b.mixer == "attn"), None)
        self.final_norm = RMSNorm(d) if final_norm else nn.Identity()
        self._segments = split_segments(self.blocks)
        self._seg_fn = run_segment

    def forward(self, x, pack: Pack | None = None):
        """x (B, L, d). pack: run on the valid positions only (hnet/packing.py); pad positions come back as 0."""
        cos = sin = mask = None
        L = x.shape[1] if pack is None else pack.M
        if self._attn_idx is not None:
            cos, sin = rope_cos_sin(L, self.rope_dim, self.rope_theta, x.device)
            mask = self.blocks[self._attn_idx].attn.attn_mask(L, x.device) if pack is None else pack
        dev = x.device.type
        cast = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else None
        seq_idx = None if pack is None else pack.seq_idx
        mix = None
        for seg in self._segments:
            out = self._seg_fn(seg, x, mix, cos, sin, mask, pack, self.final_norm, cast)
            if seg.head is None:
                return out
            x, h = out
            mix = seg.head.mamba(h, seq_idx=seq_idx)

    def compile_blocks(self, dynamic: bool):
        """Compile each segment as one graph; Mamba-2 runs eagerly between them. Masks and RoPE tables stay eager.
        Segments of the same structure share a graph (in every stack)."""
        cfg = torch._dynamo.config
        cfg.recompile_limit = max(cfg.recompile_limit, SEGMENT_RECOMPILE_LIMIT)
        self._seg_fn = torch.compile(run_segment, dynamic=dynamic)

    def describe(self) -> dict:
        return dict(d=self.d, layout=self.layout, layers=self.layers, height=self.height, heads=self.d // self.head_dim,
                    head_dim=self.head_dim, rope_dim=self.rope_dim, mlp_hidden=self.hidden, window=self.window,
                    attn_impl=None if self._attn_idx is None else self.blocks[self._attn_idx].attn.impl, rope_theta=self.rope_theta,
                    params=sum(p.numel() for p in self.parameters()))
