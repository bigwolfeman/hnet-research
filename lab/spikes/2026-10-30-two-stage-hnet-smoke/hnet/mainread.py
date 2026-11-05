"""MainRead: a content read of the main network's past outputs, added to a stage's decoder input.

Experiments: lab/experiments/refuted/2026-11-03-decoder-reads-main-by-content.md (at stage 1),
             lab/experiments/successes/2026-11-03-bytes-read-main-by-content.md (at stage 0, the bytes)
Why: lab/experiments/refuted/2026-11-03-recall-gain-by-boundary.md. With the EMA dechunk, a position reads only the
inner output of its own chunk. Chunks start at or before a word, so at the letters where the document memory pays,
that output holds nothing of the current word, and the decoder must pick the word from a bag held in one vector.
MainRead lets the position match its own prefix (in the encoder state) against every past main output. The query
must have seen the prefix: at stage 1 it had not (stage-0 chunks also start at the space), so the bytes query.

Pseudocode (one attention layer, no positional encoding, zero-init output):
    q = W_q norm(e_t)                       # e = this stage's encoder output at position t (has seen t's prefix)
    k_j, v_j = W_k norm(y_j), W_v norm(y_j) # y = MAIN outputs, one per main chunk (first d dims, as the EMA reads)
    position t may read chunk j  <=>  j <= read_idx[t]   (Stage passes main_idx: the main chunks t's EMA path sees)
    r_t = W_o attention(q_t, k_{<=read_idx[t]}, v)     ; r_t = 0 where read_idx[t] < 0 (start vector: no chunk yet)
    decoder input z_t = (EMA read) + residual + r_t
The main network stays the only path across a document: r reads only main outputs.
Probe ablation (Stage.ablate) acts on r as on the EMA read: "zero" -> 0, "shuffle" -> r of row b-1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm


class MainRead(nn.Module):
    def __init__(self, d: int, head_dim: int):
        super().__init__()
        assert d % head_dim == 0, f"d {d} not divisible by head_dim {head_dim}"
        self.d, self.head_dim, self.heads = d, head_dim, d // head_dim
        self.norm_q, self.norm_kv = RMSNorm(d), RMSNorm(d)
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.o = nn.Linear(d, d, bias=False)

    def init_weights(self, std: float):
        for lin in (self.q, self.k, self.v):
            nn.init.normal_(lin.weight, mean=0.0, std=std)
        nn.init.zeros_(self.o.weight)   # r = 0 at init: the stage starts as the plain layout

    def forward(self, e: torch.Tensor, y: torch.Tensor, read_idx: torch.Tensor) -> torch.Tensor:
        """e (B,L,d) encoder output, y (B,M,d) inner outputs, read_idx (B,L) long. Returns r (B,L,d)."""
        B, L, _ = e.shape
        M = y.shape[1]
        H, hd = self.heads, self.head_dim
        q = self.q(self.norm_q(e)).view(B, L, H, hd).transpose(1, 2)
        kv = self.norm_kv(y)
        k = self.k(kv).view(B, M, H, hd).transpose(1, 2)
        v = self.v(kv).view(B, M, H, hd).transpose(1, 2)
        # a position with no chunk yet attends to chunk 0 (no all-masked rows, no NaN); its output is zeroed below
        allow = torch.arange(M, device=e.device)[None, None, :] <= read_idx.clamp(min=0)[:, :, None]   # (B,L,M)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[:, None])
        out = self.o(out.transpose(1, 2).reshape(B, L, self.d))
        return torch.where((read_idx < 0)[..., None], torch.zeros_like(out), out)

    def describe(self) -> dict:
        return dict(d=self.d, head_dim=self.head_dim, heads=self.heads,
                    params=sum(p.numel() for p in self.parameters()))
