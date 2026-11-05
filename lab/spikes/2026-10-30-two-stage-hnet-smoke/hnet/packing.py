"""Packed inner sequences. An inner level holds (B, M, d) rows right-padded to the longest row of the batch; with
mixed domains (Go rows make about 1.4x more stage-0 chunks than edu rows) 30-35% of those positions are padding.
A Stack given a Pack runs its blocks on the valid positions only, as one (1, T_pad) sequence:

    make_pack(counts (B,), total T, M):                  # T = sum(counts), read in the stage's one host sync
        T_pad = T rounded up to `multiple`
        valid = arange(M) < counts[:, None]                # (B, M); the packed order is the row-major order of valid slots
        idx   = nonzero_static(valid.flat, size=T_pad, fill=B*M)   # flat slot of each packed position; the
                                                          # T_pad - T tail positions read slot B*M (a zero row)
        row   = min(idx // M, B - 1) ; pos = idx % M      # tail: row B-1, pos 0
        inv   = (B*M,) packed index of each slot (cumsum(valid.flat) - 1), T_pad (a zero row) for pad slots
        cu    = [0, cumsum(counts)], last entry T_pad     # the tail joins the last row; causal, so no real position sees it
    pack_index is pure tensor code with no host sync; Stage compiles it (one graph instead of ~20 eager launches,
    which ran while the GPU sat idle right after the sync).
    pack(x)   = cat(x.flat, 0)[idx]        (1, T_pad, d)
    unpack(y) = cat(y, 0)[inv]             (B, M, d)     pad slots are 0 (the padded path leaves values nobody reads)
    Mamba-2 gets seq_idx = row (state resets at each row start); attention gets varlen cu_seqlens.

Same math as the padded path for every valid position: all mixers are causal and reset at a row start.
Mamba-2 with seq_idx needs causal_conv1d's channel-last path: the in_proj width 2 d_inner + 2 d_state + nheads must
be a multiple of 8 (1288 at d 256; true for d a multiple of 64 with headdim 64), else causal_conv1d raises.
Note: .agents/notes/implemented/feature/2026-11-05-throughput-compile-and-packing.md
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Pack:
    idx: torch.Tensor      # (T_pad,) long, flat slot in (B*M + 1) of each packed position (B*M = zero row)
    inv: torch.Tensor      # (B*M,) long, packed index of each slot (T_pad = zero row)
    seq_idx: torch.Tensor  # (1, T_pad) int32, row of each packed position (Mamba-2)
    pos: torch.Tensor      # (T_pad,) long, position inside the row (RoPE)
    cu: torch.Tensor       # (B+1,) int32 cumulative row starts, cu[-1] = T_pad (varlen attention)
    max_len: int           # upper bound on a row's packed length, a multiple of `multiple` (bounds recompiles)
    B: int
    M: int


def pack_index(counts: torch.Tensor, M: int, T_pad: int):
    """counts (B,) device -> (idx, inv, seq_idx, pos, cu) of the packed layout (see the module pseudocode)."""
    B = counts.shape[0]
    counts = counts.long()
    flat = (torch.arange(M, device=counts.device)[None, :] < counts[:, None]).reshape(-1)
    idx = torch.nonzero_static(flat, size=T_pad, fill_value=-1)[:, 0]   # a fill of B * M would specialize the
    idx = torch.where(idx < 0, B * M, idx)                                 # compiled graph on M (a recompile per M)
    row = idx // M
    pos = idx - row * M
    inv = torch.where(flat, torch.cumsum(flat, 0) - 1, T_pad)
    ends = torch.cumsum(counts, 0)
    cu = torch.cat([ends.new_zeros(1), ends[:-1], ends.new_full((1,), T_pad)]).to(torch.int32)
    return idx, inv, row.clamp(max=B - 1).to(torch.int32)[None], pos, cu


def make_pack(counts: torch.Tensor, total: int, M: int, max_count: int, multiple: int, index_fn=pack_index) -> Pack:
    """counts (B,) valid positions per row (device); total = sum(counts), max_count = max(counts) (host ints).
    index_fn: pack_index or its compiled version (Stage.compile_glue)."""
    T_pad = max(multiple, -(-total // multiple) * multiple)
    idx, inv, seq_idx, pos, cu = index_fn(counts, M, T_pad)
    max_len = -(-(max_count + T_pad - total) // multiple) * multiple
    return Pack(idx=idx, inv=inv, seq_idx=seq_idx, pos=pos, cu=cu, max_len=max_len, B=counts.shape[0], M=M)


def pack_rows(x: torch.Tensor, p: Pack) -> torch.Tensor:
    """(B, M, d) -> (1, T_pad, d)."""
    B, M, d = x.shape
    flat = torch.cat([x.reshape(B * M, d), x.new_zeros(1, d)])
    return flat[p.idx][None]


def unpack_rows(y: torch.Tensor, p: Pack) -> torch.Tensor:
    """(1, T_pad, d) -> (B, M, d); pad slots are 0."""
    d = y.shape[-1]
    return torch.cat([y[0], y.new_zeros(1, d)])[p.inv].view(p.B, p.M, d)
