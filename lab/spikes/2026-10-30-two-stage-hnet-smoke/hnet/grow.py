"""Grow a hierarchy: map a trained N-stage HNetLM's weights into a fresh (N+1)-stage HNetLM.

The new stage goes directly above the main network. Every source weight keeps its role:

    source (N stages)                         target (N + 1 stages)
    embed.*, head.*                       ->  same key
    stage s < N   (net.<inner. x s>*)     ->  same key                 (encoder, decoder, router, residual, start)
    main          (net.<inner. x N>stack) ->  net.<inner. x (N+1)>stack
    stage N-1 pad_dim (d_{N-1} -> d_main) ->  new stage N's pad_dim   (requires d_new == d_{N-1})
    new stage N (encoder, decoder, router, residual_proj, start_vec)  -> keeps the target's fresh init

Pseudocode:
    for key in target: src_key = rename(key); if src_key in source: copy (shapes must match) else: must be in new stage
    every source key must be used exactly once (fail loud otherwise)

Experiment: lab/experiments/refuted/2026-11-03-grown-hierarchy-keeps-main.md
"""
from __future__ import annotations

import torch

from .model import HNetLM


def grow_state_dict(src: dict[str, torch.Tensor], src_stages: int, target: HNetLM) -> tuple[dict, list[str]]:
    """Returns (target state dict with source weights mapped in, list of target keys left at their fresh init)."""
    n = src_stages
    if target.n_stages != n + 1:
        raise ValueError(f"grow: target has {target.n_stages} stages, expected source stages + 1 = {n + 1}")
    dims = list(target.cfg["d_model"])
    if n >= 1 and dims[n] != dims[n - 1]:
        raise ValueError(f"grow: new stage width {dims[n]} must equal the outer stage width {dims[n - 1]} (pad_dim reuse)")
    src_main = "net." + "inner." * n
    tgt_main = "net." + "inner." * (n + 1)
    new_stage = "net." + "inner." * n                 # the inserted stage's own params (not its inner)
    src_pad = "net." + "inner." * (n - 1) + "pad_dim" if n >= 1 else None
    out = target.state_dict()
    used, fresh = set(), []
    for key in out:
        if key.startswith(tgt_main):
            s = src_main + key[len(tgt_main):]
        elif key == new_stage + "pad_dim" and src_pad is not None:
            s = src_pad
        elif key.startswith(new_stage):
            fresh.append(key)
            continue
        else:
            s = key
        if s not in src:
            raise KeyError(f"grow: target key {key!r} maps to source key {s!r}, which the source lacks")
        if src[s].shape != out[key].shape:
            raise ValueError(f"grow: {key!r} {tuple(out[key].shape)} vs source {s!r} {tuple(src[s].shape)}")
        out[key] = src[s].clone()
        used.add(s)
    unused = sorted(set(src) - used)
    if unused:
        raise KeyError(f"grow: source keys with no place in the target: {unused[:8]}{' ...' if len(unused) > 8 else ''}")
    return out, fresh
