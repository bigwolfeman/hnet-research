"""Coarse-path probe on a REFERENCE checkpoint (goombalab/hnet). Run with this spike's venv.

Pseudocode:
    model <- refhnet.load_ref(name)                                 # bf16, mask mode
    batches <- probe_data.probe_batches(edu val); separator 0 -> BOS 254 in inputs and targets
    stage s = refhnet.ref_stages(model)[s] ; its residual_func(out, resid, p) = out * STE(p) + resid
        out = dechunked output of the network inside stage s (the coarse path)
    base = mean over batches of CE / ln 2
    rise[s, zero]    = bpb with out = 0 at stage s       - base
    rise[s, shuffle] = bpb with out rolled by one row    - base     (another document's coarse path)
    share[s] = mean over positions of ||out|| / ||resid|| (STE(p) is 1 in the forward pass)
    ratios: per-stage boundary fraction over all positions
    write JSON

usage: .venv/bin/python probe_ref.py <hnet_2stage_L|hnet_1stage_L|...> <out.json>
"""
from __future__ import annotations

import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import refhnet as R
from probe_data import EDU_VAL, probe_batches


@contextmanager
def patched(stage, mode: str | None, sink: list | None = None):
    orig = stage.residual_func

    def f(out, residual, p):
        if mode == "zero":
            out = torch.zeros_like(out)
        elif mode == "shuffle":
            out = out.roll(1, dims=0)
        elif mode is not None:
            raise ValueError(mode)
        if sink is not None:
            sink.append((out.float().norm(dim=-1) / residual.float().norm(dim=-1).clamp_min(1e-12)).mean().item())
        return orig(out, residual, p)

    stage.residual_func = f
    try:
        yield
    finally:
        stage.residual_func = orig


@torch.no_grad()
def bpb(model, batches) -> tuple[float, list[float]]:
    ces, bounds, total = [], None, 0
    for x, y in batches:
        logits, bp = R.ref_forward(model, x)
        ces.append(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), R.to_ref_bytes(y).reshape(-1)).item())
        b = np.array([int(o.boundary_mask.sum()) for o in bp], dtype=np.int64)
        bounds = b if bounds is None else bounds + b
        total += x.numel()
    assert all(math.isfinite(c) for c in ces)
    return float(np.mean(ces)) / math.log(2), [total / int(b) for b in bounds]


def main(name: str, out_path: str):
    dev = torch.device("cuda")
    model = R.load_ref(name, dev)
    batches = [(torch.from_numpy(b[:, :-1]).to(dev), torch.from_numpy(b[:, 1:]).to(dev))
               for b in probe_batches(R.ROOT / EDU_VAL)]
    base, cum = bpb(model, batches)
    stages = R.ref_stages(model)
    rise, share = {}, {}
    for s, st in enumerate(stages):
        rise[s] = {}
        for mode in ("zero", "shuffle"):
            with patched(st, mode):
                rise[s][mode] = bpb(model, batches)[0] - base
        sink: list[float] = []
        with patched(st, None, sink):
            again = bpb(model, batches)[0]
        assert abs(again - base) < 1e-6, "the recording pass changed the output"
        share[s] = float(np.mean(sink))
        print(f"stage {s}: rise {rise[s]} share {share[s]:.4f}", flush=True)
    res = dict(model=name, base_bpb=base, rise=rise, share=share, bytes_per_chunk=cum,
               ratios=[cum[0]] + [cum[i] / cum[i - 1] for i in range(1, len(cum))])
    Path(out_path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
