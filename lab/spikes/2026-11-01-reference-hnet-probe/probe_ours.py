"""Coarse-path probe on one of OUR checkpoints. Run with the two-stage spike's venv (its `hnet` package).

Pseudocode:
    model <- build_model(ckpt.config.model), ckpt.model ; eager, bf16 autocast (as the training eval)
    batches <- probe_data.probe_batches(edu val)                   # the training probe batches
    rise[s, mode] <- probes.coarse_use(model, batches)              # the training probe itself
    share[s] <- mean over positions of ||up|| / ||resid|| at stage s's decoder input   (up = dechunked inner output)
    write JSON

usage: <two-stage venv>/bin/python probe_ours.py <final.pt> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "lab" / "spikes" / "2026-10-30-two-stage-hnet-smoke"))
sys.path.insert(0, str(HERE))

from hnet import probes  # noqa: E402
from hnet.chunking import ema_scan, gather_rows  # noqa: E402
from hnet.evaluation import run_batches  # noqa: E402
from hnet.model import LN2, build_model  # noqa: E402
from hnet.stage import Stage  # noqa: E402
from probe_data import EDU_VAL, probe_batches  # noqa: E402


def share_recorder(stage: Stage, sink: list):
    """A _scatter_out replica that also records ||up|| / ||resid|| per position. Asserts it matches the original."""
    def scatter(y, p_c, read_idx, conf, resid, dtype):
        z_ref = Stage._scatter_out(stage, y, p_c, read_idx, conf, resid, dtype)
        y = y[..., : stage.d]
        with torch.autocast(device_type=y.device.type, enabled=False):
            ema = ema_scan(y.float(), p_c)
            up = gather_rows(ema, read_idx.clamp(0, ema.shape[1] - 1))
            if stage.start_vec is not None:
                up = torch.where((read_idx < 0).unsqueeze(-1), stage.start_vec.float().expand_as(up), up)
            up = up * torch.ones_like(conf).unsqueeze(-1)          # ste(conf) is 1 in the forward pass
            assert torch.equal((up + resid).to(dtype), z_ref), "share replica drifted from Stage._scatter_out"
            sink.append((up.norm(dim=-1) / resid.norm(dim=-1).clamp_min(1e-12)).mean().item())
        return z_ref
    return scatter


def main(ckpt_path: str, out_path: str):
    dev = torch.device("cuda")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["config"]["model"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    batches = [(torch.from_numpy(b[:, :-1]).to(dev), torch.from_numpy(b[:, 1:]).to(dev))
               for b in probe_batches(ROOT / EDU_VAL)]
    amp = torch.bfloat16
    with torch.no_grad():
        m = probes.coarse_use(model, batches, dev, amp)
        shares = {}
        for s, st in enumerate(model.stages()):
            sink: list[float] = []
            st._scatter_fn = share_recorder(st, sink)
            ce, anchors, total = run_batches(model, batches, dev, amp)
            st._scatter_fn = st._scatter_out
            shares[s] = float(np.mean(sink))
            assert abs(ce / LN2 - m["probe/coarse_base_bpb"]) < 1e-4, "share pass changed the model output"
    _, anchors, total = run_batches(model, batches, dev, amp)
    cum = [total / a for a in anchors]
    res = dict(model=str(ckpt_path), base_bpb=m["probe/coarse_base_bpb"],
               rise={s: {mode: m[f"probe/coarse_{mode}_s{s}"] for mode in ("zero", "shuffle")}
                     for s in range(len(model.stages()))},
               share=shares, bytes_per_chunk=cum, ratios=[cum[0]] + [cum[i] / cum[i - 1] for i in range(1, len(cum))])
    Path(out_path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
