"""Pull grad stats, probes, and gates for group main-gradient-coherence from wandb.

    python pull.py > summary.txt     (writes grad-curves.csv next to this file)

Per run: fate = probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at the last probe step (> 0: live).
Grade window: mean of train/grad/update_rms_l{L} over logged steps lo..hi (edu 500..4000, synth 500..3000).
"""
import csv
from pathlib import Path

import numpy as np
import wandb

KEYS = [f"train/grad/{m}_l{L}" for m in ("update_rms", "norm") for L in (0, 1, 2)]
api = wandb.Api()
runs = sorted(api.runs("hnet-router", filters={"group": "main-gradient-coherence", "state": "finished"}),
              key=lambda r: (r.config["data"]["train_sources"][0]["file"].startswith("synth"), r.name))
rows = []
for r in runs:
    synth = r.config["data"]["train_sources"][0]["file"].startswith("synth")
    lo, hi = (500, 3000) if synth else (500, 4000)
    h = [x for x in r.scan_history() if x.get("train/grad/update_rms_l2") is not None]
    for x in h:
        rows.append(dict(run=r.name, id=r.id, step=x["_step"], **{k: x[k] for k in KEYS}))
    win = [x for x in h if lo <= x["_step"] <= hi]
    s = r.summary
    zero, shuf = s["probe/coarse_zero_s1"], s["probe/coarse_shuffle_s1"]
    fate = shuf - zero
    means = {L: float(np.mean([x[f"train/grad/update_rms_l{L}"] for x in win])) for L in (0, 1, 2)}
    nmeans = {L: float(np.mean([x[f"train/grad/norm_l{L}"] for x in win])) for L in (0, 1, 2)}
    dom = "recall" if synth else "edu"
    print(f"{r.name:22s} {r.id} {'synth' if synth else 'edu  '} fate(shuf-zero s1)={fate:+.4f} "
          f"{'LIVE' if fate > 0 else 'dead'}  zero_s1={zero:.4f} shuf_s1={shuf:.4f} shuf_s0={s['probe/coarse_shuffle_s0']:.4f}  "
          f"urms[{lo}..{hi}] l0={means[0]:.4f} l1={means[1]:.4f} l2={means[2]:.4f}  "
          f"gnorm l0={nmeans[0]:.4f} l1={nmeans[1]:.4f} l2={nmeans[2]:.4f}  n={len(win)} logged={len(h)}")
    print(f"    gates: val/{dom}/bpb={s.get(f'val/{dom}/bpb')} ratios={s.get('final/ratios')} "
          f"flops/byte={s.get('flops/fwd_per_byte')} bytes={s.get('train/bytes')} params={s.get('final/params')} "
          f"loss={s.get('train/loss')} step={s.get('_step')}")
out = Path(__file__).with_name("grad-curves.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
