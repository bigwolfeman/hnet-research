"""Pull grade, gates, and probes for group byte-read-on-text and its plain control from wandb.

    python pull.py > summary.txt     (writes val-curves.csv next to this file)

Grade: val/edu/bpb at step 12,000. Control: the edu arms of main-gradient-coherence (reflayout_2stage_s0..4).
Fate of stage 1 = probe/coarse_shuffle_s1 - probe/coarse_zero_s1 (> 0: live).
Read use = probe/read_shuffle_s0 - probe/read_zero_s0 (> 0: the read carries document-specific content).
"""
import csv
from pathlib import Path

import wandb

PLAIN = ["akqhdkkz", "ipcxldpl", "zplsyk1o", "5d3o336d", "7dffta9c"]
api = wandb.Api()
read = sorted(api.runs("hnet-router", filters={"group": "byte-read-on-text", "state": "finished"}),
              key=lambda r: r.name)
plain = [api.run(f"hnet-router/{i}") for i in PLAIN]
rows = []
for arm, runs in (("plain", plain), ("read", read)):
    for r in runs:
        s = r.summary
        for x in r.scan_history(keys=["val/edu/bpb", "val/go/bpb"]):
            rows.append(dict(arm=arm, run=r.name, id=r.id, step=x["_step"], edu=x["val/edu/bpb"], go=x["val/go/bpb"]))
        fate = s["probe/coarse_shuffle_s1"] - s["probe/coarse_zero_s1"]
        line = (f"{arm:5s} {r.name:20s} {r.id} edu={s['val/edu/bpb']:.4f} go={s['val/go/bpb']:.4f} "
                f"fate_s1={fate:+.4f} {'LIVE' if fate > 0 else 'dead'} coarse_shuf_s0={s['probe/coarse_shuffle_s0']:.4f}")
        if arm == "read":
            z, sh = s["probe/read_zero_s0"], s["probe/read_shuffle_s0"]
            line += f" read_zero_s0={z:.4f} read_shuf_s0={sh:.4f} read_use={sh - z:+.4f}"
        print(line)
        print(f"      gates: ratios={s.get('final/ratios')} flops/byte={s.get('flops/fwd_per_byte')} "
              f"bytes={s.get('train/bytes')} params={s.get('final/params')} loss={s.get('train/loss')} step={s.get('_step')}")
out = Path(__file__).with_name("val-curves.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
