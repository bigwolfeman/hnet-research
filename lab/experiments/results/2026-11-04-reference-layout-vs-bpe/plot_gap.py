"""Edu val bpb of the two reference-layout seeds against BPE on one bytes axis, and the gap.
Run from the repo root: uv run --with matplotlib --with numpy python <this dir>/plot_gap.py . <this dir>/gap.png > <this dir>/gap.txt
BPE bytes per eval = step x 12,582,912,000 / 180,080 (the planned bytes per BPE step; the curves file has no BPE bytes column).
"""
import csv, re, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = sys.argv[1]; out = sys.argv[2]
bpe = [(int(r["step"]) * 12582912000 / 180080, float(r["val/edu/bpb"])) for r in csv.DictReader(open(f"{R}/lab/experiments/results/2026-11-01-compute-ladder-vs-bpe/r2-eval-curves.csv")) if r["arm"] == "bpe_transformer"]
bpe = np.array(sorted(bpe))
def hnet(s):
    pts = [(int(m[1]) * 65536, float(m[2])) for m in re.finditer(r"EVAL main step (\d+): val bpb ([\d.]+)", open(f"{R}/ignored/runs/reference-layout-vs-bpe/reflayout_2stage_s{s}.log").read())]
    return np.array(pts)
h = {s: hnet(s) for s in (0, 1)}
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
a.plot(bpe[:, 0] / 1e9, bpe[:, 1], color="#0072B2", lw=2.2, label="BPE Transformer")
for s, ls in ((0, "-"), (1, "--")):
    a.plot(h[s][:, 0] / 1e9, h[s][:, 1], color="#E69F00", lw=2.2, ls=ls, label=f"H-Net reference layout, seed {s}")
    g = h[s][:, 1] - np.interp(h[s][:, 0], bpe[:, 0], bpe[:, 1])
    b.plot(h[s][:, 0] / 1e9, g, color="#E69F00", lw=2, ls=ls, label=f"seed {s}")
    print(f"seed {s}: crossing at", [f"{x/1e9:.2f}B" for x, g0, g1 in zip(h[s][1:, 0], g[:-1], g[1:]) if g0 > 0 >= g1], "last gap", round(g[-1], 4), "at", h[s][-1, 0] / 1e9)
    for B in (0.79, 1, 2, 3.15, 4, 6, 8, 10, 10.07, 11, 12, 12.58):
        i = np.argmin(abs(h[s][:, 0] - B * 1e9)); print(f"   {h[s][i,0]/1e9:6.2f}B gap {g[i]:+.4f}")
b.axhline(0, color="k", lw=1); b.axhline(-0.026, color="gray", lw=1, ls=":"); b.axhline(0.026, color="gray", lw=1, ls=":")
b.axvline(0.8 * 12.58, color="gray", lw=0.8, ls="-.")
a.axvline(0.8 * 12.58, color="gray", lw=0.8, ls="-.", label="LR decay starts (both)")
a.set_ylim(1.05, 1.6); a.set_ylabel("edu val bpb"); a.legend(); a.grid(alpha=0.3)
b.set_ylim(-0.1, 0.25); b.set_ylabel("H-Net − BPE (bpb)"); b.set_xlabel("training bytes (B)"); b.legend(); b.grid(alpha=0.3)
a.set_title("Reference layout vs BPE, edu val (dotted: ±0.026 grading band)")
fig.tight_layout(); fig.savefig(out, dpi=110)
