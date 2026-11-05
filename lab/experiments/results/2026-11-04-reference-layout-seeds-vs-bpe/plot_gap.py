"""Edu val bpb of H-Net seeds 0-2 against BPE seeds 0-1 on one bytes axis, and each H-Net seed's gap to the BPE seed mean.
Run from the repo root: uv run --with matplotlib --with numpy python <this dir>/plot_gap.py . <this dir>/gap.png > <this dir>/gap.txt
Eval lines: "EVAL main step N: val bpb X" (edu). H-Net bytes = step x 65,536; BPE bytes = step x 12,582,912,000 / 180,080.
BPE seed 0 comes from the compute-ladder curves file (it has no log in ignored/runs).
"""
import csv, re, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = sys.argv[1]; out = sys.argv[2]
BPE_B = 12582912000 / 180080
def log(path, bytes_per_step):
    t = open(f"{R}/{path}").read()
    return np.array([(int(m[1]) * bytes_per_step, float(m[2])) for m in re.finditer(r"EVAL main step (\d+): val bpb ([\d.]+)", t)])
bpe = {0: np.array(sorted((int(r["step"]) * BPE_B, float(r["val/edu/bpb"])) for r in csv.DictReader(open(f"{R}/lab/experiments/results/2026-11-01-compute-ladder-vs-bpe/r2-eval-curves.csv")) if r["arm"] == "bpe_transformer")),
       1: log("ignored/runs/reference-layout-seeds-vs-bpe/arms/bpe_transformer_s1.log", BPE_B)}
hn = {0: log("ignored/runs/reference-layout-vs-bpe/reflayout_2stage_s0.log", 65536),
      1: log("ignored/runs/reference-layout-vs-bpe/reflayout_2stage_s1.log", 65536),
      2: log("ignored/runs/reference-layout-seeds-vs-bpe/arms/reflayout_2stage_s2.log", 65536)}
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
LS = {0: "-", 1: "--", 2: ":"}
for s, c in bpe.items():
    a.plot(c[:, 0] / 1e9, c[:, 1], color="#0072B2", lw=2, ls=LS[s], label=f"BPE Transformer, seed {s}")
for s, c in hn.items():
    a.plot(c[:, 0] / 1e9, c[:, 1], color="#E69F00", lw=2, ls=LS[s], label=f"H-Net reference layout, seed {s}")
    for bs, bc in bpe.items():
        g = c[:, 1] - np.interp(c[:, 0], bc[:, 0], bc[:, 1])
        if bs == 0:
            b.plot(c[:, 0] / 1e9, g, color="#E69F00", lw=2, ls=LS[s], label=f"H-Net seed {s} − BPE seed 0")
        cross = [f"{x/1e9:.2f}B" for x, g0, g1 in zip(c[1:, 0], g[:-1], g[1:]) if g0 > 0 >= g1]
        up = [f"{x/1e9:.2f}B" for x, g0, g1 in zip(c[1:, 0], g[:-1], g[1:]) if g0 <= 0 < g1]
        print(f"H-Net s{s} vs BPE s{bs}: down-crossings {cross} up-crossings {up} final gap {g[-1]:+.4f}")
    for B in (0.79, 2, 3.15, 6, 8, 10.07, 12.58):
        i = np.argmin(abs(c[:, 0] - B * 1e9)); print(f"   {c[i,0]/1e9:6.2f}B bpb {c[i,1]:.4f}")
print("finals: H-Net", {s: round(c[-1, 1], 4) for s, c in hn.items()}, "BPE", {s: round(c[-1, 1], 4) for s, c in bpe.items()})
b.axhline(0, color="k", lw=1); b.axvline(0.8 * 12.58, color="gray", lw=0.8, ls="-.")
a.axvline(0.8 * 12.58, color="gray", lw=0.8, ls="-.", label="LR decay starts")
a.set_ylim(1.05, 1.6); a.set_ylabel("edu val bpb"); a.legend(fontsize=9); a.grid(alpha=0.3)
b.set_ylim(-0.1, 0.25); b.set_ylabel("H-Net − BPE seed 0 (bpb)"); b.set_xlabel("training bytes (B)"); b.legend(fontsize=9); b.grid(alpha=0.3)
a.set_title("Reference layout seeds vs BPE seeds, edu val")
fig.tight_layout(); fig.savefig(out, dpi=110)
