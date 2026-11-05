import sys, csv, wandb
V = sys.argv[3] if len(sys.argv) > 3 else "edu"
api = wandb.Api()
runs = sorted(api.runs("hnet-router", filters={"group": sys.argv[1]}), key=lambda r: r.name)
rows = []
for r in runs:
    if r.state != "finished":
        print("skip", r.id, r.name, r.state); continue
    src = r.config.get("data", {}).get("train_sources", [{}])[0].get("file", "")
    tag = src.split("/")[1][:8] + "_" if src.startswith("synth/") else ""
    for h in r.scan_history():
        if h.get("probe/coarse_shuffle_s1") is None: continue
        d = {"arm": tag + r.name, "run_id": r.id, "_step": h["_step"]}
        d.update({k: v for k, v in h.items() if k.startswith(("probe/coarse","probe/align","val/","train/ratio","flops/")) and isinstance(v,(int,float))})
        rows.append(d)
keys = ["arm","run_id","_step"] + sorted({k for d in rows for k in d} - {"arm","run_id","_step"})
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
arms = sorted({d["arm"] for d in rows})
def table(fn, title):
    print(title); print("step  " + "  ".join(f"{a[-12:]:>12}" for a in arms))
    for s in sorted({d["_step"] for d in rows}):
        vals = []
        for a in arms:
            m = [d for d in rows if d["arm"]==a and d["_step"]==s]
            vals.append(f"{fn(m[0]):12.4f}" if m else " "*12)
        print(f"{s:5d} " + "  ".join(vals))
table(lambda d: d["probe/coarse_shuffle_s1"]-d["probe/coarse_zero_s1"], "s1 shuffle-zero")
table(lambda d: d["probe/coarse_shuffle_s0"]-d["probe/coarse_zero_s0"], "s0 shuffle-zero")
table(lambda d: d["probe/coarse_shuffle_s1"], "shuffle_s1")
table(lambda d: d[f"val/{V}/bpb"], f"{V} bpb")
table(lambda d: d[f"val/{V}/ratio_s0"], "ratio s0")
table(lambda d: d[f"val/{V}/ratio_s1"], "ratio s1")
