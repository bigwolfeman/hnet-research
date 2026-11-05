import sys, wandb
api = wandb.Api()
for r in sorted(api.runs("hnet-router", filters={"group": sys.argv[1]}), key=lambda r: r.name):
    ks = ["train/jepa/jepa/loss","train/jepa/jepa/pred_cos","train/jepa/jepa/z_cos_same_pair","train/jepa/jepa/z_cos_other_pair"]
    rows = [h for h in r.scan_history() if any(k.startswith("train/jepa") for k in h)]
    if not rows: print(r.name, r.state, "no jepa keys"); continue
    keys = sorted({k for h in rows for k in h if k.startswith("train/jepa")})
    last = max(h["_step"] for h in rows)
    tail = [h for h in rows if h["_step"] > last - 1000]
    head = [h for h in rows if h["_step"] <= 500]
    print(r.name, r.id, r.state, "last step", last, "n tail", len(tail))
    for k in keys:
        f = lambda hs: sum(h[k] for h in hs if h.get(k) is not None) / max(1, sum(h.get(k) is not None for h in hs))
        print(f"  {k:40s} first500 {f(head):.4f}  last1000 {f(tail):.4f}")
