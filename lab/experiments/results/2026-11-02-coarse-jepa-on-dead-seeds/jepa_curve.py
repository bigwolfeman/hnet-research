import sys, wandb
api = wandb.Api()
groups = sys.argv[2:]
for g in groups:
    for r in sorted(api.runs(sys.argv[1], filters={"group": g}), key=lambda r: r.name):
        rows = [h for h in r.scan_history() if h.get("train/jepa/sigreg") is not None]
        print(g, r.name, r.id, r.state, len(rows))
        for lo in range(0, rows[-1]["_step"] + 1 if rows else 0, 200):
            b = [h for h in rows if lo <= h["_step"] < lo + 200]
            if not b: continue
            m = lambda k: sum(h[k] for h in b) / len(b)
            print(f"  {lo:5d}-{lo+199:5d} sigreg {m('train/jepa/sigreg'):7.3f} inv {m('train/jepa/inv'):.4f} top1 {m('train/jepa/top1_doc_given_code'):.3f} same {m('train/jepa/z_cos_same_pair'):+.3f} other {m('train/jepa/z_cos_other_pair'):+.3f}")
