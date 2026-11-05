"""Read-probe curve per eval for group byte-read-on-text (was the read used at any point in training?).

    python read_curve.py > read-curve.txt
"""
import wandb

api = wandb.Api()
for r in sorted(api.runs("hnet-router", filters={"group": "byte-read-on-text", "state": "finished"}), key=lambda r: r.name):
    h = [x for x in r.scan_history() if x.get("probe/read_zero_s0") is not None]
    print(f"{r.name} {r.id}: step read_zero read_shuffle edu")
    for x in h:
        print(f"   {x['_step']:6d} {x['probe/read_zero_s0']:+.4f} {x['probe/read_shuffle_s0']:+.4f} {x.get('val/edu/bpb') or float('nan'):.4f}")
