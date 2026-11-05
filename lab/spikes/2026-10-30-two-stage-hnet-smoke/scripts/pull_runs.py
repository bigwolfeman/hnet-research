"""Pull grades, coarse-path fates, and gates for finished wandb runs (one line per run, plus a gates line).

    python scripts/pull_runs.py --group G [--group G2 ...] [--id ID ...] [--curves out.csv] > summary.txt

Pseudocode:
    runs = finished runs of each --group (sorted by name), then each --id, in the order given
    per run: val/{edu,go,sql}/bpb, and for each stage s with probes:
             fate_s = probe/coarse_shuffle_s{s} - probe/coarse_zero_s{s}   (> 0: live, the anchor is 0)
    gates line: params, ratios, fwd FLOPs/byte, bytes, final loss, last step
    --curves: every eval row of val/edu/bpb and val/go/bpb for every run, as CSV

Experiment results dirs call this from a pull.sh with the exact arguments.
"""
import argparse
import csv
import math

import wandb

PROJECT = "hnet-router"


def fates(s) -> str:
    out = []
    for st in range(4):
        z, sh = s.get(f"probe/coarse_zero_s{st}"), s.get(f"probe/coarse_shuffle_s{st}")
        if z is None or sh is None:
            break
        f = sh - z
        out.append(f"s{st}: zero={z:.4f} shuf={sh:.4f} fate={f:+.4f} {'LIVE' if f > 0 else 'dead'}")
    return " | ".join(out)


def fmt(x) -> str:
    return "None" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[])
    ap.add_argument("--id", action="append", default=[])
    ap.add_argument("--curves")
    a = ap.parse_args()
    api = wandb.Api()
    runs = []
    for g in a.group:
        got = sorted(api.runs(PROJECT, filters={"group": g, "state": "finished"}), key=lambda r: r.name)
        if not got:
            raise SystemExit(f"no finished runs in group {g!r}")
        runs += [(g, r) for r in got]
    runs += [("id", api.run(f"{PROJECT}/{i}")) for i in a.id]
    rows = []
    for g, r in runs:
        s = r.summary
        loss = s.get("train/loss")
        if loss is None or not math.isfinite(loss):
            raise SystemExit(f"{r.name} {r.id}: train/loss {loss!r} is not finite")
        print(f"{g:22s} {r.name:24s} {r.id} edu={fmt(s.get('val/edu/bpb'))} go={fmt(s.get('val/go/bpb'))} "
              f"sql={fmt(s.get('val/sql/bpb'))}  {fates(s)}")
        print(f"    gates: params={s.get('final/params')} ratios={s.get('final/ratios')} "
              f"flops/byte={s.get('flops/fwd_per_byte')} bytes={s.get('train/bytes')} loss={loss:.4f} "
              f"step={s.get('_step')} peak_mem_gb={fmt(s.get('final/peak_mem_gb'))}")
        if a.curves:
            for x in r.scan_history(keys=["val/edu/bpb", "val/go/bpb"]):
                rows.append(dict(group=g, run=r.name, id=r.id, step=x["_step"], edu=x["val/edu/bpb"], go=x["val/go/bpb"]))
    if a.curves:
        with open(a.curves, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
