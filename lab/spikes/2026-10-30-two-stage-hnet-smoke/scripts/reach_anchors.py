"""Reach-limited oracle anchors for the synthetic recall language (hnet/synth.py oracle_bits(reach=R)).

    for R in reaches: bpb_R = mean oracle bits on the first anchor_bytes of recall/val.bin, where the memory holds
                      only the words that start in the last R bytes
    -> recall/meta.json["reach_bpb"] = {R: bpb_R}
bpb_R is the best a model can reach if every path that skips the main network sees at most R bytes back.
usage: python scripts/reach_anchors.py --dir $HNET_DATA_DIR/synth/recall --reach 24 48 96
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hnet.synth import SynthSpec, oracle_bits, read_lexicon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reach", type=int, nargs="+", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    meta = json.loads((d / "meta.json").read_text())
    spec = SynthSpec(**meta["spec"])
    lex = read_lexicon(d.parent / "lexicon.txt", spec)   # one lexicon for both languages (make_synth.py)
    val = np.fromfile(d / "val.bin", dtype=np.uint8)[: meta["anchor_bytes"]]
    full = float(np.nanmean(oracle_bits(spec, lex, val)))
    assert abs(full - meta["oracle_bpb"]) < 1e-9, f"oracle {full} differs from meta.json {meta['oracle_bpb']}"
    reach = dict(meta.get("reach_bpb", {}))
    for r in a.reach:
        reach[str(r)] = float(np.nanmean(oracle_bits(spec, lex, val, reach=r)))
        print(f"reach {r:6d} bytes: {reach[str(r)]:.4f} bpb", flush=True)
    meta["reach_bpb"] = dict(sorted(reach.items(), key=lambda kv: int(kv[0])))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
