"""Build the synthetic recall and no-recall byte datasets (hnet/synth.py) and their fixed bpb anchors.

Pseudocode:
    lexicon = make_lexicon(seed)                              # one lexicon for both datasets: the same spelling job
    for name, doc_vocab in [(recall, 32), (norecall, None)]:
        train.bin = generate(train_bytes), val.bin = generate(val_bytes)       # different rng streams
        meta.json = spec + anchors on the first anchor_bytes of val.bin (oracle, no-memory oracle)

usage: python scripts/make_synth.py --out $HNET_DATA_DIR/synth [--train-bytes 400000000]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hnet.synth import SynthSpec, anchors, generate, make_lexicon, write_lexicon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-bytes", type=int, default=400_000_000)
    ap.add_argument("--val-bytes", type=int, default=8_000_000)
    ap.add_argument("--anchor-bytes", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = SynthSpec()
    lex = make_lexicon(base, np.random.default_rng(a.seed))
    write_lexicon(out / "lexicon.txt", lex)
    for i, (name, vocab) in enumerate([("recall", base.doc_vocab), ("norecall", None)]):
        spec = dataclasses.replace(base, doc_vocab=vocab)
        d = out / name
        d.mkdir(exist_ok=True)
        train = generate(spec, lex, a.train_bytes, np.random.default_rng([a.seed, i, 1]))
        val = generate(spec, lex, a.val_bytes, np.random.default_rng([a.seed, i, 2]))
        train.tofile(d / "train.bin")
        val.tofile(d / "val.bin")
        meta = dict(spec=dataclasses.asdict(spec), seed=a.seed, train_bytes=len(train), val_bytes=len(val),
                    anchor_bytes=a.anchor_bytes, **anchors(spec, lex, val[: a.anchor_bytes]))
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(name, meta, flush=True)


if __name__ == "__main__":
    main()
