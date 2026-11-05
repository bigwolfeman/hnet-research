"""Does the MainRead query see the current word? Descriptive diagnostic on synthetic recall checkpoints.

    python query_prefix.py <run_dir> [...] [--batches 4]

Pseudocode:
    for target byte at in-word letter k (scripts/boundary_gain.py word positions):
        a0 = last stage-0 anchor <= t   (the stage-1 position byte t reads holds the stage-0 encoder output at a0)
        prefix seen by that stage-1 position = number of word letters in [s, a0]  = max(0, a0 - s + 1)
    report, per k = 0..3: fraction of targets whose stage-1 position has seen >= 1 letter of the word
    MainRead size: rms of r (read output) against rms of z (EMA read + residual) at stage 1, all positions
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SPIKE = Path(__file__).resolve().parents[3] / "spikes/2026-10-30-two-stage-hnet-smoke"
sys.path.insert(0, str(SPIKE)); sys.path.insert(0, str(SPIKE / "scripts"))
from boundary_gain import last_anchor_at_or_before, word_start_of_target  # noqa: E402
from hnet.data import ByteLoader  # noqa: E402
from sample import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--data-dir", type=Path, default=Path("~/data/hnet-bytes").expanduser())
    a = ap.parse_args()
    dev = torch.device("cuda")
    for run in a.runs:
        model, cfg = load(run, dev)
        variant = cfg["data"]["train_sources"][0]["file"].split("/")[1]
        tc = cfg["train"]
        loader = ByteLoader(str(a.data_dir / f"synth/{variant}/val.bin"), tc["seq_len"], tc["batch_size"], cfg["seed"], dev)
        st1 = model.stages()[1]
        sizes = []
        if st1.main_read is not None:   # hook: rms of the read output and of the decoder input
            orig = st1._read_fn
            def hooked(e, y, ri, _o=orig):
                r = _o(e, y, ri); sizes.append(r.float().pow(2).mean().sqrt().item()); return r
            st1._read_fn = hooked
            h = st1.decoder.register_forward_pre_hook(lambda m, i: sizes.append(-i[0].float().pow(2).mean().sqrt().item()))
        seen = {k: [] for k in range(4)}
        for x, y in loader.fixed(a.batches):
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                out = model(x)
            xs, ys = x.cpu().numpy().astype(np.uint8), y.cpu().numpy().astype(np.uint8)
            for r in range(xs.shape[0]):
                s, k = word_start_of_target(xs[r], ys[r])
                n0 = int(out.aux[0].counts[r])
                a0 = last_anchor_at_or_before(out.aux[0].pos[r, :n0].cpu().numpy().astype(np.int64), xs.shape[1])
                for kk in range(4):
                    m = (k == kk) & (s >= 0)
                    seen[kk].append((a0[m] >= s[m]).astype(np.float64))
        frac = {kk: float(np.concatenate(v).mean()) for kk, v in seen.items()}
        line = f"{run.parent.name}/{run.name}: stage-1 query has seen >= 1 letter of the word, by k: " + \
               " ".join(f"k={kk} {f:.3f}" for kk, f in frac.items())
        if sizes:
            r_rms = np.mean([v for v in sizes if v >= 0]); z_rms = np.mean([-v for v in sizes if v < 0])
            line += f" | read rms {r_rms:.3f} vs decoder-input rms {z_rms:.3f} (o.weight rms {st1.main_read.o.weight.float().pow(2).mean().sqrt().item():.4f})"
            h.remove(); st1._read_fn = orig
        print(line, flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
