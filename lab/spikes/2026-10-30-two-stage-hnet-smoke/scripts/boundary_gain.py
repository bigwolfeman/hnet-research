"""Per-byte stage-1 shuffle cost on synthetic recall, split by whether the current word started inside the byte's chunk.

    python scripts/boundary_gain.py <run_dir> [...] --out cells.csv [--batches 8] [--stage 1] [--part all|read]

Experiments: lab/experiments/refuted/2026-11-03-recall-gain-by-boundary.md (stage 1, part all),
             lab/experiments/successes/2026-11-03-bytes-read-main-by-content.md (also stage 0, part read)

Pseudocode:
    batches = the trainer's fixed val batches (ByteLoader.fixed, seed 1234) of the run's synth domain
    normal pass  -> per-byte CE c_t (bits) for target y_t = x_{t+1}, stage-1 anchors (byte positions) per row
    shuffle pass -> stage S ablated "shuffle" (probes.ablated(stage S, part)): part "all" rolls the stage's
                    dechunked inner output (and any main_read) across rows; part "read" rolls the main_read alone
    cost_t = c'_t - c_t
    word = maximal run of letters a-z in the row; target y_t in word [s, e) has in-word position k = (t + 1) - s
    in-word chunk start at t  <=>  the last stage-1 anchor b <= t satisfies b >= s
    cell(k, cls) = mean cost (and mean c_t, column "bits") over targets with in-word position k and class cls
    D = mean over k of [cell(k, in_word) - cell(k, before_word)]
    recall data also: oracle value per byte = oracle_bits(reach=24) - oracle_bits(full), same cells
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet.data import ByteLoader  # noqa: E402
from hnet.model import LN2  # noqa: E402
from hnet.probes import ablated  # noqa: E402
from hnet.synth import SynthSpec, oracle_bits, read_lexicon  # noqa: E402
from sample import load  # noqa: E402

K = 3
LETTER = np.zeros(256, dtype=bool)
LETTER[ord("a"): ord("z") + 1] = True


@torch.no_grad()
def per_byte_bits(model, x, y):
    out = model(x)
    ce = F.cross_entropy(out.logits.float().transpose(1, 2), y, reduction="none") / LN2
    return ce.cpu().numpy(), out.aux


def word_start_of_target(row_x: np.ndarray, row_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each t: s = first letter index of the word holding target byte t+1 (-1 if the target is no letter);
    k = (t + 1) - s."""
    full = np.concatenate([row_x[:1], row_y])          # full[i] = byte i of the row (length L + 1)
    letter = LETTER[full]
    start = np.full(len(full), -1, dtype=np.int64)
    cur = -1
    for i in range(len(full)):
        if letter[i]:
            cur = i if cur < 0 else cur
            start[i] = cur
        else:
            cur = -1
    s = start[1:]                                       # target t+1
    k = np.where(s >= 0, np.arange(1, len(full)) - s, -1)          # -1: the target is a space or separator
    return s, k


def last_anchor_at_or_before(anchors: np.ndarray, L: int) -> np.ndarray:
    a = np.full(L, -1, dtype=np.int64)
    a[anchors[(anchors >= 0) & (anchors < L)]] = anchors[(anchors >= 0) & (anchors < L)]
    return np.maximum.accumulate(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--data-dir", type=Path, default=Path("~/data/hnet-bytes").expanduser())
    ap.add_argument("--stage", type=int, default=1, help="stage to ablate")
    ap.add_argument("--part", choices=("all", "read"), default="all", help="probes.ablated part")
    a = ap.parse_args()
    dev = torch.device("cuda")
    rows_out = []
    for run in a.runs:
        model, cfg = load(run, dev)
        src = cfg["data"]["train_sources"][0]["file"]            # synth/<variant>/train.bin
        variant = src.split("/")[1]
        meta = json.loads((a.data_dir / f"synth/{variant}/meta.json").read_text())
        spec = SynthSpec(**meta["spec"])
        lexicon = read_lexicon(a.data_dir / "synth/lexicon.txt", spec)
        tc = cfg["train"]
        loader = ByteLoader(str(a.data_dir / f"synth/{variant}/val.bin"), tc["seq_len"], tc["batch_size"], cfg["seed"], dev)
        target = list(model.stages())[a.stage]
        cost_all, cls_all, k_all, orc_all, bits_all = [], [], [], [], []
        for x, y in loader.fixed(a.batches):
            with torch.autocast("cuda", torch.bfloat16):
                c, aux = per_byte_bits(model, x, y)
                with ablated(target, "shuffle", part=a.part):
                    c2, _ = per_byte_bits(model, x, y)
            xs, ys = x.cpu().numpy().astype(np.uint8), y.cpu().numpy().astype(np.uint8)
            B, L = xs.shape
            for r in range(B):
                s, k = word_start_of_target(xs[r], ys[r])
                n = int(aux[1].counts[r])
                anch = aux[1].pos[r, :n].cpu().numpy().astype(np.int64)
                last = last_anchor_at_or_before(anch, L)
                cls = np.where(s < 0, -1, (last >= s).astype(np.int64))   # 1 in-word, 0 before-word, -1 not a letter
                cost_all.append(c2[r] - c[r]); cls_all.append(cls); k_all.append(k); bits_all.append(c[r])
                if variant == "recall":
                    full = np.concatenate([xs[r][:1], ys[r]])
                    if (full == 0).any():
                        v = oracle_bits(spec, lexicon, full, reach=24) - oracle_bits(spec, lexicon, full)
                        orc_all.append(v[1:])                              # bits of byte i -> target index i - 1
                    else:
                        orc_all.append(np.full(L, np.nan))
        cost, cls, k, bits = map(np.concatenate, (cost_all, cls_all, k_all, bits_all))
        orc = np.concatenate(orc_all) if orc_all else np.full_like(cost, np.nan)
        name = f"{run.parent.name}/{run.name}"   # unique per checkpoint (two arms may share variant and seed)
        D = []
        for kk in range(1, K + 1):
            cell = {}
            for cc, lab in ((1, "in_word"), (0, "before_word")):
                m = (k == kk) & (cls == cc)
                cell[lab] = float(cost[m].mean())
                o = orc[m]
                rows_out.append(dict(arm=name, k=kk, cls=lab, n=int(m.sum()), shuffle_cost=cell[lab],
                                     bits=float(bits[m].mean()), oracle_value=float(np.nanmean(o)) if np.isfinite(o).any() else float("nan")))
            D.append(cell["in_word"] - cell["before_word"])
        # descriptive (not graded): where the cost sits, by in-word position (k = 0 is a word's first letter,
        # -1 a space or separator target), all chunk-start classes pooled
        for kk in range(-1, 9):
            m = k == kk
            if m.any():
                rows_out.append(dict(arm=name, k=kk, cls="all", n=int(m.sum()), shuffle_cost=float(cost[m].mean()),
                                     bits=float(bits[m].mean()), oracle_value=float(np.nanmean(orc[m])) if np.isfinite(orc[m]).any() else float("nan")))
        print(f"{name}: in-word chunk starts at k=1..{K}: "
              + " ".join(str(int(((k == kk) & (cls == 1)).sum())) for kk in range(1, K + 1))
              + f" of " + " ".join(str(int((k == kk).sum())) for kk in range(1, K + 1)), flush=True)
        print(f"{name}: stage {a.stage} part {a.part}: mean shuffle cost {cost.mean():.4f} bits/byte  "
              f"mean bits {bits.mean():.4f}  D {np.mean(D):+.4f}  per k "
              + " ".join(f"{d:+.4f}" for d in D), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    for r in rows_out:
        print(r)


if __name__ == "__main__":
    main()
