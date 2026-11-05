"""Do the learned chunks follow words, or only the space byte? Eval-time analysis of trained 2-stage checkpoints.

    python scripts/chunk_semantics.py <ckpt.pt or run_dir> [...] --out <dir> [--rows 128] [--len 2048]
    python scripts/chunk_semantics.py --wandb <run_id> [...] --out <dir>      # alignment probes over training

Experiment: lab/experiments/successes/2026-11-04-chunks-find-hidden-words.md

Pseudocode:
    rows  = edu val rows that start at a document start (hnet/data.py doc_starts)
    word position k[i] = offset of byte i inside its maximal non-whitespace run (-1 on whitespace)
    cut at index c = a stage chunk starts at byte c (probes.stage_cuts)
    edit: each 0x20 with ASCII letters on both sides is deleted with p = 0.5; join = first byte of the second word
    main:  join rate = mean cut[j] over joins j
           control(k) = mean cut[i] over bytes at position k inside intact (join-free) runs longer than k
           k is capped at K_CAP (positions 8 and up pool; amendment 2026-11-04 14:00)
           matched control = mean over joins of control(k_j); joins whose k has < 100 control bytes are dropped
    other 1: intact text, cut rate at k = 0, 1, 2 and pooled k >= 3
    other 2: intact text, for k = 1..4: mean byte loss (bits) where a chunk starts vs where none starts
    readings: the same at stage 1; words per stage-1 chunk; most frequent multi-word stage-1 chunks
    semantic score: hnet/semantics.py AUROC of the router p per level (word, hidden, morph, phrase, gotok)
    chunks.html: passages drawn as in the H-Net paper's Fig 4: stage-0 chunks as alternating blue / gray boxes, each
                 stage-1 chunk outlined in orange, hidden word starts (joins) underlined; one block per checkpoint
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet import probes  # noqa: E402
from hnet.data import ByteLoader, pad_rows  # noqa: E402
from hnet.evaluation import run_batches  # noqa: E402
from hnet.model import LN2  # noqa: E402
from hnet.semantics import K_CAP, build_levels, delete_spaces, semantic_report, word_pos  # noqa: E402
from hnet.data import doc_starts  # noqa: E402
from sample import load  # noqa: E402

MIN_CTRL = 100


def run_model(model, rows: list[np.ndarray], device, batch: int = 16):
    """Per row: (list of per-stage cut arrays (L+1,), byte loss in bits (L,), nan at byte 0)."""
    res = []
    for b in range(0, len(rows), batch):
        chunk = rows[b:b + batch]
        x, lengths = pad_rows(chunk, model.multiple, device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x, lengths)
        ce = F.cross_entropy(out.logits[:, :-1].float().transpose(1, 2), x[:, 1:].long(), reduction="none") / LN2
        for i, r in enumerate(chunk):
            loss = np.full(len(r), np.nan)
            loss[1:] = ce[i, :len(r) - 1].cpu().numpy()
            res.append(([probes.stage_cuts(a, i, len(r)) for a in out.aux], loss))
    return res


def run_ids(k: np.ndarray) -> np.ndarray:
    """Id of the non-whitespace run each byte is in (-1 on whitespace)."""
    starts = (k == 0).cumsum() - 1
    return np.where(k >= 0, starts, -1)


def join_vs_control(rows, joins, cuts_s) -> dict:
    """Main measure at one stage, on edited rows."""
    ctrl_hits, ctrl_n, join_k, join_cut = Counter(), Counter(), [], []
    for x, j, c in zip(rows, joins, cuts_s):
        k_raw = word_pos(x)
        rid = run_ids(k_raw)
        k = np.minimum(k_raw, K_CAP)
        dirty = np.zeros(rid.max() + 2, dtype=bool)
        dirty[rid[j]] = True
        length = np.bincount(rid[rid >= 0], minlength=rid.max() + 1)
        cut = c[:len(x)]
        ok = (k_raw >= 1) & ~j & (rid >= 0)
        ok &= ~dirty[rid] & (length[np.maximum(rid, 0)] > k_raw)
        for kk, hit in zip(k[ok], cut[ok]):
            ctrl_n[int(kk)] += 1
            ctrl_hits[int(kk)] += int(hit)
        join_k += k[j].tolist()
        join_cut += cut[j].tolist()
    kept = [(kk, h) for kk, h in zip(join_k, join_cut) if ctrl_n[kk] >= MIN_CTRL]
    ctrl = [ctrl_hits[kk] / ctrl_n[kk] for kk, _ in kept]
    by_k = {kk: {"join_rate": float(np.mean([h for k2, h in kept if k2 == kk])),
                 "n_join": sum(1 for k2, _ in kept if k2 == kk),
                 "control_rate": ctrl_hits[kk] / ctrl_n[kk], "n_control": ctrl_n[kk]}
            for kk in sorted({kk for kk, _ in kept}) if kk <= 10}
    return {"join_rate": float(np.mean([h for _, h in kept])), "matched_control_rate": float(np.mean(ctrl)),
            "n_joins": len(join_k), "n_joins_kept": len(kept), "dropped_frac": 1 - len(kept) / max(len(join_k), 1),
            "by_k": by_k}


def position_rates(rows, cuts_s) -> dict:
    hits, n = Counter(), Counter()
    for x, c in zip(rows, cuts_s):
        k = np.minimum(word_pos(x), 3)
        for kk in (-1, 0, 1, 2, 3):
            m = k == kk
            hits[kk] += int(c[:len(x)][m].sum())
            n[kk] += int(m.sum())
    return {("ws" if kk < 0 else f"k{kk}" if kk < 3 else "k3+"): hits[kk] / max(n[kk], 1) for kk in (-1, 0, 1, 2, 3)}


def surprise(rows, cuts_s, losses) -> dict:
    out = {}
    for kk in range(1, 5):
        on, off = [], []
        for x, c, loss in zip(rows, cuts_s, losses):
            m = word_pos(x) == kk
            cut = c[:len(x)]
            on.append(loss[m & cut]); off.append(loss[m & ~cut])
        on, off = np.concatenate(on), np.concatenate(off)
        out[f"k{kk}"] = {"bits_cut": float(np.nanmean(on)) if len(on) else math.nan,
                         "bits_nocut": float(np.nanmean(off)) if len(off) else math.nan,
                         "n_cut": int(len(on)), "n_nocut": int(len(off))}
    return out


def stage1_units(rows, cuts1, top: int = 40) -> dict:
    words, multi = Counter(), Counter()
    for x, c in zip(rows, cuts1):
        starts = word_pos(x) == 0
        edges = [0, *np.flatnonzero(c[:len(x)]).tolist(), len(x)]
        for a, b in zip(edges[:-1], edges[1:]):
            n = int(starts[a:b].sum())
            words[n] += 1
            if n >= 2:
                multi[bytes(x[a:b]).decode("utf-8", errors="replace")] += 1
    total = sum(words.values())
    return {"words_per_chunk": {str(n): v / total for n, v in sorted(words.items()) if n <= 6},
            "top_multiword": multi.most_common(top)}


CSS = """body{font-family:system-ui,sans-serif;margin:24px;max-width:1500px;color:#111}
.t{font-family:ui-monospace,monospace;font-size:15px;line-height:2.5;white-space:pre-wrap;word-break:break-all}
.o{border:2.5px solid #E69F00;border-radius:6px;padding:3px 1px;margin:0 2px;box-decoration-break:clone;-webkit-box-decoration-break:clone}
.a0,.a1{border:1px solid #555;border-radius:3px;padding:1px 0;margin:0 1px;box-decoration-break:clone;-webkit-box-decoration-break:clone}
.a0{background:#9ecae1}.a1{background:#f0f0f0}.w{color:#777}
.j{border-bottom:3px solid #08306b;font-weight:700}
h2{margin-top:36px;border-bottom:1px solid #ccc}h3{font-size:14px;margin:14px 0 2px;color:#444}"""
LEGEND = ('<p><span class="t"><span class="o"><span class="a0">stage-0</span><span class="a1">chunks</span></span></span>'
          ' blue / gray boxes = stage-0 chunks &nbsp; orange outline = one stage-1 chunk &nbsp; '
          '<span class="t"><span class="j">x</span></span> = hidden word start (a deleted space) &nbsp; '
          '<span class="t w">\u00b7</span> space &nbsp; <span class="t w">\u21b5</span> newline</p>')


def _utf8_len(lead: int) -> int:
    return 1 if lead < 0xC0 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4


def _chars(x: np.ndarray, join: np.ndarray | None, a: int, b: int) -> str:
    """Bytes a..b as display text. A multi-byte UTF-8 character shows at its lead byte; continuation bytes are empty."""
    out = []
    for i in range(a, b):
        v = int(x[i])
        if v == 32:
            ch = '<span class="w">\u00b7</span>'
        elif v == 10:
            ch = '<span class="w">\u21b5</span>\n'
        elif v == 9:
            ch = '<span class="w">\u2192</span>'
        elif 0x80 <= v < 0xC0:
            ch = ""
        else:
            ch = html.escape(bytes(x[i:i + _utf8_len(v)]).decode("utf-8", errors="replace"))
        out.append(f'<span class="j">{ch}</span>' if join is not None and join[i] and ch else ch)
    return "".join(out)


def chunk_html(x: np.ndarray, cuts: list[np.ndarray], join: np.ndarray | None = None) -> str:
    """Stage-0 chunks as alternating boxes, grouped into orange-outlined stage-1 chunks."""
    L = len(x)
    c1 = cuts[1][:L] if len(cuts) > 1 else np.zeros(L, dtype=bool)
    c0 = cuts[0][:L] | c1
    e1 = [0, *np.flatnonzero(c1).tolist(), L]
    parts, parity = [], 0
    for a1, b1 in zip(e1[:-1], e1[1:]):
        e0 = [a1, *(a1 + 1 + c for c in np.flatnonzero(c0[a1 + 1:b1]).tolist()), b1]
        inner = []
        for a0, b0 in zip(e0[:-1], e0[1:]):
            inner.append(f'<span class="a{parity}">{_chars(x, join, a0, b0)}</span>')
            parity ^= 1
        parts.append(f'<span class="o">{"".join(inner)}</span>')
    return f'<div class="t">{"".join(parts)}</div>'


def passages(a) -> dict[str, np.ndarray]:
    d = a.data_dir
    src = {"edu": d / "fineweb-edu/val.bin", "go": d / "code/go_val.bin", "sql": d / "code/sql_val.bin"}
    n = {"edu": 3, "go": 2, "sql": 1}
    return {f"{k} {i + 1}": r for k, p in src.items() for i, r in enumerate(doc_starts(p, n[k], a.render_len, seed=7))}


def write_page(path: Path, blocks: dict[str, dict[str, str]], title: str):
    body = [f"<h1>{html.escape(title)}</h1>", LEGEND]
    for passage, by_ckpt in blocks.items():
        body.append(f"<h2>{html.escape(passage)}</h2>")
        body += [f"<h3>{html.escape(name)}</h3>{snippet}" for name, snippet in by_ckpt.items()]
    path.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                    f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")


def analyze(path: Path, a, device, levels) -> tuple[dict, dict[str, str]]:
    model, cfg = load(path, device)
    loader = ByteLoader(str(a.data_dir / "fineweb-edu/val.bin"), cfg["train"]["seq_len"], cfg["train"]["batch_size"],
                        cfg["seed"], device)
    gate_bpb = run_batches(model, loader.fixed(a.eval_batches or cfg["train"]["eval_batches"]), device, torch.bfloat16)[0] / LN2
    rows = doc_starts(a.data_dir / "fineweb-edu/val.bin", a.rows, a.len, seed=0)
    rng = np.random.default_rng(0)
    edited, joins = zip(*(delete_spaces(r, rng) for r in rows))
    intact = run_model(model, rows, device)
    edit = run_model(model, list(edited), device)
    n_stage = len(intact[0][0])
    res = {"checkpoint": str(path), "gate_edu_bpb": gate_bpb,
           "cuts_per_byte_intact": [sum(int(r[0][s].sum()) for r in intact) / sum(map(len, rows)) for s in range(n_stage)]}
    for s in range(n_stage):
        res[f"s{s}"] = {
            "hidden_words": join_vs_control(list(edited), list(joins), [r[0][s] for r in edit]),
            "position_rates": position_rates(rows, [r[0][s] for r in intact]),
            "surprise": surprise(rows, [r[0][s] for r in intact], [r[1] for r in intact]),
        }
    res["stage1_units"] = stage1_units(rows, [r[0][1] for r in intact]) if n_stage > 1 else None
    res["semantic"] = semantic_report(model, levels, device)
    if n_stage > 1:
        both = sum(int((r[0][1] & r[0][0]).sum()) for r in intact)
        res["s1_cuts_on_s0_cuts"] = both / max(sum(int(r[0][1].sum()) for r in intact), 1)
    snippets = {}
    rng = np.random.default_rng(1)
    for name, x in passages(a).items():
        snippets[f"{name}, intact"] = chunk_html(x, run_model(model, [x], device)[0][0])
        if name.startswith("edu"):
            xe, je = delete_spaces(x, rng)
            snippets[f"{name}, spaces deleted"] = chunk_html(xe, run_model(model, [xe], device)[0][0], je)
    del model
    torch.cuda.empty_cache()
    return res, snippets


def wandb_align(run_ids_: list[str]) -> dict:
    import wandb
    api, keys = wandb.Api(), [f"probe/align_s{s}_{t}_f1" for s in (0, 1) for t in ("model", "stride")]
    out = {}
    for rid in run_ids_:
        run = api.run(f"hnet-router/{rid}")
        pts = [{"step": r.get("_step"), **{k: r.get(k) for k in keys}}
               for r in run.scan_history(keys=["_step", *keys]) if r.get(keys[0]) is not None]
        out[rid] = {"name": run.name, "points": pts}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", type=Path)
    ap.add_argument("--wandb", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("~/data/hnet-bytes").expanduser())
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gold-dir", type=Path, default=Path("~/data/hnet-bytes/gold").expanduser())
    ap.add_argument("--render-len", type=int, default=700)
    ap.add_argument("--eval-batches", type=int, default=None, help="gate batches (default: the training eval batches)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blocks: dict[str, dict[str, str]] = {}
    levels = build_levels(doc_starts(a.data_dir / "fineweb-edu/val.bin", a.rows, a.len, seed=3),
                          doc_starts(a.data_dir / "code/go_val.bin", a.rows // 2, a.len, seed=3), a.gold_dir,
                          morph_rows=doc_starts(a.data_dir / "fineweb-edu/val.bin", 4 * a.rows, a.len, seed=4))
    for path in a.checkpoints:
        res, snippets = analyze(path, a, device, levels)
        name = f"{path.parent.name}__{path.stem}" if path.suffix == ".pt" else path.name
        for passage, snippet in snippets.items():
            blocks.setdefault(passage, {})[name] = snippet
        (a.out / f"{name}.json").write_text(json.dumps(res, indent=1))
        h = res["s0"]["hidden_words"]
        print(f"{name}: bpb {res['gate_edu_bpb']:.4f} cuts/byte {[round(c, 4) for c in res['cuts_per_byte_intact']]} "
              f"s0 join {h['join_rate']:.4f} vs control {h['matched_control_rate']:.4f} "
              f"(joins {h['n_joins']}, dropped {h['dropped_frac']:.3%})", flush=True)
        print("   " + "  ".join(f"{k.removeprefix('sem/').removesuffix('_auroc')} {v:.3f}" for k, v in res["semantic"].items()
                               if k.endswith("_auroc")), flush=True)
    if blocks:
        write_page(a.out / "chunks.html", blocks, "Learned chunks, stage 0 and stage 1")
    if a.wandb:
        (a.out / "wandb-align.json").write_text(json.dumps(wandb_align(a.wandb), indent=1))
        print("wandb alignment curves written", flush=True)


if __name__ == "__main__":
    main()
