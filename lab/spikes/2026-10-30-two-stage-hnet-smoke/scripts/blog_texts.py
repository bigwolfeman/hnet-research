"""Our chunks on the Cartesia blog's own figure texts, against the blog H-Net's and LH-Net's chunks.

    python scripts/blog_texts.py <ckpt.pt or run_dir> [...] --figdata <dir> --out <dir>

Experiment: lab/experiments/planned/2026-11-05-chunks-match-blog-texts.md
Figure data: ignored/blogs-raw/figdata/ (README there: every chunk_starts list is UTF-8 byte offsets of chunk starts).

Pseudocode:
    texts  = fig10a (src English), fig10b (src Go), fig11 (all); readings also on the other fig10/fig12 texts
    ours   = per stage, cut indices from one forward per text (chunk_semantics.run_model, no prefix byte)
    blog   = models.hnet_stage{0,1}.chunk_starts ; lhnet = models.lhnet_stage{0,1}.chunk_starts (fig10 only)
    canon(c, x) = first non-whitespace byte at or after c if x[c] is whitespace, else c ; drop 0 and len
    F1(pred, ref) = pooled boundary F1 of canon cut sets
    main:    per seed, F1(ours s1, blog s1) vs F1(stride at our s1 bytes/chunk, blog s1) on English + Go
    other 1: fig11 "New words": chunks inside occurrence k of each word = 1 + our s0 cuts in (b0, b1)
    other 2: English, in-word canon cuts (non-whitespace on both sides): F1(ours s0, LH-Net s0) vs F1(ours s0, H-Net s0)
    gates:   n_bytes re-encode, blog s1 subset of s0, edu val bpb on the training eval batches, stride rate within 2%
    html:    per English / Go text, ours (each seed) vs blog H-Net vs LH-Net, drawn by chunk_semantics.chunk_html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnet import probes  # noqa: E402
from hnet.data import ByteLoader  # noqa: E402
from hnet.evaluation import run_batches  # noqa: E402
from hnet.model import LN2  # noqa: E402
from chunk_semantics import CSS, LEGEND, chunk_html, run_model  # noqa: E402
from sample import load  # noqa: E402

_WS = probes._SPACE
NEW_WORDS = ("borvexila", "mipralune", "velnorith")


def load_figdata(d: Path) -> dict[str, list[dict]]:
    out = {k: json.loads((d / f"{k}.json").read_text()) for k in ("fig10a_language", "fig10b_code", "fig11_icl", "fig12_ood_code")}
    for recs in out.values():
        for r in recs:
            r["x"] = np.frombuffer(r["text"].encode("utf-8"), dtype=np.uint8)
    return out


def canon(cuts, x: np.ndarray) -> set[int]:
    """Gap rule: a cut on whitespace moves to the next non-whitespace byte. Cuts at 0 and len(x) are dropped."""
    L, out = len(x), set()
    for c in cuts:
        c = int(c)
        while 0 < c < L and _WS[x[c]]:
            c += 1
        if 0 < c < L:
            out.add(c)
    return out


def in_word(cs: set[int], x: np.ndarray) -> set[int]:
    return {c for c in cs if not _WS[x[c - 1]] and not _WS[x[c]]}


def f1(pred: list[set[int]], ref: list[set[int]]) -> dict:
    tp = sum(len(p & r) for p, r in zip(pred, ref))
    npred, nref = sum(map(len, pred)), sum(map(len, ref))
    p, r = tp / max(npred, 1), tp / max(nref, 1)
    return {"f1": 2 * p * r / max(p + r, 1e-9), "precision": p, "recall": r, "n_pred": npred, "n_ref": nref}


def stride(L: int, r: float) -> list[int]:
    return [int(round(k * r)) for k in range(1, int(L / r) + 1)]


def word_starts(x: np.ndarray) -> list[int]:
    return [i for i in range(1, len(x)) if _WS[x[i - 1]] and not _WS[x[i]]]


def bool_cuts(starts, L: int) -> np.ndarray:
    a = np.zeros(L + 1, dtype=bool)
    s = np.asarray([c for c in starts if 0 < c < L], dtype=int)
    a[s] = True
    return a


def gates(fig: dict) -> dict:
    for recs in fig.values():
        for r in recs:
            assert len(r["x"]) == r["n_bytes"], (r["id"], len(r["x"]), r["n_bytes"])
            m = r["models"]
            assert set(m["hnet_stage1"]["chunk_starts"]) <= set(m["hnet_stage0"]["chunk_starts"]), r["id"]
    return {"n_bytes_ok": True, "blog_s1_subset_s0": True}


def ours(model, recs: list[dict], device) -> list[list[np.ndarray]]:
    return [res[0] for res in run_model(model, [r["x"] for r in recs], device, batch=8)]


def occurrence_counts(rec: dict, cuts0: np.ndarray) -> dict[str, list[int]]:
    text, out = rec["text"], {}
    for w in NEW_WORDS:
        counts = []
        for m in re.finditer(rf"\b{re.escape(w)}\b", text):
            b0 = len(text[: m.start()].encode())
            b1 = b0 + len(w.encode())
            counts.append(1 + int(cuts0[b0 + 1: b1].sum()))
        if counts:
            out[w] = counts
    return out


def blog_occurrence_counts(rec: dict) -> dict[str, list[int]]:
    return occurrence_counts(rec, bool_cuts(rec["models"]["hnet_stage0"]["chunk_starts"], len(rec["x"])))


def analyze(path: Path, fig: dict, a, device) -> tuple[dict, dict[str, str]]:
    model, cfg = load(path, device)
    loader = ByteLoader(str(a.data_dir / "fineweb-edu/val.bin"), cfg["train"]["seq_len"], cfg["train"]["batch_size"],
                        cfg["seed"], device)
    gate_bpb = run_batches(model, loader.fixed(cfg["train"]["eval_batches"]), device, torch.bfloat16)[0] / LN2
    eng = [r for r in fig["fig10a_language"] if r["src"] == "English"]
    go = [r for r in fig["fig10b_code"] if r["src"] == "Go"]
    assert len(eng) == 3 and len(go) == 3, (len(eng), len(go))
    res = {"checkpoint": str(path), "gate_edu_bpb": gate_bpb}

    def score(recs: list[dict], cuts: list[list[np.ndarray]]) -> dict:
        xs = [r["x"] for r in recs]
        out = {}
        for s in (0, 1):
            mine = [canon(np.flatnonzero(c[s]), x) for c, x in zip(cuts, xs)]
            blog = [canon(r["models"][f"hnet_stage{s}"]["chunk_starts"], x) for r, x in zip(recs, xs)]
            n_ours = sum(int(c[s][1:len(x)].sum()) for c, x in zip(cuts, xs))
            bpc = sum(map(len, xs)) / max(n_ours + len(xs), 1)          # bytes per chunk (each text opens a chunk)
            st = [canon(stride(len(x), bpc), x) for x in xs]
            st_rate = sum(len(stride(len(x), bpc)) + 1 for x in xs) / (n_ours + len(xs))
            out[f"s{s}"] = {"ours": f1(mine, blog), "stride": f1(st, blog),
                            "word_chunker": f1([canon(word_starts(x), x) for x in xs], blog),
                            "ours_bytes_per_chunk": bpc, "stride_rate_over_ours": st_rate,
                            "blog_bytes_per_chunk": sum(map(len, xs)) / sum(len(r["models"][f"hnet_stage{s}"]["chunk_starts"]) for r in recs)}
            if "lhnet_stage0" in recs[0]["models"]:
                lh = [canon(r["models"][f"lhnet_stage{s}"]["chunk_starts"], x) for r, x in zip(recs, xs)]
                out[f"s{s}"]["ours_vs_lhnet"] = f1(mine, lh)
                out[f"s{s}"]["lhnet_vs_blog"] = f1(lh, blog)
                if s == 0:
                    iw = lambda sets: [in_word(c, x) for c, x in zip(sets, xs)]  # noqa: E731
                    out["in_word_s0"] = {"ours_vs_lhnet": f1(iw(mine), iw(lh)), "ours_vs_hnet": f1(iw(mine), iw(blog))}
        return out

    cut = {"English": ours(model, eng, device), "Go": ours(model, go, device)}
    res["main_pooled"] = score(eng + go, cut["English"] + cut["Go"])
    res["by_language"] = {"English": score(eng, cut["English"]), "Go": score(go, cut["Go"])}
    others = {}
    for key in ("fig10a_language", "fig10b_code"):
        for src in sorted({r["src"] for r in fig[key]} - {"English", "Go"}):
            recs = [r for r in fig[key] if r["src"] == src]
            others[f"{key}:{src}"] = score(recs, ours(model, recs, device))
    res["readings_other_languages"] = others
    icl = fig["fig11_icl"]
    icl_cuts = ours(model, icl, device)
    res["fig11"] = {}
    for r, c in zip(icl, icl_cuts):
        L = len(r["x"])
        e = {"chunks_ours": [int(c[s][1:L].sum()) + 1 for s in (0, 1)],
             "chunks_blog": [len(r["models"][f"hnet_stage{s}"]["chunk_starts"]) for s in (0, 1)]}
        occ = occurrence_counts(r, c[0])
        if occ:
            e["occurrences_s0_ours"] = occ
            e["occurrences_s0_blog"] = blog_occurrence_counts(r)
        res["fig11"][r["name"]] = e
    snippets = {}
    for src, recs in (("English", eng), ("Go", go), ("fig11", icl)):
        for r, c in zip(recs, cut.get(src) or icl_cuts):
            snippets[f"{src}: {r['name']}"] = chunk_html(r["x"], c)
    del model
    torch.cuda.empty_cache()
    return res, snippets


def blog_snippets(fig: dict) -> dict[str, dict[str, str]]:
    out = {}
    for src, recs in (("English", [r for r in fig["fig10a_language"] if r["src"] == "English"]),
                      ("Go", [r for r in fig["fig10b_code"] if r["src"] == "Go"]), ("fig11", fig["fig11_icl"])):
        for r in recs:
            L, m = len(r["x"]), r["models"]
            d = {"blog H-Net": chunk_html(r["x"], [bool_cuts(m["hnet_stage0"]["chunk_starts"], L),
                                                   bool_cuts(m["hnet_stage1"]["chunk_starts"], L)])}
            if "lhnet_stage0" in m:
                d["blog LH-Net (2025)"] = chunk_html(r["x"], [bool_cuts(m["lhnet_stage0"]["chunk_starts"], L),
                                                              bool_cuts(m["lhnet_stage1"]["chunk_starts"], L)])
            out[f"{src}: {r['name']}"] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--figdata", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("~/data/hnet-bytes").expanduser())
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    a.out.mkdir(parents=True, exist_ok=True)
    fig = load_figdata(a.figdata)
    summary = {"gates": gates(fig), "runs": {}}
    page = blog_snippets(fig)
    for p in a.checkpoints:
        name = f"{p.parent.name}/{p.stem}" if p.is_file() else p.name
        res, snips = analyze(p, fig, a, device)
        summary["runs"][name] = res
        (a.out / f"{name.replace('/', '__')}.json").write_text(json.dumps(res, indent=1))
        for k, v in snips.items():
            page.setdefault(k, {})[f"ours: {name}"] = v
        m = res["main_pooled"]["s1"]
        print(f"{name}: bpb {res['gate_edu_bpb']:.4f} s1 F1 ours {m['ours']['f1']:.3f} stride {m['stride']['f1']:.3f} "
              f"word {m['word_chunker']['f1']:.3f} (ours {m['ours_bytes_per_chunk']:.2f} B/chunk, blog "
              f"{m['blog_bytes_per_chunk']:.2f}, stride rate x{m['stride_rate_over_ours']:.3f})", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    body = [f"<h1>Our chunks vs the blog's, on the blog's texts</h1>", LEGEND]
    for k, by in page.items():
        body.append(f"<h2>{html.escape(k)}</h2>")
        body += [f"<h3>{html.escape(n)}</h3>{s}" for n, s in by.items()]
    (a.out / "blog_texts.html").write_text("<!DOCTYPE html><html><head><meta charset='utf-8'><title>blog texts</title>"
                                           f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")


if __name__ == "__main__":
    main()
