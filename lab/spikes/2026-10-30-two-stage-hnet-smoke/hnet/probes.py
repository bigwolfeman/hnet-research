"""Hierarchy probes (Principle 3). All run inside training as eval hooks.

Pseudocode:
    cuts(stage s, row)  = byte cut indices c in (0, len):  start anchor at t -> c = t,  end anchor at t -> c = t + 1
    word_cuts(row)      = {c : is_space(x[c]) != is_space(x[c-1])}          # whitespace word boundaries
    boundary_alignment: precision/recall/F1 of cuts(s) vs word_cuts, on a val slice,
                        and the same for a fixed-stride chunker with the same chunks per byte
    in_context: for each paragraph and each occurrence k of its made-up word:
                chunks_k = 1 + #cuts(s) strictly inside the word span
                report mean chunks_k by k, per stage
    render: text with '|' at stage-0-only cuts and '#' at cuts that are also stage-1 (or deeper) cuts
    coarse_use: for each stage s and mode in {zero, shuffle}:
                rise_s,mode = bpb(val batches | stage s ablated) - bpb(val batches)
                (Stage.ablate: zero -> `up` = 0; shuffle -> `up` from another document)
                a small shuffle rise at stage s = the decoder at s ignores the network below it
                a stage with a main_read also logs read_{zero,shuffle}_s: the read alone ablated (Stage.read_ablate)
"""
from __future__ import annotations

import html
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .data import pad_rows
from .evaluation import run_batches
from .model import LN2

_SPACE = np.zeros(256, dtype=bool)
_SPACE[[9, 10, 11, 12, 13, 32]] = True


def stage_cuts(aux, row: int, length: int) -> np.ndarray:
    """Bool array (length+1,) marking cut indices for one stage and one batch row."""
    n = int(aux.counts[row])
    pos = aux.pos[row, :n].cpu().numpy()
    c = pos + (1 if aux.anchor == "end" else 0)
    cuts = np.zeros(length + 1, dtype=bool)
    c = c[(c > 0) & (c < length)]
    cuts[c] = True
    return cuts


def word_cuts(x: np.ndarray) -> np.ndarray:
    cuts = np.zeros(len(x) + 1, dtype=bool)
    sp = _SPACE[x]
    cuts[1: len(x)] = sp[1:] != sp[:-1]
    return cuts


def stride_cuts(length: int, bytes_per_chunk: float) -> np.ndarray:
    cuts = np.zeros(length + 1, dtype=bool)
    k = np.arange(1, int(length / bytes_per_chunk) + 1)
    c = np.round(k * bytes_per_chunk).astype(int)
    cuts[c[(c > 0) & (c < length)]] = True
    return cuts


def prf(pred: list[np.ndarray], ref: list[np.ndarray]) -> tuple[float, float, float]:
    tp = sum(int((p & r).sum()) for p, r in zip(pred, ref))
    npred = sum(int(p.sum()) for p in pred)
    nref = sum(int(r.sum()) for r in ref)
    prec, rec = tp / max(npred, 1), tp / max(nref, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


@torch.no_grad()
def run_rows(model, rows: list[np.ndarray], device):
    x, lengths = pad_rows(rows, model.multiple, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(x, lengths)
    return out.aux


@torch.no_grad()
def boundary_alignment(model, rows: list[np.ndarray], device) -> dict:
    aux = run_rows(model, rows, device)
    ref = [word_cuts(r) for r in rows]
    total = sum(len(r) for r in rows)
    metrics = {"probe/ref_cut_rate": sum(int(r.sum()) for r in ref) / total}
    for s, a in enumerate(aux):
        pred = [stage_cuts(a, i, len(r)) for i, r in enumerate(rows)]
        per_byte = sum(int(p.sum()) for p in pred) / total
        stride = [stride_cuts(len(r), 1.0 / max(per_byte, 1e-9)) for r in rows]
        for tag, cuts in (("model", pred), ("stride", stride)):
            p, r, f = prf(cuts, ref)
            metrics.update({f"probe/align_s{s}_{tag}_precision": p, f"probe/align_s{s}_{tag}_recall": r,
                            f"probe/align_s{s}_{tag}_f1": f})
        metrics[f"probe/align_s{s}_cuts_per_byte"] = per_byte
    return metrics


def load_probe(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


@torch.no_grad()
def in_context(model, paragraphs: list[dict], device, max_occ: int = 5) -> dict:
    rows = [np.frombuffer(p["text"].encode("utf-8"), dtype=np.uint8) for p in paragraphs]
    aux = run_rows(model, rows, device)
    metrics = {}
    for s, a in enumerate(aux):
        by_occ: list[list[int]] = [[] for _ in range(max_occ)]
        for i, p in enumerate(paragraphs):
            cuts = stage_cuts(a, i, len(rows[i]))
            for k, m in enumerate(re.finditer(rf"\b{re.escape(p['word'])}\b", p["text"])):
                if k >= max_occ:
                    break
                b0 = len(p["text"][: m.start()].encode())
                b1 = b0 + len(p["word"].encode())
                by_occ[k].append(1 + int(cuts[b0 + 1: b1].sum()))
        for k, v in enumerate(by_occ):
            metrics[f"probe/icl_s{s}_occ{k + 1}"] = float(np.mean(v)) if v else float("nan")
        if by_occ[0] and by_occ[-1]:
            metrics[f"probe/icl_s{s}_drop_1_to_{max_occ}"] = float(np.mean(by_occ[0]) - np.mean(by_occ[-1]))
    return metrics


@torch.no_grad()
def render(model, row: np.ndarray, device) -> list[str]:
    """Return one string: '|' = stage-0 cut, '#' = cut shared with a deeper stage, newline shown as a return arrow."""
    aux = run_rows(model, [row], device)
    cuts = [stage_cuts(a, 0, len(row)) for a in aux]
    deep = np.zeros(len(row) + 1, dtype=bool)
    for c in cuts[1:]:
        deep |= c
    pieces, last = [], 0
    for c in np.flatnonzero(cuts[0] | deep):
        pieces.append(bytes(row[last:c]).decode("utf-8", errors="replace"))
        pieces.append("#" if deep[c] else "|")
        last = c
    pieces.append(bytes(row[last:]).decode("utf-8", errors="replace"))
    return ["".join(pieces).replace("\n", "↵")]


def render_html(text: str) -> str:
    return "<pre style='white-space:pre-wrap;font-size:13px'>" + html.escape(text) + "</pre>"


@contextmanager
def ablated(stage, mode: str, part: str = "all"):
    """Run `stage` with its coarse path ablated inside the block. part "all": the EMA read and any main_read
    (Stage.ablate); part "read": the main_read output alone (Stage.read_ablate)."""
    attr = {"all": "ablate", "read": "read_ablate"}[part]
    if part == "read" and getattr(stage, "main_read", None) is None:
        raise ValueError("part='read' needs a stage with a main_read")
    setattr(stage, attr, mode)
    try:
        yield
    finally:
        setattr(stage, attr, None)


@torch.no_grad()
def coarse_use(model, batches, device, amp) -> dict:
    """bpb rise when each stage's coarse path is zeroed or shuffled across rows. Batches need >= 2 rows."""
    was_training = model.training
    model.eval()
    base = run_batches(model, batches, device, amp)[0]
    m = {"probe/coarse_base_bpb": base / LN2}
    for s, st in enumerate(model.stages()):
        for mode in ("zero", "shuffle"):
            with ablated(st, mode):
                m[f"probe/coarse_{mode}_s{s}"] = (run_batches(model, batches, device, amp)[0] - base) / LN2
            if getattr(st, "main_read", None) is not None:
                with ablated(st, mode, part="read"):
                    m[f"probe/read_{mode}_s{s}"] = (run_batches(model, batches, device, amp)[0] - base) / LN2
    model.train(was_training)
    return m
