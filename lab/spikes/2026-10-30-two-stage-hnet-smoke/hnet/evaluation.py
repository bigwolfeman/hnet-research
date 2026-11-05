"""Per-domain eval hook.

Pseudocode:
    evaluate_domain(model, batches, name, bpt):
        for (x, y) in batches: logits, aux = model(x) ; ce_sum += CE ; for each stage k: anchors_k += sum(aux_k.counts)
        bpb              = mean CE / ln 2
        bytes_per_chunk_k = total input bytes / anchors_k                # raw bytes per stage-k chunk (cumulative)
        ratio_k          = anchors_{k-1} / anchors_k                     # relative to the stage's own input (anchors_-1 = bytes)
        hnet_over_bpe_k  = bytes_per_chunk_k / bpt                       # Cartesia Fig. 6; > 1: H-Net compresses more than BPE
    evaluate(model, domains): run it per domain ; the primary domain also fills the legacy val/* keys and FLOPs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import bpe, tokens
from .flops import flops_per_byte, token_flops_per_byte
from .model import LN2, total_loss


def stage_stats(anchor_sums: list[int], total_bytes: int) -> tuple[list[float], list[float]]:
    """(cumulative bytes per chunk, per-stage ratio) from summed anchor counts per stage."""
    cum = [total_bytes / a for a in anchor_sums]
    prev = [total_bytes] + list(anchor_sums[:-1])
    return cum, [p / a for p, a in zip(prev, anchor_sums)]


def domain_metrics(name: str, ce_mean: float, anchor_sums: list[int], total_bytes: int, bpt: float | None) -> dict:
    cum, rel = stage_stats(anchor_sums, total_bytes)
    m = {f"val/{name}/bpb": ce_mean / LN2, f"val/{name}/ce": ce_mean}
    for k, (c, r) in enumerate(zip(cum, rel)):
        m[f"val/{name}/bytes_per_chunk_s{k}"] = c
        m[f"val/{name}/ratio_s{k}"] = r
        if bpt is not None:
            m[f"val/{name}/hnet_over_bpe_s{k}"] = bpe.hnet_over_bpe(c, bpt)
    if bpt is not None:
        m[f"val/{name}/bpe_bytes_per_token"] = bpt
    return m


@torch.no_grad()
def run_batches(model, batches, device, amp):
    ce_sum, anchors, total = 0.0, None, 0
    for x, y in batches:
        with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda"):
            out = model(x)
        lengths = torch.full((x.shape[0],), x.shape[1], device=device)
        _, ce = total_loss(out, y, lengths)
        ce_sum += ce.item()
        a = np.array([int(o.counts.sum()) for o in out.aux], dtype=np.int64)
        anchors = a if anchors is None else anchors + a
        total += x.numel()
    return ce_sum / len(batches), anchors.tolist(), total


def evaluate(model, domain_batches: dict, bpts: dict, mcfg, seq_len, device, amp, primary: str):
    """domain_batches: {name: batches}. Returns (metrics, ratios of the primary domain)."""
    model.eval()
    m, ratios = {}, None
    for name, batches in domain_batches.items():
        ce, anchors, total = run_batches(model, batches, device, amp)
        m.update(domain_metrics(name, ce, anchors, total, bpts.get(name)))
        if name == primary:
            _, ratios = stage_stats(anchors, total)
            ratios = [float(r) for r in ratios]
            fl = flops_per_byte(mcfg, seq_len, ratios)
            m.update({"val/ce": ce, "val/bpb": ce / LN2, "flops/fwd_per_byte": fl["fwd"],
                      "flops/train_per_byte": fl["train"], "flops/main_seq_len": fl["main_seq_len"]})
            for s, r in enumerate(ratios):
                m[f"val/ratio_s{s}"] = r
            m["val/ratio_total"] = float(np.prod(ratios))
            for k, v in fl["breakdown"].items():
                m[f"flops/frac_{k}"] = v / fl["fwd"]
    model.train()
    return m, ratios


# ---- token-level models (model.kind == "token") ----

@torch.no_grad()
def run_token_windows(model, windows: torch.Tensor, nbytes: np.ndarray, batch_size: int, device, amp):
    """windows: (n, L+1) int32 on device. Returns (sum of token CE in nats, tokens scored, bytes those tokens stand for)."""
    ce_sum = torch.zeros((), device=device, dtype=torch.float64)
    nb = torch.from_numpy(nbytes.astype(np.int64)).to(device)
    n_bytes = torch.zeros((), device=device, dtype=torch.int64)
    for i in range(0, windows.shape[0], batch_size):
        w = windows[i: i + batch_size].long()
        x, y = w[:, :-1], w[:, 1:]
        with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda"):
            logits = model(x).logits
        ce_sum += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").double()
        n_bytes += nb[y].sum()
    return float(ce_sum), windows.shape[0] * (windows.shape[1] - 1), int(n_bytes)


def token_domain_metrics(name: str, ce_sum: float, n_tokens: int, n_bytes: int) -> dict:
    """Keys shared with the byte models (val/{d}/bpb, val/{d}/ce) plus val/{d}/ce_per_token and bytes_per_token."""
    return {f"val/{name}/bpb": tokens.bpb(ce_sum, n_bytes), f"val/{name}/ce": ce_sum / n_bytes,
            f"val/{name}/ce_per_token": ce_sum / n_tokens, f"val/{name}/bytes_per_token": n_bytes / n_tokens}


def evaluate_tokens(model, domain_windows: dict, nbytes: np.ndarray, mcfg, seq_len, batch_size, device, amp, primary: str):
    """domain_windows: {name: (n, L+1) int32 device tensor}. FLOPs/byte uses the primary domain's measured bytes per token."""
    model.eval()
    m = {}
    for name, w in domain_windows.items():
        ce_sum, n_tok, n_bytes = run_token_windows(model, w, nbytes, batch_size, device, amp)
        m.update(token_domain_metrics(name, ce_sum, n_tok, n_bytes))
        if name == primary:
            fl = token_flops_per_byte(mcfg, seq_len, n_bytes / n_tok)
            m.update({"val/ce": m[f"val/{name}/ce"], "val/bpb": m[f"val/{name}/bpb"],
                      "flops/fwd_per_byte": fl["fwd"], "flops/train_per_byte": fl["train"],
                      "flops/main_seq_len": fl["main_seq_len"], "flops/fwd_per_token": fl["fwd_per_token"]})
    model.train()
    return m
