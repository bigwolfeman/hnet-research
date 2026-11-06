"""Frozen chunk boundaries from outside the model: the blog's BLT-like control (Cartesia Part II, Table 2).

A frozen byte LM (the proxy) scores each byte. Byte t starts a chunk at stage s when the proxy's next-byte entropy
before t is above threshold theta_s. The H-Net then trains with these boundaries in place of its routers
(hnet/routers.py "given"); everything else (architecture, ratio, LR) is the end-to-end model's.

Pseudocode:
    H_t   = entropy of proxy(x_<t) over byte t   (fp32; H_0 = +inf, so byte 0 starts a chunk)
    b_s,t = H_t > theta_s                        theta_0 < theta_1 < ...  ->  stage-1 starts are a subset of stage 0
    calibrate(rows, ratios): theta_s = quantile of H over calibration bytes (t > 0) at 1 - 1 / prod(ratios[:s+1])
Causal: b_t depends on x_<t only. The proxy's FLOPs are not counted (as in the blog).
model.boundaries config:  {kind: entropy, proxy: <final.pt of a byte LM>, thresholds: [theta_0, theta_1, ...]}
Experiment: lab/experiments/successes/2026-11-05-learned-vs-entropy-boundaries.md
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class EntropyBoundaries:
    """Callable (x, lengths) -> list of (B, L) bool masks, one per stage. Holds a frozen proxy; not an nn.Module, so
    the proxy stays out of the H-Net's parameters, optimizer and state_dict."""

    def __init__(self, proxy: torch.nn.Module, thresholds: list[float], source: dict):
        th = [float(t) for t in thresholds]
        if any(b <= a for a, b in zip(th, th[1:])):
            raise ValueError(f"entropy thresholds must rise stage by stage (nested boundaries): {th}")
        self.proxy, self.thresholds, self.source = proxy.eval().requires_grad_(False), th, source

    @torch.no_grad()
    def entropy(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(B, L) fp32 next-byte entropy in nats; position t scores byte t from x_<t; position 0 is +inf."""
        self.proxy.eval()
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            logits = self.proxy(x, lengths).logits
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        h = -(lp.exp() * lp).sum(-1)
        return F.pad(h, (1, 0), value=float("inf"))

    def __call__(self, x: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        h = self.entropy(x, lengths)
        return [h > t for t in self.thresholds]

    def describe(self) -> dict:
        return dict(kind="entropy", thresholds=self.thresholds, **self.source)


def load_proxy(path: Path, device) -> tuple[torch.nn.Module, dict]:
    from .model import build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    if mcfg.get("stages"):
        raise ValueError(f"{path}: the entropy proxy must be a byte LM with no stages")
    proxy = build_model(mcfg)
    proxy.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
    src = dict(proxy=str(path), proxy_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               proxy_step=ckpt.get("step"), proxy_params=sum(p.numel() for p in proxy.parameters()))
    return proxy.to(device), src


def build_boundaries(cfg: dict | None, n_stages: int, device) -> EntropyBoundaries | None:
    """model.boundaries -> a boundary source, or None (routers decide)."""
    if not cfg:
        return None
    if cfg.get("kind") != "entropy":
        raise ValueError(f"model.boundaries.kind {cfg.get('kind')!r}: only 'entropy' exists")
    th = cfg.get("thresholds")
    if th is None or len(th) != n_stages:
        raise ValueError(f"model.boundaries.thresholds {th}: need one per stage ({n_stages}); "
                         "run scripts/calibrate_entropy.py")
    proxy, src = load_proxy(Path(cfg["proxy"]), device)
    return EntropyBoundaries(proxy, th, src)


def calibrate(entropies: list[np.ndarray], ratios: list[float]) -> list[float]:
    """Thresholds that give 1 / prod(ratios[:s+1]) boundaries per byte at stage s over the given entropies (t > 0)."""
    h = np.concatenate([e[1:] for e in entropies]).astype(np.float64)
    cum = np.cumprod(np.asarray(ratios, dtype=np.float64))
    return [float(np.quantile(h, 1.0 - 1.0 / r)) for r in cum]
