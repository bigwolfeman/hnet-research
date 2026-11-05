"""Coarse-level JEPA: an auxiliary loss that asks the main network's output to carry what two views share.

Pseudocode (LeJEPA form, as in Balestriero and LeCun, arXiv 2610.39302: no predictor, no stop-gradient, no EMA
teacher; SIGReg keeps the embeddings an isotropic Gaussian, so a constant embedding costs a large loss):
    h(x)   = mean over valid main-network positions of the main network's output   (forward hook on Main)
             Main's input is detached in this pass: the loss trains the main network and the projector only. The
             outer encoders feed the routers, and a JEPA gradient into them swung the chunk ratios (stage 1 up to
             1,606 against a target of 3) because the view rows carry no ratio loss.
    z      = proj(h)                                   # MLP to proj_dim; one batch of N pairs, both views
    inv    = mean over pairs, dims of (z_view - mean of the pair's two views)^2
    sigreg = SIGReg(z_a) + SIGReg(z_b), halved          # Epps-Pulley test vs N(0, 1) on random 1-D projections
    loss   = lambda * sigreg + (1 - lambda) * inv
    train.py adds jepa.weight * loss to the LM loss; the view rows add no LM loss and no ratio loss
Why views of one thing ("good views"): the paper reports that JEPAs on good views need far fewer samples than
next-token prediction (whose views are parts of one document) to reach the same representation alignment.
Why not SimSiam: with 2 pairs per step and no batch statistics it collapsed to a constant embedding in all three
seeds of the first coarse-JEPA run, and the LM loss did not hold it (the decoders already ignored the inner path).
Experiment: lab/experiments/planned/2026-11-02-coarse-jepa-on-dead-seeds.md
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage import Main, Stage


class _StopAfterMain(Exception):
    """Raised by the hook once Main has run: the JEPA pass needs no decoder and no LM head (saves their fwd/bwd)."""


def main_net(model) -> Main:
    net = model.net
    while isinstance(net, Stage):
        net = net.inner
    assert isinstance(net, Main), f"no Main leaf in {type(model.net).__name__}"
    return net


def sigreg(z: torch.Tensor, gen: torch.Generator, slices: int, knots: int) -> torch.Tensor:
    """Sketched isotropic-Gaussian regularizer (LeJEPA): Epps-Pulley statistic of z (N, K) on `slices` random unit
    directions, against N(0, 1). 0 for a perfect standard Gaussian batch, about N for a collapsed one."""
    n, k = z.shape
    a = torch.randn(k, slices, generator=gen, device=z.device)
    a = a / a.norm(dim=0, keepdim=True)
    t = torch.linspace(-5, 5, knots, device=z.device)
    phi = torch.exp(-0.5 * t * t)                                  # CF of N(0, 1), also the weight
    xt = (z.float() @ a).unsqueeze(-1) * t                         # (N, slices, knots)
    err = (xt.cos().mean(0) - phi).square() + xt.sin().mean(0).square()
    return torch.trapezoid(err * phi, t, dim=-1).mean() * n


class CoarseJEPA(nn.Module):
    def __init__(self, model, hidden: int, proj_dim: int, lam: float, slices: int, knots: int, seed: int):
        super().__init__()
        main = main_net(model)
        self.d = d = main.d
        self.lam, self.slices, self.knots, self.seed = lam, slices, knots, seed
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, proj_dim))
        self._capture: list | None = None
        # a plain attribute, not a submodule: the hook belongs to the model and holds no parameters
        self._handle = main.register_forward_hook(self._hook)
        self._pre_handle = main.register_forward_pre_hook(self._detach_input)

    def _detach_input(self, module, args):
        if self._capture is not None:
            return (args[0].detach(),) + tuple(args[1:])
        return None
    def _hook(self, module, args, output):
        if self._capture is not None:
            self._capture.append((output[0], args[1]))   # (B, M, d) main output, (B,) valid main positions
            raise _StopAfterMain

    def embed(self, model, x, lengths) -> torch.Tensor:
        """(B, d) fp32 mean of the main output over valid positions. The forward stops right after Main."""
        self._capture = []
        try:
            model(x, lengths)
            raise AssertionError("the forward did not reach Main")
        except _StopAfterMain:
            (h, counts), = self._capture
        finally:
            self._capture = None
        valid = (torch.arange(h.shape[1], device=h.device)[None, :] < counts[:, None]).unsqueeze(-1)
        return (h.float() * valid).sum(1) / counts.clamp(min=1).unsqueeze(-1).float()

    def loss(self, model, xa, la, xb, lb, step: int) -> tuple[torch.Tensor, dict]:
        # one forward for both views (the narrower view is zero-padded to the wider width): each model forward has
        # one host sync per stage and many small kernels, so two forwards cost far more than one of twice the rows
        w = max(xa.shape[1], xb.shape[1])
        x = torch.cat([F.pad(xa, (0, w - xa.shape[1])), F.pad(xb, (0, w - xb.shape[1]))])
        z = self.proj(self.embed(model, x, torch.cat([la, lb]))).float()
        n = xa.shape[0]
        za, zb = z[:n], z[n:]
        inv = (za - zb).square().mean() / 4                        # = mean square distance to the pair's center
        # random directions from the step: deterministic, and they draw nothing from the global rng
        gen = torch.Generator(device=z.device).manual_seed(self.seed * 1_000_003 + step)
        sr = 0.5 * (sigreg(za, gen, self.slices, self.knots) + sigreg(zb, gen, self.slices, self.knots))
        loss = self.lam * sr + (1 - self.lam) * inv
        with torch.no_grad():
            # collapse check: views of different pairs should stay less alike than views of the same pair
            zn_a, zn_b = F.normalize(za, dim=-1), F.normalize(zb, dim=-1)
            sim = zn_a @ zn_b.T
            same = sim.diagonal().mean()
            other = (sim.sum() - sim.diagonal().sum()) / max(1, n * n - n)
            top1 = (sim.argmax(1) == torch.arange(n, device=z.device)).float().mean()
        return loss, {"jepa/loss": loss.detach(), "jepa/inv": inv.detach(), "jepa/sigreg": sr.detach(),
                      "jepa/z_cos_same_pair": same, "jepa/z_cos_other_pair": other, "jepa/top1_doc_given_code": top1}
