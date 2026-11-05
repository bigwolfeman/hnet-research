"""Parcae looped main network: prelude, core loop with stable diagonal injection, coda.

Port of Parcae (Prairie, Novack, Berg-Kirkpatrick, Fu 2026, "Parcae: Scaling Laws For Stable Looped Language
Models", arXiv 2604.12946; code github.com/SandyResearch/parcae at 4b86c2a, MIT License, Copyright (c) 2026 Hayden
Prairie; license text in hnet/LICENSE.parcae): parcae_lm/modules/injection.py DiagonalInjection, parcae_lm/models/parcae/parcae.py iterate_forward,
_sample_batch_depths ("poisson-truncated-full", "per-sequence"), initialize_state ("like-init"), _no_grad_prefix,
and the init of parcae_lm/models/parcae/init.py (A_log 0, dt from the decay target sqrt(1/5), B identity).

Pseudocode:
    e = prelude(h)                                   # a Stack ending in RMSNorm (Parcae prelude_norm: true)
    decay = exp(-softplus(dt_bias) * exp(A_log))     # in (0, 1) per channel: the state map is a contraction
    u = softplus(dt_bias) * (e @ B^T)                # the injected input, computed once (e is the same every step)
    x = TruncNormal(0, std, +-3 std)                 # the loop state; a fixed generator in eval mode
    train: per row i: T_i = max(1, Poisson(R)); k_i = min(T_i, s); n_i = T_i - k_i
           no grad (autocast cache off):  for t < max n:  x_i <- core(decay * x_i + u_i)  where t < n_i
           with grad:                     for t < max k:  x_i <- core(decay * x_i + u_i)  where t < k_i
    eval:  R steps for every row
    y = coda(C x)                                    # a Stack ending in RMSNorm (Parcae ln_f)
The core is a Stack with no final norm: Parcae has no norm between iterations, only inside the blocks.
Differences from the code: u is computed once, not at every step (the same values); the eval state comes from a
fixed-seed generator so that the coarse-path probe compares runs with the same state; the blocks are ours (init rule
of hnet/stage.py init_stack over prelude + R x core + coda, not Parcae's scaled-zero; no value embeddings).
Experiment: lab/experiments/refuted/2026-11-04-parcae-coarse-main.md
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Stack
from .stage import Main, init_stack


@contextmanager
def no_grad_prefix():
    """Parcae _no_grad_prefix: autocast caches weight casts; a cast made under no_grad has no grad_fn, and the steps
    with grad would reuse it, so the core weights would get no gradient. Turn the cache off for the prefix."""
    cache = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        with torch.no_grad():
            yield
    finally:
        torch.set_autocast_cache_enabled(cache)


def truncated_full_depths(total: torch.Tensor, s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, k): the last min(total, s) steps get gradient (Parcae _truncated_full_depths)."""
    k = torch.clamp(total, max=s)
    return total - k, k


class ParcaeMain(Main):
    def __init__(self, prelude: Stack, core: Stack, coda: Stack, mean_recurrence: int, backprop_depth: int,
                 decay_init: float, state_std: float, seed: int):
        nn.Module.__init__(self)
        if not (prelude.d == core.d == coda.d):
            raise ValueError("parcae: prelude, core and coda need one width")
        if mean_recurrence < 1 or not (1 <= backprop_depth <= mean_recurrence):
            raise ValueError(f"parcae: need mean_recurrence >= 1 and 1 <= backprop_depth <= mean_recurrence, got "
                             f"{mean_recurrence}, {backprop_depth}")
        if not 0.0 < decay_init < 1.0:
            raise ValueError(f"parcae: decay_init {decay_init} must be in (0, 1)")
        d = core.d
        self.prelude, self.core, self.coda = prelude, core, coda
        self.R, self.s, self.decay_init, self.seed = int(mean_recurrence), int(backprop_depth), float(decay_init), int(seed)
        self.A_log = nn.Parameter(torch.zeros(d))
        self.dt_bias = nn.Parameter(torch.zeros(d))
        self.B = nn.Parameter(torch.empty(d, d))
        self.C = nn.Linear(d, d, bias=False)
        for p in (self.A_log, self.dt_bias, self.B):
            p._no_weight_decay = True       # Parcae excludes the injection parameters from weight decay
        self.register_buffer("draws", torch.zeros((), dtype=torch.long))  # depth draws so far (resumable sampler)
        self.state_std = float(state_std)
        self.last_depth: tuple[float, int] | None = None   # (mean total depth, max total depth) of the last train step

    @property
    def d(self) -> int:
        return self.core.d

    @property
    def loops(self) -> int:
        return self.R

    def decay(self) -> torch.Tensor:
        return torch.exp(-F.softplus(self.dt_bias) * torch.exp(self.A_log))

    def sample_depths(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row (n, k) for one training forward: Poisson total depth with mean R, at least 1, last s with grad."""
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(self.draws))
        self.draws += 1
        total = torch.clamp(torch.poisson(torch.full((rows,), float(self.R)), generator=g), min=1).long()
        return truncated_full_depths(total, self.s)

    def init_state(self, e: torch.Tensor) -> torch.Tensor:
        """Parcae state_init "like-init": normal at the init std, truncated at 3 std. fp32, like the residual stream."""
        g = None if self.training else torch.Generator(device=e.device).manual_seed(self.seed)
        x = torch.empty(e.shape, device=e.device, dtype=torch.float32)
        s = self.state_std
        return nn.init.trunc_normal_(x, mean=0.0, std=s, a=-3 * s, b=3 * s, generator=g)

    def step(self, x, u, decay):
        return self.core(x * decay + u)

    uses_pack = False   # per-row loop depths need the padded (B, M) layout; the outer Stage passes no Pack

    def forward(self, h, lengths, pos, pack=None, given=None):
        assert not given, "frozen boundaries left over at the main network: one mask per stage"
        e = self.prelude(h)
        u = F.softplus(self.dt_bias) * (e @ self.B.t()).float()
        decay = self.decay()
        x = self.init_state(e)
        if not self.training:
            for _ in range(self.R):
                x = self.step(x, u, decay)
        else:
            n, k = self.sample_depths(h.shape[0])
            self.last_depth = (float((n + k).float().mean()), int((n + k).max()))
            n, k = n.to(h.device), k.to(h.device)
            with no_grad_prefix():
                for t in range(int(n.max())):
                    x = torch.where((t < n)[:, None, None], self.step(x, u, decay), x)
            for t in range(int(k.max())):
                x = torch.where((t < k)[:, None, None], self.step(x, u, decay), x)
        return self.coda(self.C(x)), []

    def init_weights(self, std: float, n_res: int):
        n = n_res + self.prelude.height + self.R * self.core.height + self.coda.height
        for st in (self.prelude, self.core, self.coda):
            init_stack(st, std, n)
        nn.init.normal_(self.C.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.A_log.zero_()                                          # A = 1
            dt0 = -math.log(self.decay_init)                            # exp(-dt0) = decay_init
            self.dt_bias.fill_(dt0 + math.log(-math.expm1(-dt0)))       # softplus^-1(dt0)
            self.B.copy_(torch.eye(self.d))

    @torch.no_grad()
    def log_stats(self) -> dict:
        """Parcae's monitors: the depth of the last training forward and the injection decay (max: spectral norm)."""
        dec = self.decay()
        out = {"parcae/decay_max": float(dec.max()), "parcae/decay_mean": float(dec.mean())}
        if self.last_depth is not None:
            out.update({"parcae/depth_mean": self.last_depth[0], "parcae/depth_max": float(self.last_depth[1])})
        return out

    def compile_blocks(self, dynamic: bool):
        for st in (self.prelude, self.core, self.coda):
            st.compile_blocks(dynamic=dynamic)

    def describe(self) -> dict:
        return dict(kind="parcae", prelude=self.prelude.describe(), core=self.core.describe(), coda=self.coda.describe(),
                    mean_recurrence=self.R, backprop_depth=self.s, decay_init=self.decay_init,
                    sampling="poisson-truncated-full per-sequence", state_init="like-init", injection="diagonal")
