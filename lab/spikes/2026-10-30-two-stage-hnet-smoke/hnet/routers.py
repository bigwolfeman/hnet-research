"""Router registry. A router turns encoder states into a ChunkPlan.

Pseudocode:
    lookahead (2025):  p_t = (1 - cos(q(h_{t-1}), k(h_t))) / 2,  p_0 = 1,  b_t = p_t >= 0.5
        b_t = 1 means byte t STARTS a chunk. Chunk vector = h at the start position.
        Decoder position t reads the chunk containing t:  read_t = cumsum(b)_t - 1.
    eoc (end of chunk, our hypothesis):  p_t = sigmoid(w . h_t + c),  e_t = p_t >= 0.5,
        last valid position forced to 1. e_t = 1 means byte t is the LAST of a chunk.
        Chunk vector = h at the end position. Decoder position t in chunk k reads main output
        k-1:  read_t = cumsum(e)_t - e_t - 1  (-1 -> learned start vector). Strictly causal.
    given (frozen boundaries, the blog's BLT-like control):  p_t = given_t in {0, 1} from outside the model
        (hnet/boundaries.py, e.g. a proxy LM's next-byte entropy above a threshold), b_t = given_t. No parameters,
        no gradient through p; chunks start at the given bytes, read_t = cumsum(b)_t - 1 as in lookahead.
    surprise (stage 0 only; in-context tokenization):  an aux head reads the encoder state and predicts the next byte,
        q_t = softmax(W h_t) ~ P(x_{t+1} | x_<=t), trained with its own CE (weight lm_weight, gradient into the encoder).
        H_t = entropy of q_{t-1} (nats, detached): how unsure the encoder was about byte t before reading it.
        Hz_t = (H_t - mu) / sigma, mu and sigma running stats of H (buffers, updated in training only).
        z_t = logit(clamp(p^lookahead_t)) + beta * (Hz_t - tau),  p_t = sigmoid(z_t),  p_0 = 1.  beta, tau learned.
        A repeated new word becomes predictable from context, so H_t falls and its inner cuts can go, as with the
        entropy proxy, but every part is trained end to end. Start anchors, read_t = cumsum(b)_t - 1 as in lookahead.
conf_t = p_t if the position was selected else 1 - p_t (2025 `selected_probs`), fed to the STE.
Note on the 2025 code: q is applied to h_{t-1} and k to h_t (modules/dc.py), so the formula in the
task text with q_t, k_{t-1} is the same cosine with the roles written the other way round; we copy the code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ROUTERS: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        ROUTERS[name] = cls
        cls.name = name
        return cls

    return deco


def build_router(name: str, d_model: int, **kwargs) -> "Router":
    if name not in ROUTERS:
        raise KeyError(f"unknown router {name!r}; registered: {sorted(ROUTERS)}")
    return ROUTERS[name](d_model, **kwargs)


@dataclass
class ChunkPlan:
    p: torch.Tensor          # (B,L) fp32 boundary prob (differentiable)
    mask: torch.Tensor       # (B,L) bool anchor positions (valid only)
    conf: torch.Tensor       # (B,L) fp32 selected-class prob for the STE
    read_idx: torch.Tensor   # (B,L) long: chunk index read by the decoder, -1 = start vector
    anchor: str              # "start" or "end"
    aux_loss: torch.Tensor | None = None   # scalar router loss, already weighted (surprise: next-byte CE); None: none


class Router(nn.Module):
    name: str
    anchor: str
    uses_start_vector: bool
    needs_given = False   # True: boundaries come from outside the model (forward's `given`), not from h
    needs_next = False    # True: the router trains on the next byte at each position (forward's `nxt`, stage 0 only)

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def boundary_prob(self, h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def read_index(self, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, h: torch.Tensor, lengths: torch.Tensor, given: torch.Tensor | None = None,
                nxt: torch.Tensor | None = None) -> ChunkPlan:
        """given (B,L) bool: frozen boundaries at this level's positions; only for routers with needs_given.
        nxt (B,L) long: the byte after each position, -100 where there is none; only for routers with needs_next."""
        B, L, _ = h.shape
        if self.needs_given != (given is not None):
            raise ValueError(f"router {self.name!r}: needs_given={self.needs_given} but given is "
                             f"{'set' if given is not None else 'None'} (model.boundaries configures it)")
        if self.needs_next != (nxt is not None):
            raise ValueError(f"router {self.name!r}: needs_next={self.needs_next} but nxt is "
                             f"{'set' if nxt is not None else 'None'} (only stage 0 gets the next bytes)")
        valid = torch.arange(L, device=h.device)[None, :] < lengths[:, None]
        aux_loss = None
        with torch.autocast(device_type=h.device.type, enabled=False):
            if self.needs_given:
                p = given.float()
            elif self.needs_next:
                p, aux_loss = self.boundary_prob_next(h.float(), lengths, nxt)
                p = p.clamp(0.0, 1.0)
            else:
                p = self.boundary_prob(h.float(), lengths).clamp(0.0, 1.0)
        mask = (p >= 0.5) & valid
        conf = torch.where(mask, p, 1.0 - p)
        return ChunkPlan(p=p, mask=mask, conf=conf, read_idx=self.read_index(mask), anchor=self.anchor,
                         aux_loss=aux_loss)

    def describe(self) -> dict:
        return dict(name=self.name, anchor=self.anchor, params=sum(p.numel() for p in self.parameters()))


@register("lookahead")
class LookaheadRouter(Router):
    anchor = "start"
    uses_start_vector = False

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():  # identity init, as in the 2025 code
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def boundary_prob(self, h, lengths):
        cos = (F.normalize(self.q_proj(h[:, :-1]), dim=-1) * F.normalize(self.k_proj(h[:, 1:]), dim=-1)).sum(-1)
        p = ((1 - cos) / 2).clamp(0.0, 1.0)
        return F.pad(p, (1, 0), value=1.0)

    def read_index(self, mask):
        return torch.cumsum(mask, dim=1) - 1


@register("surprise")
class SurpriseRouter(LookaheadRouter):
    """Lookahead cosine plus the encoder's own next-byte entropy (module docstring). Stage 0 only.
    The entropy enters standardized by running stats (BatchNorm-style buffers, updated in training only, so the
    term at byte t never sees other positions of the same batch): its scale falls fast early in training, and a
    raw threshold would swing the ratio between all-cut and no-cut."""
    needs_next = True

    def __init__(self, d_model: int, vocab: int = 256, lm_weight: float = 1.0, beta_init: float = 1.0,
                 tau_init: float = 0.0, momentum: float = 0.1, eps: float = 1e-3):
        super().__init__(d_model)
        self.lm = nn.Linear(d_model, vocab, bias=False)   # init_weights leaves router weights alone (_no_reinit)
        nn.init.normal_(self.lm.weight, std=d_model ** -0.5)
        self.lm.weight._no_reinit = True
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))   # threshold in running-std units
        self.register_buffer("h_mean", torch.tensor(math.log(vocab)))
        self.register_buffer("h_var", torch.tensor(1.0))
        self.lm_weight, self.momentum, self.eps = float(lm_weight), float(momentum), float(eps)

    def boundary_prob(self, h, lengths):
        raise RuntimeError("router 'surprise' needs the next bytes: call forward(..., nxt=...)")

    def boundary_prob_next(self, h, lengths, nxt):
        logits = self.lm(h)                                              # (B,L,V): predicts the byte after t
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), nxt.reshape(-1), ignore_index=-100)
        with torch.no_grad():
            logq = F.log_softmax(logits, dim=-1)
            ent = -(logq.exp() * logq).sum(-1)                           # H of P(x_{t+1} | x_<=t), nats
            ent = F.pad(ent[:, :-1], (1, 0))                             # at t: entropy over byte t, before reading it
            if self.training:
                valid = (torch.arange(ent.shape[1], device=ent.device)[None, :] < lengths[:, None]) & (
                    torch.arange(ent.shape[1], device=ent.device)[None, :] > 0)
                n = valid.sum().clamp_min(1)
                mu = (ent * valid).sum() / n
                var = (((ent - mu) ** 2) * valid).sum() / n
                self.h_mean.lerp_(mu, self.momentum)
                self.h_var.lerp_(var, self.momentum)
            ent_z = (ent - self.h_mean) / (self.h_var.sqrt() + 1e-4)
        p_cos = super().boundary_prob(h, lengths).clamp(self.eps, 1.0 - self.eps)
        z = torch.logit(p_cos) + self.beta * (ent_z - self.tau)
        p = torch.sigmoid(z)
        p = torch.cat([torch.ones_like(p[:, :1]), p[:, 1:]], dim=1)      # p_0 = 1, as lookahead
        return p, self.lm_weight * ce

    def describe(self) -> dict:
        return dict(super().describe(), lm_weight=self.lm_weight, momentum=self.momentum, beta=float(self.beta),
                    tau=float(self.tau), h_mean=float(self.h_mean), h_var=float(self.h_var))


@register("given")
class GivenRouter(Router):
    """Frozen boundaries from outside the model (hnet/boundaries.py). Start anchors, like lookahead."""
    anchor = "start"
    uses_start_vector = False
    needs_given = True

    def boundary_prob(self, h, lengths):
        raise RuntimeError("router 'given' has no boundary_prob: its boundaries come in through forward(given=...)")

    def read_index(self, mask):
        return torch.cumsum(mask, dim=1) - 1


@register("eoc")
class EndOfChunkRouter(Router):
    anchor = "end"
    uses_start_vector = True

    def __init__(self, d_model: int, force_last: bool = False, init_std: float = 0.02, read_own_end: bool = True):
        super().__init__(d_model)
        if force_last and read_own_end:
            # A forced boundary at the last valid position encodes "the sequence ends here" (future info);
            # with read_own_end that position would read it. Fail loud instead of leaking at inference.
            raise ValueError("eoc: force_last=True is incompatible with read_own_end=True")
        self.force_last = force_last
        # read_own_end=True: the last byte of chunk j reads main output j (complete and causal at that byte),
        # the tokenized-LM alignment. False lags one chunk: that byte reads output j-1.
        self.read_own_end = read_own_end
        self.head = nn.Linear(d_model, 1)
        nn.init.normal_(self.head.weight, std=init_std)
        nn.init.zeros_(self.head.bias)
        self.head.weight._no_reinit = True

    def boundary_prob(self, h, lengths):
        p = torch.sigmoid(self.head(h).squeeze(-1))
        if self.force_last:
            last = torch.arange(h.shape[1], device=h.device)[None, :] == (lengths[:, None] - 1)
            p = torch.where(last, torch.ones_like(p), p)
        return p

    def read_index(self, mask):
        ends_through_t = torch.cumsum(mask, dim=1)
        if self.read_own_end:
            return ends_through_t - 1
        return ends_through_t - mask.long() - 1
