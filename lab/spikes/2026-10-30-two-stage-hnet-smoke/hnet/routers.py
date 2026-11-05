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
conf_t = p_t if the position was selected else 1 - p_t (2025 `selected_probs`), fed to the STE.
Note on the 2025 code: q is applied to h_{t-1} and k to h_t (modules/dc.py), so the formula in the
task text with q_t, k_{t-1} is the same cosine with the roles written the other way round; we copy the code.
"""
from __future__ import annotations

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


class Router(nn.Module):
    name: str
    anchor: str
    uses_start_vector: bool
    needs_given = False   # True: boundaries come from outside the model (forward's `given`), not from h

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def boundary_prob(self, h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def read_index(self, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, h: torch.Tensor, lengths: torch.Tensor, given: torch.Tensor | None = None) -> ChunkPlan:
        """given (B,L) bool: frozen boundaries at this level's positions; only for routers with needs_given."""
        B, L, _ = h.shape
        if self.needs_given != (given is not None):
            raise ValueError(f"router {self.name!r}: needs_given={self.needs_given} but given is "
                             f"{'set' if given is not None else 'None'} (model.boundaries configures it)")
        valid = torch.arange(L, device=h.device)[None, :] < lengths[:, None]
        with torch.autocast(device_type=h.device.type, enabled=False):
            p = given.float() if self.needs_given else self.boundary_prob(h.float(), lengths).clamp(0.0, 1.0)
        mask = (p >= 0.5) & valid
        conf = torch.where(mask, p, 1.0 - p)
        return ChunkPlan(p=p, mask=mask, conf=conf, read_idx=self.read_index(mask), anchor=self.anchor)

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
