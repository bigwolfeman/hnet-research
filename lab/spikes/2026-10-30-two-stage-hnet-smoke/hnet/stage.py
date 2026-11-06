"""Recursive Stage (Principle 1). Every level of the hierarchy is the same interface:

    net(h (B,L,d), lengths (B,), pos (B,L)) -> (h_out (B,L,d), aux: list[StageAux])

`Main` is the leaf. `Stage(encoder, router, inner, decoder)` wraps any such net.

Stage.forward = _pre (compiled) -> host sync for M -> _gather_in (compiled) [-> pack_index (compiled), pack_inner]
                -> inner -> _scatter_out (compiled) -> decoder.
Stage.forward pseudocode:
    e      = encoder(h)                                   # causal SWA stack
    resid  = residual_proj(e)                             # fp32, zero weight init
    plan   = router(e, lengths, given[0] at pos)           # anchors, p, conf, read_idx; given: frozen boundaries or None
    idx, counts, valid = select_chunks(plan.mask)         # right-padded to a multiple of `multiple`
    x      = gather(e, idx) * valid                       # (B,M,d)
    y, aux = inner(pad_dims(x), counts, gather(pos, idx)) # recursion
    ema    = ema_scan(y, gather(plan.p, idx))             # fp32 parallel scan
    up     = gather(ema, plan.read_idx)                   # -1 -> learned start vector
    z      = up * STE(plan.conf) + resid                  # fp32, 2025 residual_func
    main_idx = plan.read_idx                 if inner is Main   # last main chunk each position can see (-1: none)
             = inner_main_idx[plan.read_idx] otherwise          # composite: position -> inner position -> main chunk
    z     += main_read(e, main_out[..., :d], main_idx)  # optional content read of past MAIN outputs (hnet/mainread.py)
    return decoder(z), [own aux (+ main_out, main_idx)] + aux

main_idx is the set the EMA path already sees: a position reads inner position read_idx, and that inner output
depends on main chunks <= inner_main_idx[read_idx]. So a read at any stage has the EMA path's causality, and its
keys are always main outputs (never a stage's outputs, which carry local content through their residual).

Probe-only ablation (`stage.ablate`, set by probes.ablated; None in training):
    "zero":    up = 0                  # the decoder sees only its residual
    "shuffle": up = up from row b-1    # coarse vectors of another document, same scale
    The main_read output r is ablated the same way (zero, or r from row b-1).
    `stage.read_ablate` (probes.ablated(..., part="read")) ablates r alone and keeps `up`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn

from .blocks import Stack
from .chunking import chunk_index, chunk_stats, ema_scan, gather_rows, ratio_loss, ste
from .mainread import MainRead
from .packing import Pack, make_pack, pack_index
from .routers import Router


@dataclass
class StageAux:
    ratio_loss: torch.Tensor   # scalar, weighted by the trainer
    frac: torch.Tensor         # scalar, anchors / valid positions (1 / ratio)
    counts: torch.Tensor       # (B,) anchors per row
    lengths: torch.Tensor      # (B,) input positions per row
    pos: torch.Tensor          # (B,M) byte position of every anchor
    anchor: str                # "start" | "end"
    p_mean: torch.Tensor       # scalar, mean boundary prob over valid positions (with grad: RatioController term)
    target_ratio: float
    loss_weight: float
    main_out: torch.Tensor | None = None   # (B,M_main,d_main) main network outputs (our Stage only)
    main_idx: torch.Tensor | None = None   # (B,L) last main chunk each input position of this stage can see, -1: none
    p: torch.Tensor | None = None          # (B,L) detached fp32 router boundary prob at each input position
    in_pos: torch.Tensor | None = None     # (B,L) byte position of each input position (hnet/semantics.py)
    router_loss: torch.Tensor | None = None  # scalar, already weighted (router "surprise": next-byte CE); None: none


class Main(nn.Module):
    """Leaf network: global causal attention. Same interface as Stage.

    loops = R >= 1 runs the same stack R times (weight-tied; Phase 2 of docs/constitution.md):
        for r in 1..R: h = stack(h)          # each pass ends in the stack's final RMSNorm
    The init counts the looped depth: output projections scale by 1/sqrt(n_res + R * stack.height).
    Experiment: lab/experiments/refuted/2026-11-03-looped-coarse-main.md
    loops > 1 erases the input on text (no input injection); the looped main is hnet/parcae.py ParcaeMain (a subclass).
    """

    uses_pack = True   # Stage builds a Pack for an inner net that takes one (hnet/packing.py)

    def __init__(self, stack: Stack, loops: int = 1):
        super().__init__()
        if loops < 1:
            raise ValueError(f"main.loops {loops}: must be >= 1")
        self.stack, self.loops = stack, int(loops)

    def forward(self, h, lengths, pos, pack: Pack | None = None, given=None):
        assert not given, "frozen boundaries left over at the main network: one mask per stage"
        for _ in range(self.loops):
            h = self.stack(h, pack)
        return h, []

    def init_weights(self, std: float, n_res: int):
        init_stack(self.stack, std, n_res + self.loops * self.stack.height)

    @property
    def d(self) -> int:
        return self.stack.d

    def compile_blocks(self, dynamic: bool):
        self.stack.compile_blocks(dynamic=dynamic)

    def stage_params(self):
        return list(self.parameters())

    def describe(self) -> dict:
        return dict(kind="main", loops=self.loops, **self.stack.describe())


class Stage(nn.Module):
    uses_pack = True

    def __init__(self, encoder: Stack, router: Router, inner: nn.Module, decoder: Stack, d: int, d_inner: int,
                 multiple: int, target_ratio: float, loss_weight: float, stage_idx: int,
                 main_read: MainRead | None = None, pack_inner: bool = False):
        super().__init__()
        self.pack_inner = pack_inner   # run the inner net's stacks on valid positions only (hnet/packing.py, CUDA)
        assert d_inner >= d, "inner width must be >= outer width (pad dims are learned)"
        self.encoder, self.router, self.inner, self.decoder = encoder, router, inner, decoder
        self.d, self.d_inner, self.multiple, self.stage_idx = d, d_inner, multiple, stage_idx
        self.target_ratio, self.loss_weight = target_ratio, loss_weight
        self.residual_proj = nn.Linear(d, d)  # fp32 params; zero weight, PyTorch-default bias (as the 2025 code)
        self.pad_dim = nn.Parameter(torch.zeros(d_inner - d)) if d_inner > d else None
        self.start_vec = nn.Parameter(torch.zeros(d)) if router.uses_start_vector else None
        nn.init.zeros_(self.residual_proj.weight)   # bias keeps nn.Linear's default init: the 2025 code zeroes only the
        # weight. Found by the init comparison in lab/experiments/refuted/2026-11-02-reference-init-in-our-code.md
        self.ablate: str | None = None       # probe-only, see module docstring
        self.read_ablate: str | None = None  # probe-only: ablate the main_read output alone
        self.main_read = main_read
        self._pre_fn, self._gather_fn, self._scatter_fn = self._pre, self._gather_in, self._scatter_out
        self._read_fn = main_read.forward if main_read is not None else None
        self._pack_fn = pack_index

    # The forward is three pure pieces around the one host sync (M). Each piece compiles as one graph.
    def _pre(self, e, lengths, given=None, nxt=None):
        """Residual, router plan, anchor counts and ratio loss. No host sync. given: frozen boundaries (B,L) or None.
        nxt: next byte per position (B,L) for a router with needs_next, else None."""
        B, L, _ = e.shape
        valid = torch.arange(L, device=e.device)[None, :] < lengths[:, None]
        with torch.autocast(device_type=e.device.type, enabled=False):
            resid = self.residual_proj(e.float())
        plan = self.router(e, lengths, given, nxt)
        counts = plan.mask.sum(dim=1)
        rl, frac, p_mean = ratio_loss(plan.p, plan.mask, valid, self.target_ratio)
        return resid, plan.p, plan.mask, plan.conf, plan.read_idx, counts, rl, frac, p_mean, plan.aux_loss

    def _gather_in(self, e, p, mask, counts, pos, M: int):
        """Gather anchor states into the (B,M,d_inner) input of the inner net."""
        idx, cvalid = chunk_index(mask, counts, M)
        x = gather_rows(e, idx) * cvalid.unsqueeze(-1).to(e.dtype)
        if self.pad_dim is not None:
            x = torch.cat([x, self.pad_dim.to(x.dtype).expand(*x.shape[:2], -1)], dim=-1)
        return x, gather_rows(pos, idx), gather_rows(p, idx)

    def _scatter_out(self, y, p_c, read_idx, conf, resid, dtype, ablate: str | None = None):
        """EMA-smooth the inner output, read it back at every position, add the residual. Returns the decoder input."""
        y = y[..., : self.d]
        with torch.autocast(device_type=y.device.type, enabled=False):
            ema = ema_scan(y.float(), p_c)
            ridx = read_idx.clamp(0, ema.shape[1] - 1)
            up = gather_rows(ema, ridx)
            if self.start_vec is not None:
                up = torch.where((read_idx < 0).unsqueeze(-1), self.start_vec.float().expand_as(up), up)
            if ablate == "zero":
                up = torch.zeros_like(up)
            elif ablate == "shuffle":
                up = up.roll(1, dims=0)
            elif ablate is not None:
                raise ValueError(f"ablate {ablate!r}: expected None, 'zero' or 'shuffle'")
            z = up * ste(conf).unsqueeze(-1) + resid
        return z.to(dtype)

    def compile_glue(self, dynamic: bool):
        """Compile the chunking glue. `dynamic` is False only for a level whose input length never changes."""
        self._pre_fn = torch.compile(self._pre, dynamic=dynamic)
        self._gather_fn = torch.compile(self._gather_in, dynamic=True)
        self._scatter_fn = torch.compile(self._scatter_out, dynamic=True)
        self._pack_fn = torch.compile(pack_index, dynamic=True)
        if self.main_read is not None:
            self._read_fn = torch.compile(self.main_read.forward, dynamic=True)

    def forward(self, h, lengths, pos, pack: Pack | None = None, given=None, nxt=None):
        """given: frozen boundaries, a list of (B, L_bytes) bool masks, one per stage from this one inward
        (hnet/boundaries.py); this stage reads its own at the byte position of each input. None: the router decides.
        nxt: (B,L) next byte per input position, for a stage-0 router with needs_next (HNetLM.forward); inner
        stages never get it."""
        e = self.encoder(h, pack)
        g = None if given is None else gather_rows(given[0], pos)
        resid, p, mask, conf, read_idx, counts, rl, frac, p_mean, router_loss = self._pre_fn(e, lengths, g, nxt)
        M, total, m_max = chunk_stats(counts, e.shape[1], self.multiple)  # the one host sync of this stage
        x, pos_in, p_c = self._gather_fn(e, p, mask, counts, pos, M)
        inner_pack = None
        if self.pack_inner and e.is_cuda and getattr(self.inner, "uses_pack", False):
            inner_pack = make_pack(counts, total, M, m_max, self.multiple, self._pack_fn)
        y, inner_aux = self.inner(x, counts, pos_in, inner_pack, None if given is None else given[1:])
        if isinstance(self.inner, Main):
            main_out, main_idx = y, read_idx
        else:
            main_out = inner_aux[0].main_out
            main_idx = compose_read_index(read_idx, inner_aux[0].main_idx)
        # an ablated probe pass runs the eager glue, so the compiled training graph never sees `ablate`
        scatter = self._scatter_fn if self.ablate is None else partial(self._scatter_out, ablate=self.ablate)
        z = scatter(y, p_c, read_idx, conf, resid, e.dtype)
        if self.main_read is not None:
            r = self._read_fn(e, main_out[..., : self.d], main_idx)
            mode = self.ablate or self.read_ablate
            if mode == "zero":
                r = torch.zeros_like(r)
            elif mode == "shuffle":
                r = r.roll(1, dims=0)
            elif mode is not None:
                raise ValueError(f"ablate {mode!r}: expected None, 'zero' or 'shuffle'")
            z = z + r.to(z.dtype)
        out = self.decoder(z, pack)
        aux = StageAux(ratio_loss=rl, frac=frac, counts=counts, lengths=lengths, pos=pos_in,
                       anchor=self.router.anchor, p_mean=p_mean,
                       target_ratio=self.target_ratio, loss_weight=self.loss_weight,
                       main_out=main_out, main_idx=main_idx, p=p.detach(), in_pos=pos, router_loss=router_loss)
        return out, [aux] + inner_aux

    def init_weights(self, std: float, n_res: int):
        n = n_res + self.encoder.height + self.decoder.height   # 2025 _init_weights: residual count over the levels
        init_stack(self.encoder, std, n)
        init_stack(self.decoder, std, n)
        if self.main_read is not None:
            self.main_read.init_weights(std)
        self.inner.init_weights(std, n)

    def stage_params(self):
        """Parameters owned by this stage, excluding the inner net."""
        ps = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.router.parameters())
        ps += list(self.residual_proj.parameters())
        ps += list(self.main_read.parameters()) if self.main_read is not None else []
        ps += [q for q in (self.pad_dim, self.start_vec) if q is not None]
        return ps

    def describe(self) -> dict:
        return dict(kind="stage", stage_idx=self.stage_idx, d=self.d, d_inner=self.d_inner,
                    multiple=self.multiple, target_ratio=self.target_ratio, loss_weight=self.loss_weight,
                    encoder=self.encoder.describe(), decoder=self.decoder.describe(),
                    router=self.router.describe(), inner=self.inner.describe(),
                    main_read=self.main_read.describe() if self.main_read is not None else None)


def compose_read_index(read_idx: torch.Tensor, inner_main_idx: torch.Tensor) -> torch.Tensor:
    """read_idx (B,L) into inner positions (-1: none), inner_main_idx (B,M) per inner position -> (B,L) main chunk."""
    up = torch.gather(inner_main_idx, 1, read_idx.clamp(min=0))
    return torch.where(read_idx < 0, torch.full_like(up, -1), up)


def init_stack(stack: Stack, std: float, n_residuals: int):
    """Normal(0, std); output projections scaled by 1/sqrt(n_residuals) (2025 _init_weights)."""
    for name, m in stack.named_modules():
        if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
            s = std / (n_residuals ** 0.5) if ("out_proj" in name or "fc2" in name) else std
            nn.init.normal_(m.weight, mean=0.0, std=s)
