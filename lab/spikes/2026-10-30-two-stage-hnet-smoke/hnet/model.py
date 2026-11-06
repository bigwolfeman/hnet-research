"""HNetLM: byte embedding -> recursive Stage/Main -> LM head, built from a plain config dict.

Config (see configs/model/*.yaml):
    d_model:  [d_0, ..., d_N]       N stages + main; widths must be non-decreasing
    stages:   [{enc_layers, dec_layers, window, attn_impl?, enc_layout?, dec_layout?,
                router:{name,target_ratio,loss_weight,kwargs}}, ...]
              enc_layout / dec_layout (hnet/blocks.py parse_layout, e.g. "M2") replace enc_layers / dec_layers;
              without them a stage has "T<enc_layers>" / "T<dec_layers>" (attention + MLP blocks)
    main:     {layers, loops?}          loops R >= 1: the main stack runs R times, weight-tied (default 1)
              {kind: parcae, layers: null, parcae: {prelude, core, coda, mean_recurrence, backprop_depth,
               decay_init, seed}}: the Parcae looped main network (hnet/parcae.py)
    ssm:      {d_state, d_conv, expand, headdim, chunk_size}   Mamba-2 settings for m/M blocks (hnet/ssm.py)
    rope_dim: dims of each head that RoPE rotates (null = all; the reference H-Net rotates 32 of 64)
    pack_inner: inner levels run their stacks on valid positions only (hnet/packing.py; CUDA; same math)
    boundaries: {kind: entropy, proxy, thresholds}: frozen boundaries from a proxy byte LM (hnet/boundaries.py);
              every stage router must then be "given". null: the routers decide (end-to-end)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Stack
from .boundaries import build_boundaries
from .mainread import MainRead
from .parcae import ParcaeMain
from .routers import build_router
from .stage import Main, Stage, StageAux


@dataclass
class LMOutput:
    logits: torch.Tensor
    aux: list[StageAux]


def _cfg_get(cfg, key, default=None):
    return cfg[key] if key in cfg else default


def stage_layouts(stage: dict) -> tuple[str, str]:
    """(encoder layout, decoder layout) of one stage config. Shared by the model builder and the FLOPs counter."""
    enc = _cfg_get(stage, "enc_layout") or f"T{stage['enc_layers']}"
    dec = _cfg_get(stage, "dec_layout") or f"T{stage['dec_layers']}"
    return enc, dec


class HNetLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = cfg["vocab_size"]
        self.multiple = cfg["chunk_multiple"]
        dims = list(cfg["d_model"])
        stages = list(cfg["stages"])
        kind = _cfg_get(cfg, "kind", "hnet")
        assert kind in ("hnet", "token"), f"model.kind {kind!r}"
        # kind=token: the same net over BPE ids (vocab_size = tokenizer size); an isotropic Transformer has no stages.
        assert kind == "hnet" or not stages, "a token model has no stages"
        assert len(dims) == len(stages) + 1, "d_model needs one entry per stage plus one for main"
        self.n_stages = len(stages)
        self.embed = nn.Embedding(self.vocab, dims[0])
        self.net = self._build(0, dims, stages)
        self.head = nn.Linear(dims[0], self.vocab, bias=False)
        self.init_weights()
        given = [s["router"]["name"] == "given" for s in stages]
        bcfg = _cfg_get(cfg, "boundaries")
        if any(given) and not (all(given) and bcfg):
            raise ValueError("router 'given' needs model.boundaries, on every stage")
        if bcfg and not all(given):
            raise ValueError("model.boundaries is set: every stage router must be 'given'")
        # plain attribute, not a submodule: the frozen proxy stays out of parameters(), the optimizer and state_dict
        self.boundary_source = build_boundaries(bcfg, self.n_stages, "cpu")
        deep = [i for i, st in enumerate(self.stages()) if st.router.needs_next and i > 0]
        if deep:
            raise ValueError(f"routers that need the next byte work at stage 0 only; stages {deep} have one")
        self.needs_next = bool(self.stages()) and self.stages()[0].router.needs_next

    def _apply(self, fn, recurse=True):
        """.to() / .cuda() also move the boundary proxy (not a registered submodule)."""
        super()._apply(fn, recurse)
        if getattr(self, "boundary_source", None) is not None:
            self.boundary_source.proxy._apply(fn, recurse)
        return self

    def _stack(self, d, layout, window, impl=None):
        c = self.cfg
        return Stack(d, layout, c["head_dim"], window, c["mlp_ratio"], c["mlp_multiple_of"], c["rope_theta"],
                     impl or c["attn_impl"], ssm=_cfg_get(c, "ssm"), rope_dim=_cfg_get(c, "rope_dim"))

    def _build(self, level, dims, stages):
        if level == len(stages):
            m = self.cfg["main"]
            if _cfg_get(m, "kind", "plain") == "parcae":
                return self._parcae(dims[level], m)
            return Main(self._stack(dims[level], f"T{m['layers']}", None), loops=int(_cfg_get(m, "loops", 1)))
        s = stages[level]
        r = s["router"]
        enc, dec = stage_layouts(s)
        mr = _cfg_get(s, "main_read")
        return Stage(
            encoder=self._stack(dims[level], enc, s["window"], s.get("attn_impl")),
            router=build_router(r["name"], dims[level], **dict(r.get("kwargs") or {})),
            inner=self._build(level + 1, dims, stages),
            decoder=self._stack(dims[level], dec, s["window"], s.get("attn_impl")),
            d=dims[level], d_inner=dims[level + 1], multiple=self.multiple,
            target_ratio=float(r["target_ratio"]), loss_weight=float(r["loss_weight"]), stage_idx=level,
            main_read=MainRead(dims[level], int(mr["head_dim"])) if mr else None,
            pack_inner=bool(_cfg_get(self.cfg, "pack_inner", False)))

    def _parcae(self, d, m):
        """main.kind parcae (hnet/parcae.py). main.layers and main.loops must be unset: the depths are prelude/core/coda."""
        if _cfg_get(m, "layers") is not None or int(_cfg_get(m, "loops", 1)) != 1:
            raise ValueError("main.kind parcae: set main.layers null and main.loops 1 (depths are prelude/core/coda)")
        p = m["parcae"]
        stack = lambda n, norm: Stack(d, f"T{n}", self.cfg["head_dim"], None, self.cfg["mlp_ratio"],
                                      self.cfg["mlp_multiple_of"], self.cfg["rope_theta"], self.cfg["attn_impl"],
                                      ssm=_cfg_get(self.cfg, "ssm"), rope_dim=_cfg_get(self.cfg, "rope_dim"),
                                      final_norm=norm)
        return ParcaeMain(stack(p["prelude"], True), stack(p["core"], False), stack(p["coda"], True),
                          mean_recurrence=int(p["mean_recurrence"]), backprop_depth=int(p["backprop_depth"]),
                          decay_init=float(p["decay_init"]), state_std=float(self.cfg["init_std"]),
                          seed=int(p["seed"]))

    def init_weights(self):
        std = self.cfg["init_std"]
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0)  # 2025: std 1 embeddings
        nn.init.normal_(self.head.weight, mean=0.0, std=std)
        self.net.init_weights(std, 0)

    def forward(self, x, lengths=None) -> LMOutput:
        B, L = x.shape
        assert L % self.multiple == 0, f"pad sequence to a multiple of {self.multiple}"
        if lengths is None:
            lengths = torch.full((B,), L, device=x.device, dtype=torch.long)
        pos = torch.arange(L, device=x.device)[None, :].expand(B, -1)
        given = None if self.boundary_source is None else self.boundary_source(x, lengths)
        if self.needs_next:   # the byte after each position; -100 past the last valid byte (router "surprise")
            nxt = torch.where(pos[:, :-1] < (lengths[:, None] - 1), x[:, 1:].long(), -100)
            nxt = torch.cat([nxt, torch.full_like(nxt[:, :1], -100)], dim=1)
            h, aux = self.net(self.embed(x), lengths, pos, given=given, nxt=nxt)
        else:
            h, aux = self.net(self.embed(x), lengths, pos, given=given)
        return LMOutput(self.head(h), aux)

    def stages(self) -> list[Stage]:
        """Stages from the outermost (byte level) inward. Empty for a byte or token Transformer."""
        out, net = [], self.net
        while isinstance(net, Stage):
            out.append(net)
            net = net.inner
        return out

    def compile_stacks(self):
        """Stage 0 sees fixed lengths (static shapes). Deeper levels see data-dependent lengths (dynamic shapes)."""
        if self.boundary_source is not None:
            self.boundary_source.proxy.compile_stacks()
        net, level = self.net, 0
        while True:
            if isinstance(net, Main):
                net.compile_blocks(dynamic=level > 0)
                break
            net.compile_glue(dynamic=level > 0)
            net.encoder.compile_blocks(dynamic=level > 0)
            net.decoder.compile_blocks(dynamic=level > 0)
            net, level = net.inner, level + 1

    # ---- parameters grouped by stage (per-stage LR multipliers, 2025 apply_lr_multiplier) ----
    def stage_param_groups(self) -> list[tuple[int, list[tuple[str, nn.Parameter]]]]:
        """[(stage_level, [(name, param)])]; level == n_stages is the main net. Embedding/head join level 0."""
        owner: dict[int, int] = {}
        net, level = self.net, 0
        while True:
            for p in net.stage_params():
                owner[id(p)] = level
            if isinstance(net, Main):
                break
            net, level = net.inner, level + 1
        for p in list(self.embed.parameters()) + list(self.head.parameters()):
            owner[id(p)] = 0
        out: dict[int, list] = {}
        for name, p in self.named_parameters():
            out.setdefault(owner[id(p)], []).append((name, p))
        return sorted(out.items())

    def describe(self) -> dict:
        return dict(vocab_size=self.vocab, chunk_multiple=self.multiple, d_model=list(self.cfg["d_model"]),
                    attn_impl=self.cfg["attn_impl"], init_std=self.cfg["init_std"],
                    params=sum(p.numel() for p in self.parameters()), tree=self.net.describe(),
                    boundaries=None if self.boundary_source is None else self.boundary_source.describe())


def build_model(cfg) -> nn.Module:
    """Model from a config dict: kind hnet / token -> HNetLM ; kind ref_hnet -> RefHNetLM (reference H-Net code,
    imported only here because it needs the reference environment: hnet/refmodel.py)."""
    if _cfg_get(cfg, "kind", "hnet") == "ref_hnet":
        m = cfg.get("main") or {}
        if int(_cfg_get(m, "loops", 1)) != 1 or _cfg_get(m, "kind", "plain") != "plain":
            raise ValueError("main.loops / main.kind: the reference H-Net code (kind ref_hnet) has no looped main")
        from .refmodel import RefHNetLM
        return RefHNetLM(cfg)
    return HNetLM(cfg)


def lm_loss(logits, y, lengths):
    """Mean next-byte CE (nats) over valid positions."""
    B, L, V = logits.shape
    valid = torch.arange(L, device=y.device)[None, :] < lengths[:, None]
    ce = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1), reduction="none").view(B, L)
    return (ce * valid).sum() / valid.sum()


def total_loss(out: LMOutput, y, lengths, lam: torch.Tensor | None = None):
    """CE + sum_s (loss_weight_s * ratio_loss_s + router_loss_s + lam_s * G_s). lam: per-stage RatioController
    multipliers; None = off. router_loss: a router's own weighted loss (router "surprise"), else none."""
    ce = lm_loss(out.logits, y, lengths)
    if not out.aux:
        return ce, ce
    loss = ce + sum(a.loss_weight * a.ratio_loss for a in out.aux)
    loss = loss + sum(a.router_loss for a in out.aux if a.router_loss is not None)
    if lam is not None:
        loss = loss + (lam * torch.stack([a.p_mean for a in out.aux])).sum()
    return loss, ce


def measured_ratios(aux: list[StageAux]) -> list[float]:
    """Per-stage bytes-per-chunk relative to the stage's own input (1 / anchor fraction)."""
    return [1.0 / max(float(a.frac), 1e-9) for a in aux]


LN2 = math.log(2.0)
