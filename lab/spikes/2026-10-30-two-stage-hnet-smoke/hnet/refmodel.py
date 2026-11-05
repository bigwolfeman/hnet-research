"""RefHNetLM: the reference H-Net (goombalab/hnet @ 3673fe1) behind our HNetLM interface (model.kind == "ref_hnet").

The trainer, evals, probes, ladder, and resume code then run unchanged on the reference architecture.
Experiment: lab/experiments/successes/2026-11-01-reference-hnet-at-our-scale.md

Needs the reference probe spike's environment (lab/spikes/2026-11-01-reference-hnet-probe/README.md): mamba_ssm,
causal_conv1d, the torch flash_attn shim, and `refnet` (the reference package renamed by vendor_refnet.py so it can
share a process with our `hnet`).

Pseudocode:
    build:   refnet.HNetForCausalLM(HNetConfig(cfg.ref, d_model, vocab)) ; reference init_weights(init_std)
    forward(x, lengths):
        mask = position < lengths
        logits, routers = ref(x, mask)                                   # mask mode, no packing
        for stage s (outermost first):
            valid_s = input positions of stage s (stage 0: mask; stage s+1: the first counts_s chunk slots)
            F_s = boundary fraction over valid_s ;  G_s = mean boundary prob over valid_s (keeps grad)
            ratio_loss_s = (1-F)(1-G) + F G (N-1)) N/(N-1)               # reference load_balancing_loss
            pos_s = byte position of every stage-s anchor (start anchors)
        return LMOutput(logits, [StageAux ...])
    stages():  handles whose `.ablate` wraps that stage's residual_func: zero / shuffle the dechunked inner output
    stage_param_groups():  reference _apply_lr_multiplier ownership: every parameter of the depth-s HNet -> s,
                           deeper HNets override; embeddings and LM head -> 0; the innermost main network -> n_stages
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .stage import StageAux

_ROOT = Path(__file__).resolve().parents[4]
for _p in (_ROOT / "lab" / "spikes" / "2026-11-01-reference-hnet-probe" / "shims", _ROOT / "ignored" / "ref"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from refnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig  # noqa: E402
from refnet.models.mixer_seq import HNetForCausalLM  # noqa: E402


def _sorted_anchor_index(mask: torch.Tensor) -> torch.Tensor:
    """(B,L) bool -> (B,L) indices with the True positions first, in order (the reference ChunkLayer's ordering)."""
    L = mask.shape[1]
    return torch.argsort(torch.arange(L, device=mask.device)[None, :] + (~mask).long() * L, dim=1)


class RefStage:
    """Probe handle for one reference HNet level. Setting `.ablate` to 'zero' / 'shuffle' / None patches the
    level's residual_func, which receives the dechunked inner output (the coarse path)."""

    def __init__(self, hnet):
        self.hnet = hnet

    @property
    def ablate(self):
        return getattr(self.hnet, "_probe_ablate", None)

    @ablate.setter
    def ablate(self, mode):
        h = self.hnet
        if not hasattr(h, "_residual_func_orig"):
            h._residual_func_orig = h.residual_func
        orig = h._residual_func_orig
        if mode is None:
            h.residual_func = orig
        elif mode == "zero":
            h.residual_func = lambda out, resid, p: orig(torch.zeros_like(out), resid, p)
        elif mode == "shuffle":
            h.residual_func = lambda out, resid, p: orig(out.roll(1, dims=0), resid, p)
        else:
            raise ValueError(f"ablate {mode!r}: expected None, 'zero' or 'shuffle'")
        h._probe_ablate = mode


class RefHNetLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        r = cfg["ref"]
        self.vocab = cfg["vocab_size"]
        self.multiple = cfg["chunk_multiple"]
        self.n_stages = len(cfg["stages"])
        self.targets = [float(s["router"]["target_ratio"]) for s in cfg["stages"]]
        self.loss_weights = [float(s["router"]["loss_weight"]) for s in cfg["stages"]]
        assert all(n > 1 for n in self.targets)
        hcfg = HNetConfig(arch_layout=r["arch_layout"], d_model=list(cfg["d_model"]),
                          d_intermediate=list(r["d_intermediate"]), vocab_size=self.vocab,
                          ssm_cfg=SSMConfig(**r["ssm_cfg"]), attn_cfg=AttnConfig(**r["attn_cfg"]),
                          tie_embeddings=bool(r.get("tie_embeddings", False)))
        self.ref = HNetForCausalLM(hcfg)
        self.ref.init_weights(initializer_range=cfg["init_std"])
        assert len(self._levels()) == self.n_stages + 1, "arch_layout depth must match len(stages)"

    def _levels(self) -> list:
        out, h = [], self.ref.backbone
        while True:
            out.append(h)
            if h.is_innermost:
                return out
            h = h.main_network

    def forward(self, x, lengths=None):
        from .model import LMOutput
        B, L = x.shape
        if lengths is None:
            lengths = torch.full((B,), L, device=x.device, dtype=torch.long)
        mask = torch.arange(L, device=x.device)[None, :] < lengths[:, None]
        out = self.ref(x, mask=mask)
        aux, valid, pos = [], mask, torch.arange(L, device=x.device)[None, :].expand(B, -1)
        stage_lengths = lengths
        for s, bo in enumerate(out.bpred_output):
            bmask = bo.boundary_mask & valid                       # (B, L_s)
            prob = bo.boundary_prob[..., -1].float()
            v = valid.float()
            nv = v.sum().clamp_min(1.0)
            F = bmask.float().sum() / nv
            G = (prob * v).sum() / nv
            N = self.targets[s]
            rl = ((1 - F) * (1 - G) + F * G * (N - 1)) * N / (N - 1)
            counts = bmask.sum(dim=1)
            M = int(counts.max())
            idx = _sorted_anchor_index(bmask)[:, :M]
            pos_s = torch.gather(pos, 1, idx)
            aux.append(StageAux(ratio_loss=rl, frac=F.detach(), counts=counts, lengths=stage_lengths, pos=pos_s,
                                anchor="start", p_mean=G, target_ratio=N, loss_weight=self.loss_weights[s],
                                p=prob.detach(), in_pos=pos))
            valid = torch.arange(M, device=x.device)[None, :] < counts[:, None]
            pos, stage_lengths = pos_s, counts
        return LMOutput(out.logits, aux)

    def stages(self) -> list[RefStage]:
        return [RefStage(h) for h in self._levels()[:-1]]

    def compile_stacks(self):
        """No-op: Mamba-2 runs fused Triton kernels; the reference code is not torch.compile-clean."""

    def stage_param_groups(self):
        owner: dict[int, int] = {}
        for lvl, h in enumerate(self._levels()):
            for p in h.parameters():
                owner[id(p)] = lvl                                  # deeper levels overwrite, as in the reference
        for p in list(self.ref.embeddings.parameters()) + list(self.ref.lm_head.parameters()):
            owner[id(p)] = 0
        out: dict[int, list] = {}
        for name, p in self.named_parameters():
            out.setdefault(owner[id(p)], []).append((name, p))
        return sorted(out.items())

    def describe(self) -> dict:
        return dict(kind="ref_hnet", vocab_size=self.vocab, chunk_multiple=self.multiple,
                    d_model=list(self.cfg["d_model"]), ref=self.cfg["ref"], targets=self.targets,
                    loss_weights=self.loss_weights, init_std=self.cfg["init_std"],
                    params=sum(p.numel() for p in self.parameters()),
                    params_per_level={lvl: sum(p.numel() for _, p in ps) for lvl, ps in self.stage_param_groups()})


def ref_state_to_ours(ref: RefHNetLM, ours: nn.Module) -> dict[str, torch.Tensor]:
    """State dict for OUR HNetLM holding the reference model's weights, level by level.

    Needs the same configuration in both (e.g. ref_hnet_2stage and reflayout_2stage). Mapping:
        embeddings -> embed ; lm_head -> head
        level s (HNet):  encoder / decoder (Isotropic) -> Stage.encoder / Stage.decoder (Stack)
                         routing_module.{q,k}_proj_layer -> router.{q,k}_proj ; residual_proj -> residual_proj
                         child HNet's pad_dimension -> Stage.pad_dim
        innermost main network (Isotropic) -> Main.stack
        Isotropic layer i -> Block i: norm1, norm2 ; Mamba2 mixer -> mamba (same module) ;
            attention Wqkv -> attn.qkv, out_proj -> attn.out_proj ;
            SwiGLU fc1 = [value | gate] -> gate_up = [gate | value] (our SwiGLU gates the first half) ; fc2 -> fc2
    Fails loudly on any type, shape, or coverage mismatch.
    """
    from .stage import Main, Stage

    out: dict[str, torch.Tensor] = {}

    def put(name, t, like):
        assert tuple(t.shape) == tuple(like.shape), (name, tuple(t.shape), tuple(like.shape))
        out[name] = t.detach().clone()

    ours_sd = ours.state_dict()

    def stack(prefix: str, iso):
        blocks = getattr(ours.get_submodule(prefix), "blocks")
        assert len(blocks) == len(iso.layers), (prefix, len(blocks), len(iso.layers))
        for i, (blk, lay) in enumerate(zip(blocks, iso.layers)):
            p = f"{prefix}.blocks.{i}"
            put(f"{p}.norm1.weight", lay.norm1.weight, ours_sd[f"{p}.norm1.weight"])
            if blk.mixer == "mamba":
                from mamba_ssm.modules.mamba2 import Mamba2   # the reference's Mamba2Wrapper only adds inference step()
                assert isinstance(lay.mixer, Mamba2) and type(blk.mamba) is Mamba2, (p, type(lay.mixer).__name__)
                for k, v in lay.mixer.state_dict().items():
                    put(f"{p}.mamba.{k}", v, ours_sd[f"{p}.mamba.{k}"])
            else:
                assert hasattr(lay.mixer, "Wqkv"), (p, type(lay.mixer).__name__)
                assert lay.mixer.Wqkv.bias is None and lay.mixer.out_proj.bias is None, p
                put(f"{p}.attn.qkv.weight", lay.mixer.Wqkv.weight, ours_sd[f"{p}.attn.qkv.weight"])
                put(f"{p}.attn.out_proj.weight", lay.mixer.out_proj.weight, ours_sd[f"{p}.attn.out_proj.weight"])
            has_mlp = getattr(lay, "mlp", None) is not None
            assert has_mlp == (blk.mlp is not None), (p, has_mlp)
            if has_mlp:
                value, gate = lay.mlp.fc1.weight.chunk(2, dim=0)
                put(f"{p}.mlp.gate_up.weight", torch.cat([gate, value], 0), ours_sd[f"{p}.mlp.gate_up.weight"])
                put(f"{p}.mlp.fc2.weight", lay.mlp.fc2.weight, ours_sd[f"{p}.mlp.fc2.weight"])
                put(f"{p}.norm2.weight", lay.norm2.weight, ours_sd[f"{p}.norm2.weight"])
        put(f"{prefix}.final_norm.weight", iso.rmsnorm.weight, ours_sd[f"{prefix}.final_norm.weight"])

    put("embed.weight", ref.ref.embeddings.weight, ours_sd["embed.weight"])
    put("head.weight", ref.ref.lm_head.weight, ours_sd["head.weight"])
    levels, net, prefix = ref._levels(), ours.net, "net"
    for s, h in enumerate(levels):
        if h.is_innermost:
            assert isinstance(net, Main), (prefix, type(net).__name__)
            stack(f"{prefix}.stack", h.main_network)
            break
        assert isinstance(net, Stage), (prefix, type(net).__name__)
        stack(f"{prefix}.encoder", h.encoder)
        stack(f"{prefix}.decoder", h.decoder)
        put(f"{prefix}.router.q_proj.weight", h.routing_module.q_proj_layer.weight, ours_sd[f"{prefix}.router.q_proj.weight"])
        put(f"{prefix}.router.k_proj.weight", h.routing_module.k_proj_layer.weight, ours_sd[f"{prefix}.router.k_proj.weight"])
        for k in ("weight", "bias"):
            put(f"{prefix}.residual_proj.{k}", getattr(h.residual_proj, k), ours_sd[f"{prefix}.residual_proj.{k}"])
        child_pad = getattr(levels[s + 1], "pad_dimension", None)
        assert (child_pad is None) == (net.pad_dim is None), (prefix, child_pad is None)
        if child_pad is not None:
            put(f"{prefix}.pad_dim", child_pad, ours_sd[f"{prefix}.pad_dim"])
        net, prefix = net.inner, f"{prefix}.inner"
    missing = sorted(set(ours_sd) - set(out))
    assert not missing, f"not mapped: {missing}"
    return out
