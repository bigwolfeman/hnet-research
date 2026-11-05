"""Load the reference H-Net (goombalab/hnet) and our spike model behind one probe interface.

Pseudocode:
    paths: shims/ first on sys.path (torch flash_attn, Mamba-1 stub), then ignored/ref/hnet, then our spike
    load_ref(name):  HNetForCausalLM(config json) in bf16 <- ignored/ref/ckpt/<name>.pt ; eval
    ref stages:      stage s = the HNet module at depth s (backbone, backbone.main_network, ...)
                     its coarse path = dechunked inner output, combined in residual_func(out, resid, p)
    ref_forward(x):  x bytes with our doc separator 0 -> reference BOS 254 ; mask = all True (no packing)
                     returns logits (B,L,256) and per-stage boundary masks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                   # repo root
REF_DIR = ROOT / "ignored" / "ref" / "hnet"
CKPT_DIR = ROOT / "ignored" / "ref" / "ckpt"
OUR_SPIKE = ROOT / "lab" / "spikes" / "2026-10-30-two-stage-hnet-smoke"
for p in (HERE / "shims", REF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

REF_BOS = 254
OUR_SEP = 0


def ref_config(name: str):
    from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
    cfg = json.loads((REF_DIR / "configs" / f"{name}.json").read_text())
    return HNetConfig(**{k: v for k, v in cfg.items() if k not in ("attn_cfg", "ssm_cfg")},
                      attn_cfg=AttnConfig(**cfg["attn_cfg"]), ssm_cfg=SSMConfig(**cfg["ssm_cfg"]))


def load_ref(name: str, device) -> torch.nn.Module:
    from hnet.models.mixer_seq import HNetForCausalLM
    model = HNetForCausalLM(ref_config(name), device=device, dtype=torch.bfloat16)
    sd = torch.load(CKPT_DIR / f"{name}.pt", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    model.eval()
    return model


def ref_stages(model) -> list:
    """HNet modules from the outermost stage inward, excluding the innermost (main) one."""
    out, h = [], model.backbone
    while not h.is_innermost:
        out.append(h)
        h = h.main_network
    return out


def to_ref_bytes(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == OUR_SEP, torch.full_like(x, REF_BOS), x)


def ref_forward(model, x: torch.Tensor):
    """x (B,L) int64 bytes in our encoding. Returns (logits fp32-castable, list of RoutingModuleOutput per stage)."""
    xr = to_ref_bytes(x)
    mask = torch.ones_like(xr, dtype=torch.bool)
    out = model(xr, mask=mask)
    return out.logits, out.bpred_output


def import_our_spike():
    if str(OUR_SPIKE) not in sys.path:
        sys.path.insert(0, str(OUR_SPIKE))
