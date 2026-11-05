import copy
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(4)


ROUTER_VARIANTS = {
    "eoc": ("eoc", {"read_own_end": True, "force_last": False}),
    "eoc_lag": ("eoc", {"read_own_end": False, "force_last": True}),
    "lookahead": ("lookahead", {}),
}


def make_cfg(router: str = "eoc", n_stages: int = 2, multiple: int = 32, attn_impl: str = "sdpa_mask",
             ratios=(4.0, 2.0), force_last: bool | None = None, stage_layers=None, main_read: dict | None = None) -> dict:
    """Tiny model config for CPU tests (fp32). stage_layers: [(enc, dec)] per stage; default 1/1 everywhere."""
    stages = []
    windows = [64, 32, 16]
    for s in range(n_stages):
        name, kw = ROUTER_VARIANTS[router]
        kw = dict(kw) if force_last is None or name != "eoc" else dict(kw, force_last=force_last)
        enc, dec = stage_layers[s] if stage_layers else (1, 1)
        stages.append(dict(enc_layers=enc, dec_layers=dec, window=windows[s], main_read=main_read,
                           router=dict(name=name, target_ratio=ratios[s], loss_weight=0.03, kwargs=kw)))
    return dict(vocab_size=256, chunk_multiple=multiple, d_model=[32, 48, 64][: n_stages + 1] if n_stages == 2 else [32, 64],
                head_dim=16, mlp_ratio=2.6667, mlp_multiple_of=16, rope_theta=10000.0, init_std=0.02,
                attn_impl=attn_impl, main=dict(layers=2), stages=stages)


# (n_stages, stage_layers) shapes every structural test covers. "thin": stage 1 has no blocks, only chunk/dechunk.
SHAPES = {"1stage": (1, None), "2stage": (2, None), "2stage_thin": (2, [(1, 1), (0, 0)])}


def randomize(model: torch.nn.Module, std: float = 0.15, seed: int = 0):
    """Break zero/identity inits so a leak anywhere (residual, pad dims, start vector, router) is visible."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n and n.endswith("weight"):
                continue
            if n.startswith("embed"):
                continue
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return model


@pytest.fixture
def cfg_factory():
    return make_cfg
