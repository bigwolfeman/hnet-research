"""Gate 4: forward/backward finite for both routers, 1 and 2 stages; plus flops sanity and registry."""
import math

import pytest
import torch

from hnet.flops import flops_per_byte
from hnet.model import HNetLM, measured_ratios, total_loss
from hnet.routers import ROUTERS
from tests.conftest import make_cfg


@pytest.mark.parametrize("router", ["eoc", "eoc_lag", "lookahead"])
@pytest.mark.parametrize("n_stages", [1, 2])
def test_fwd_bwd_finite(router, n_stages):
    torch.manual_seed(0)
    cfg = make_cfg(router, n_stages)
    m = HNetLM(cfg)
    x = torch.randint(0, 256, (3, 257))
    xin, y = x[:, :256], x[:, 1:]
    lengths = torch.tensor([256, 200, 130])
    out = m(xin, lengths)
    loss, ce = total_loss(out, y, lengths)
    assert torch.isfinite(loss)
    assert abs(float(ce) - math.log(256)) < 0.5  # untrained: near uniform
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    # router must receive gradient (ratio loss / EMA / STE)
    rg = [p.grad.abs().sum() for n, p in m.named_parameters() if "router" in n]
    assert all(g > 0 for g in rg)
    assert len(out.aux) == n_stages and len(measured_ratios(out.aux)) == n_stages


def test_registry():
    assert {"eoc", "lookahead"} <= set(ROUTERS)


def test_flops_monotone_in_ratio():
    cfg = make_cfg("eoc", 2)
    a = flops_per_byte(cfg, 4096, [4.0, 2.0])["fwd"]
    b = flops_per_byte(cfg, 4096, [2.0, 2.0])["fwd"]
    assert b > a  # a lower compression ratio costs more FLOPs per byte


def test_flops_hand_count():
    # 1 stage, ratio 1: one SWA block + head, L <= window so ctx = (L+1)/2
    cfg = make_cfg("lookahead", 1)
    cfg["stages"][0].update(enc_layers=1, dec_layers=0, window=10_000)
    cfg["main"]["layers"] = 0
    d0, d1, V, L = 32, 64, 256, 128
    h0 = int(-(-d0 * cfg["mlp_ratio"] // cfg["mlp_multiple_of"]) * cfg["mlp_multiple_of"])
    exp = 2 * d0 * V + (8 * d0 * d0 + 6 * d0 * h0 + 4 * d0 * (L + 1) / 2) + 4 * d0 * d0 + 2 * d0 * d0
    assert abs(flops_per_byte(cfg, L, [1.0])["fwd"] - exp) < 1e-6
