"""Main.loops (hnet/stage.py): the main stack runs R times, weight-tied; FLOPs and init count the looped depth."""
import copy

import pytest
import torch

from hnet.flops import flops_per_byte
from hnet.model import HNetLM, build_model
from tests.conftest import make_cfg, randomize


def _cfg(layers, loops):
    c = make_cfg("lookahead", 1)
    c["main"] = dict(layers=layers, loops=loops)
    return c


def test_loops_apply_the_same_stack_r_times():
    torch.manual_seed(0)
    m = randomize(HNetLM(_cfg(2, 3)), seed=0).eval()
    main = m.net.inner
    h = torch.randn(2, 32, 64)
    with torch.no_grad():
        ref = main.stack(main.stack(main.stack(h)))
        out, aux = main(h, None, None)
    assert torch.equal(out, ref) and aux == []


def test_params_tied_and_flops_scale():
    p1 = sum(p.numel() for p in HNetLM(_cfg(2, 1)).parameters())
    p3 = sum(p.numel() for p in HNetLM(_cfg(2, 3)).parameters())
    assert p1 == p3
    f1 = flops_per_byte(_cfg(2, 1), 1024, [4.0])["breakdown"]["main"]
    f3 = flops_per_byte(_cfg(2, 3), 1024, [4.0])["breakdown"]["main"]
    assert f3 == pytest.approx(3 * f1)
    assert flops_per_byte(_cfg(2, 3), 1024, [4.0])["breakdown"]["main"] == pytest.approx(
        flops_per_byte(_cfg(6, 1), 1024, [4.0])["breakdown"]["main"])


def test_init_counts_looped_depth():
    """Output projections scale by 1/sqrt(n_res + R * height): a T2 x 3 main matches a T6 main's out_proj std."""
    torch.manual_seed(0)
    a = HNetLM(_cfg(2, 3)).net.inner.stack
    torch.manual_seed(0)
    b = HNetLM(_cfg(6, 1)).net.inner.stack
    sa = torch.cat([p.flatten() for n, p in a.named_parameters() if "out_proj" in n or "fc2" in n]).std().item()
    sb = torch.cat([p.flatten() for n, p in b.named_parameters() if "out_proj" in n or "fc2" in n]).std().item()
    assert sa == pytest.approx(sb, rel=0.05)


def test_loops_must_be_positive():
    with pytest.raises(ValueError, match="loops"):
        HNetLM(_cfg(2, 0))


def test_ref_hnet_rejects_loops():
    c = dict(kind="ref_hnet", main=dict(layers=2, loops=2))
    with pytest.raises(ValueError, match="no looped main"):
        build_model(c)
