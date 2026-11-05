"""Coarse-path probe (probes.coarse_use, Stage.ablate): off by default, changes only what it says, cleans up after itself."""
import math

import pytest
import torch

from hnet import probes
from hnet.model import HNetLM
from tests.conftest import SHAPES, make_cfg, randomize


def build(shape, router="lookahead"):
    n_stages, stage_layers = SHAPES[shape]
    torch.manual_seed(0)
    return randomize(HNetLM(make_cfg(router, n_stages, stage_layers=stage_layers)).eval())


def batches(n=2, B=3, L=256):
    g = torch.Generator().manual_seed(3)
    out = []
    for _ in range(n):
        t = torch.randint(1, 256, (B, L + 1), generator=g)
        out.append((t[:, :-1], t[:, 1:]))
    return out


@pytest.mark.parametrize("shape", list(SHAPES))
def test_ablation_modes(shape):
    m = build(shape)
    x = batches(1)[0][0]
    with torch.no_grad():
        base = m(x).logits
        for st in m.stages():
            for mode in ("zero", "shuffle"):
                with probes.ablated(st, mode):
                    got = m(x).logits
                assert (got - base).abs().max().item() > 1e-4, f"{mode} at stage {st.stage_idx} changed nothing"
            assert st.ablate is None
        assert torch.equal(m(x).logits, base), "logits differ after the probe restored every stage"


def test_unknown_mode_raises():
    m = build("2stage")
    with probes.ablated(m.stages()[0], "bogus"), pytest.raises(ValueError, match="ablate"):
        m(batches(1)[0][0])
    assert m.stages()[0].ablate is None


@pytest.mark.parametrize("shape", list(SHAPES))
def test_coarse_use_keys(shape):
    m = build(shape).train()
    out = probes.coarse_use(m, batches(), torch.device("cpu"), torch.bfloat16)
    n = SHAPES[shape][0]
    want = {"probe/coarse_base_bpb"} | {f"probe/coarse_{k}_s{s}" for s in range(n) for k in ("zero", "shuffle")}
    assert set(out) == want
    assert all(math.isfinite(v) for v in out.values())
    assert m.training, "coarse_use must restore train mode"
    assert all(st.ablate is None for st in m.stages())
