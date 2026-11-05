"""Gate 1 (most important): no information flows from bytes > t to logits <= t.

Two forms, both fp32 on CPU:
  A. perturb: change bytes at positions > t (same lengths); logits[:, :t+1] must not change.
  B. prefix:  run the prefix x[:t+1] alone (padded, lengths = t+1); its logits must equal the full run's.
Form B also catches anything that depends on the sequence length or on the forced last boundary.
"""
import pytest
import torch

from hnet.model import HNetLM
from tests.conftest import SHAPES, make_cfg, randomize

L = 256
TOL = 1e-4


def build(router, shape, seed=0, **kw):
    n_stages, stage_layers = SHAPES[shape]
    torch.manual_seed(seed)
    m = HNetLM(make_cfg(router, n_stages, stage_layers=stage_layers, **kw)).eval()
    return randomize(m, seed=seed)


def pad_to(x, multiple):
    n = x.shape[1]
    tgt = -(-n // multiple) * multiple
    return torch.cat([x, torch.randint(0, 256, (x.shape[0], tgt - n))], 1)  # random pad bytes on purpose


@pytest.mark.parametrize("router", ["eoc", "eoc_lag", "lookahead"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_boundaries_nondegenerate(router, shape):
    """The causality tests are only meaningful if chunking is neither all-ones nor a single chunk."""
    m = build(router, shape)
    x = torch.randint(0, 256, (4, L))
    with torch.no_grad():
        out = m(x)
    for a in out.aux:
        assert 0.1 < float(a.frac) < 0.9, f"degenerate chunking frac={float(a.frac)}"


@pytest.mark.parametrize("router", ["eoc", "eoc_lag", "lookahead"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("seed", [0, 1])
def test_perturb_future(router, shape, seed):
    m = build(router, shape, seed)
    g = torch.Generator().manual_seed(100 + seed)
    x = torch.randint(0, 256, (3, L), generator=g)
    with torch.no_grad():
        base = m(x).logits
        for t in (0, 5, 31, 77, 130, 200, L - 2):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, 256, x2[:, t + 1:].shape, generator=g)
            got = m(x2).logits
            err = (got[:, : t + 1] - base[:, : t + 1]).abs().max().item()
            assert err < TOL, f"LEAK {router} shape={shape} t={t}: max err {err}"
            # sanity: the future logits do change, so the perturbation is real
            if t < L - 8:
                assert (got[:, t + 1:] - base[:, t + 1:]).abs().max().item() > 1e-3


@pytest.mark.parametrize("router", ["eoc", "eoc_lag", "lookahead"])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("multiple", [32, 128])
def test_prefix_equals_full(router, shape, multiple):
    m = build(router, shape, multiple=multiple)
    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, 256, (2, 512), generator=g)
    with torch.no_grad():
        base = m(x).logits
        for t in (0, 9, 63, 100, 255, 400):
            n = t + 1
            xp = pad_to(x[:, :n], multiple)
            got = m(xp, torch.full((2,), n)).logits[:, :n]
            err = (got - base[:, :n]).abs().max().item()
            assert err < TOL, f"PREFIX MISMATCH {router} shape={shape} t={t}: {err}"
