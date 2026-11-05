"""RatioController: a signed integral term that holds each stage's ratio at its target, whichever way the LM pulls."""
import math

import pytest
import torch
import torch.nn.functional as F

from hnet.chunking import RatioController, ratio_loss
from hnet.model import HNetLM, total_loss
from tests.conftest import make_cfg, randomize


def test_direction_and_fixed_point():
    c = RatioController([4.0, 2.0, 4.0], lr=0.1, device="cpu")
    # stage 0 chunks too small (F > 1/N), stage 1 on target, stage 2 chunks too big
    c.update([torch.tensor(0.32), torch.tensor(0.5), torch.tensor(0.2)])
    assert c.lam[0] > 0 and c.lam[2] < 0
    assert math.isclose(float(c.lam[1]), 0.0, abs_tol=1e-7)
    sd = c.state_dict()
    c.update([torch.tensor(0.5)] * 3)
    c.load_state_dict(sd)
    assert torch.equal(c.lam, sd["lam"])


def test_total_loss_lambda_term():
    torch.manual_seed(0)
    m = randomize(HNetLM(make_cfg("lookahead", 2)).eval())
    x = torch.randint(1, 256, (2, 256))
    lengths = torch.full((2,), 256)
    with torch.no_grad():
        out = m(x, lengths)
    fixed, ce = total_loss(out, x, lengths)
    same, _ = total_loss(out, x, lengths, torch.zeros(len(out.aux)))
    assert torch.allclose(fixed, same)
    lam = torch.tensor([0.5, -2.0])
    got, _ = total_loss(out, x, lengths, lam)
    want = fixed + sum(float(l) * float(a.p_mean) for l, a in zip(lam, out.aux))
    assert math.isclose(float(got), want, abs_tol=1e-5)


@pytest.mark.parametrize("N,lm_ratio", [(2.0, 4.0), (2.0, 1.11), (4.0, 8.0), (4.0, 2.0)])
def test_holds_target_against_either_pull(N, lm_ratio):
    """Toy plant: boundary logits theta, boundary = p > 0.5. An 'LM' term (BCE toward graded per-position preferences
    whose own ratio is lm_ratio) pulls the ratio coarser or finer than N. The fixed 0.03 ratio loss alone settles near
    lm_ratio; with the integral term the ratio converges to N and lam takes the sign that cancels the pull.
    (A uniform pull lm * G is a bang-bang plant under Adam and limit-cycles; real LM preferences are graded.)"""
    torch.manual_seed(0)
    n = 4096
    ell = 2 * torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1 / lm_ratio)) + 2 * torch.randn(n)
    theta = torch.nn.Parameter(torch.randn(n))
    opt = torch.optim.Adam([theta], lr=0.01)
    ctl = RatioController([N], lr=0.01, device="cpu")
    valid = torch.ones(n, dtype=torch.bool)
    ratios = []
    for _ in range(2000):
        p = torch.sigmoid(theta)
        rl, frac, g = ratio_loss(p, p > 0.5, valid, N)
        loss = F.binary_cross_entropy_with_logits(theta, torch.sigmoid(ell)) + 0.03 * rl + ctl.lam[0] * g
        opt.zero_grad()
        loss.backward()
        opt.step()
        ctl.update([frac])
        ratios.append(1 / max(float(frac), 1e-9))
    tail = sum(ratios[-500:]) / 500
    assert abs(tail / N - 1) < 0.01, (N, lm_ratio, tail, float(ctl.lam[0]))
    assert (float(ctl.lam[0]) > 0) == (lm_ratio < N)   # LM wants finer chunks -> lam > 0 pushes G down
