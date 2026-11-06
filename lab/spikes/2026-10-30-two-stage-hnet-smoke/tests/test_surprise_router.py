"""Router "surprise" (hnet/routers.py): lookahead cosine plus the encoder's own next-byte entropy, stage 0 only.
Causality is in tests/test_causality.py. Here: the aux CE trains the head and the encoder, the entropy reaches p only
through beta and tau (detached), a repeated pattern lowers the entropy, and misuse fails loud."""
import pytest
import torch

from hnet.model import HNetLM, total_loss
from tests.conftest import make_cfg


def build(seed=0, n_stages=2):
    torch.manual_seed(seed)
    return HNetLM(make_cfg("surprise", n_stages))


def test_loss_includes_router_ce_and_grads_flow():
    m = build()
    x = torch.randint(0, 256, (2, 128))
    out = m(x)
    a0 = out.aux[0]
    assert a0.router_loss is not None and torch.isfinite(a0.router_loss)
    assert all(a.router_loss is None for a in out.aux[1:])
    loss, ce = total_loss(out, x, torch.full((2,), 128))
    assert float(loss.detach()) > float(ce.detach()) + 0.5 * float(a0.router_loss.detach())   # ln 256 scale CE at init is in the loss
    loss.backward()
    r = m.stages()[0].router
    for name, p in (("lm", r.lm.weight), ("beta", r.beta), ("tau", r.tau), ("q", r.q_proj.weight)):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad on {name}"
    enc = [p for p in m.stages()[0].encoder.parameters() if p.grad is not None]
    assert enc and any(p.grad.abs().sum() > 0 for p in enc), "router CE must train the encoder"


def test_entropy_is_detached_from_p():
    m = build()
    r = m.stages()[0].router
    h = torch.randn(2, 64, r.d_model, requires_grad=True)
    nxt = torch.randint(0, 256, (2, 64))
    p, _ = r.boundary_prob_next(h, torch.full((2,), 64), nxt)
    p.sum().backward()
    assert r.lm.weight.grad is None or r.lm.weight.grad.abs().sum() == 0, "p must not train the next-byte head"
    assert r.beta.grad.abs() > 0 and r.tau.grad.abs() > 0
    assert torch.all(p[:, 0] == 1.0)


def test_lower_entropy_means_fewer_cuts():
    """Same cosine term; a confident head (entropy ~0 < tau) gives lower p than a uniform head (entropy ln 256)."""
    r = build().stages()[0].router
    h = torch.randn(2, 64, r.d_model).abs()      # positive, so lm row 0 wins at every position below
    nxt = torch.randint(0, 256, (2, 64))
    L = torch.full((2,), 64)
    with torch.no_grad():
        r.tau.fill_(2.0)
        r.lm.weight.zero_()
        p_uniform, _ = r.boundary_prob_next(h, L, nxt)
        r.lm.weight[0].fill_(50.0)
        p_sure, _ = r.boundary_prob_next(h, L, nxt)
    assert (p_sure[:, 1:] < p_uniform[:, 1:]).all()


def test_misuse_fails_loud():
    cfg = make_cfg("lookahead", 2)
    cfg["stages"][1]["router"] = dict(name="surprise", target_ratio=2.0, loss_weight=0.03, kwargs={})
    with pytest.raises(ValueError, match="stage 0 only"):
        HNetLM(cfg)
    r = build().stages()[0].router
    with pytest.raises(ValueError, match="needs_next"):
        r(torch.randn(1, 32, r.d_model), torch.full((1,), 32))
    with pytest.raises(RuntimeError, match="needs the next bytes"):
        r.boundary_prob(torch.randn(1, 32, r.d_model), torch.full((1,), 32))


def test_running_stats_update_in_training_only():
    m = build()
    r = m.stages()[0].router
    x = torch.randint(0, 256, (2, 128))
    before = (float(r.h_mean), float(r.h_var))
    m.eval()
    with torch.no_grad():
        m(x)
    assert (float(r.h_mean), float(r.h_var)) == before, "eval must not move the running stats"
    m.train()
    with torch.no_grad():
        m(x)
    assert float(r.h_mean) != before[0]
    assert {"h_mean", "h_var"} <= {k.split(".")[-1] for k in m.state_dict()}, "resume must restore the running stats"
