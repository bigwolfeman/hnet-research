"""Frozen boundaries (hnet/boundaries.py, router "given"): the H-Net chunks exactly where the proxy says, causally,
and the proxy stays out of the trained model."""
import copy

import numpy as np
import pytest
import torch

from hnet.boundaries import calibrate
from hnet.model import HNetLM, total_loss
from hnet.routers import build_router
from tests.conftest import make_cfg


def test_calibrate_hits_rates():
    rng = np.random.default_rng(0)
    ents = [rng.gamma(2.0, 1.0, 4097) for _ in range(8)]
    th = calibrate(ents, [3.0, 3.0])
    h = np.concatenate([e[1:] for e in ents])
    assert th[0] < th[1]
    assert abs((h > th[0]).mean() - 1 / 3) < 2e-3 and abs((h > th[1]).mean() - 1 / 9) < 2e-3


def test_given_router_follows_mask_and_fails_loud():
    r = build_router("given", 16)
    h = torch.randn(2, 10, 16)
    lengths = torch.tensor([10, 7])
    g = torch.rand(2, 10) < 0.4
    g[:, 0] = True
    plan = r(h, lengths, g)
    valid = torch.arange(10)[None] < lengths[:, None]
    assert torch.equal(plan.mask, g & valid)
    assert torch.equal(plan.read_idx, torch.cumsum(plan.mask, 1) - 1)
    with pytest.raises(ValueError):
        r(h, lengths)
    with pytest.raises(ValueError):
        build_router("lookahead", 16)(h, lengths, g)


@pytest.fixture
def entropy_model(tmp_path):
    torch.manual_seed(0)
    proxy_cfg = dict(make_cfg(n_stages=1), stages=[], d_model=[32])   # a byte LM: no stages
    proxy = HNetLM(proxy_cfg)
    path = tmp_path / "proxy.pt"
    torch.save(dict(model=proxy.state_dict(), config=dict(model=proxy_cfg), step=0), path)
    cfg = make_cfg("lookahead", n_stages=2, multiple=32)
    for s in cfg["stages"]:
        s["router"] = dict(name="given", target_ratio=3.0, loss_weight=0.0, kwargs={})
    # thresholds from the proxy's own entropy on random bytes: about 1/3 and 1/9 boundaries per byte
    x = torch.randint(0, 256, (4, 256))
    from hnet.boundaries import EntropyBoundaries
    ents = list(EntropyBoundaries(proxy, [], {}).entropy(x, torch.full((4,), 256)).numpy())
    cfg["boundaries"] = dict(kind="entropy", proxy=str(path), thresholds=calibrate(ents, [3.0, 3.0]))
    return HNetLM(cfg).eval(), x


def test_model_chunks_at_given_boundaries(entropy_model):
    m, x = entropy_model
    lengths = torch.full((x.shape[0],), x.shape[1])
    masks = m.boundary_source(x, lengths)
    out = m(x, lengths)
    s0, s1 = out.aux
    assert s0.counts.tolist() == masks[0].sum(1).tolist()
    assert s1.counts.tolist() == masks[1].sum(1).tolist()
    for b in range(x.shape[0]):   # stage-1 anchors sit at the bytes of mask 1, a subset of stage 0's
        n = int(s1.counts[b])
        assert set(s1.pos[b, :n].tolist()) == set(torch.nonzero(masks[1][b]).flatten().tolist())
        assert set(s1.pos[b, :n].tolist()) <= set(s0.pos[b, : int(s0.counts[b])].tolist())


def test_proxy_is_not_trained_or_saved(entropy_model):
    m, _ = entropy_model
    proxy_ids = {id(p) for p in m.boundary_source.proxy.parameters()}
    assert not proxy_ids & {id(p) for p in m.parameters()}
    assert not any(k.startswith("boundary") for k in m.state_dict())
    assert all(not p.requires_grad for p in m.boundary_source.proxy.parameters())
    assert all(st.router.name == "given" and not list(st.router.parameters()) for st in m.stages())


@torch.no_grad()
def test_entropy_boundaries_are_causal(entropy_model):
    m, x = entropy_model
    y = x.clone()
    y[:, 150:] = torch.randint(0, 256, (x.shape[0], 106))
    a, b = m(x).logits[:, :150], m(y).logits[:, :150]
    assert (a - b).abs().max() < 1e-4, (a - b).abs().max()


def test_backward_reaches_the_hnet(entropy_model):
    m, x = entropy_model
    m.train()
    out = m(x)
    loss, _ = total_loss(out, x.roll(-1, 1), torch.full((x.shape[0],), x.shape[1]))
    loss.backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if "residual_proj" in n or "encoder" in n)
    assert all(p.grad is None for p in m.boundary_source.proxy.parameters())


def test_mismatched_config_fails_loud(entropy_model, tmp_path):
    m, _ = entropy_model
    cfg = copy.deepcopy(m.cfg)
    cfg["stages"][1]["router"]["name"] = "lookahead"
    with pytest.raises(ValueError):
        HNetLM(cfg)
    cfg = copy.deepcopy(m.cfg)
    del cfg["boundaries"]
    with pytest.raises(ValueError):
        HNetLM(cfg)
