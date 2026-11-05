"""ParcaeMain (hnet/parcae.py): prelude -> core loop with diagonal injection -> coda, Poisson depth per row in training."""
import math

import pytest
import torch
import torch.nn.functional as F

from hnet.flops import flops_per_byte, main_block_passes
from hnet.model import HNetLM, build_model
from hnet.parcae import ParcaeMain, truncated_full_depths
from tests.conftest import make_cfg, randomize

DECAY = math.sqrt(1 / 5)


def _cfg(prelude=1, core=1, coda=1, R=3, s=2):
    c = make_cfg("lookahead", 1)
    c["main"] = dict(kind="parcae", layers=None, loops=1,
                     parcae=dict(prelude=prelude, core=core, coda=coda, mean_recurrence=R, backprop_depth=s,
                                 decay_init=DECAY, seed=0))
    return c


def _main(model) -> ParcaeMain:
    m = model.net.inner
    assert isinstance(m, ParcaeMain)
    return m


def test_init_matches_parcae():
    m = _main(HNetLM(_cfg()))
    assert torch.allclose(m.decay(), torch.full((m.d,), DECAY), atol=1e-6)    # exp(-dt A), A = 1
    assert torch.equal(m.B, torch.eye(m.d))                                     # B identity
    assert torch.allclose(F.softplus(m.dt_bias), torch.full((m.d,), -math.log(DECAY)), atol=1e-6)
    assert isinstance(m.core.final_norm, torch.nn.Identity)                     # no norm between iterations
    assert not isinstance(m.prelude.final_norm, torch.nn.Identity)              # prelude_norm
    assert not isinstance(m.coda.final_norm, torch.nn.Identity)


def test_eval_runs_R_injected_core_steps():
    torch.manual_seed(0)
    m = _main(randomize(HNetLM(_cfg(R=3)), seed=1).eval())
    h = torch.randn(2, 16, m.d)
    with torch.no_grad():
        e = m.prelude(h)
        u = F.softplus(m.dt_bias) * (e @ m.B.t())
        x = m.init_state(e)
        for _ in range(3):
            x = m.core(x * m.decay() + u)
        ref = m.coda(m.C(x))
        out, aux = m(h, None, None)
        again, _ = m(h, None, None)
    assert torch.allclose(out, ref, atol=1e-6) and aux == []
    assert torch.equal(out, again)                                              # eval state is deterministic


def test_train_depth_per_row_and_truncated_grad():
    torch.manual_seed(0)
    model = randomize(HNetLM(_cfg(R=3, s=2)), seed=2).train()
    m = _main(model)
    n = torch.tensor([0, 2, 1]); k = torch.tensor([1, 2, 2])
    m.sample_depths = lambda rows: (n, k)
    h = torch.randn(3, 16, m.d)
    x0 = torch.randn(3, 16, m.d, generator=torch.Generator().manual_seed(123))
    m.init_state = lambda e_: x0.clone()
    with torch.no_grad():
        u = F.softplus(m.dt_bias) * (m.prelude(h) @ m.B.t())
    m.train()
    out, _ = m(h, None, None)
    with torch.no_grad():
        for i in range(3):
            x = x0[i:i + 1]
            for _ in range(int(n[i] + k[i])):
                x = m.core(x * m.decay() + u[i:i + 1])
            assert torch.allclose(out[i:i + 1], m.coda(m.C(x)), atol=1e-5), i
    out.square().mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name             # prelude, core, coda, A, dt, B, C


def test_sampler_is_poisson_truncated_full_and_resumable():
    m = _main(HNetLM(_cfg(R=6, s=3))).train()
    n, k = m.sample_depths(4000)
    total = n + k
    assert total.min() >= 1 and k.max() <= 3 and torch.equal(k, total.clamp(max=3))
    assert abs(total.float().mean().item() - 6) < 0.2
    state = {kk: v.clone() for kk, v in m.state_dict().items()}
    a = m.sample_depths(8)
    m.load_state_dict(state)
    assert torch.equal(m.sample_depths(8)[0], a[0])                             # the draw counter is in the state
    assert torch.equal(truncated_full_depths(torch.tensor([1, 5]), 3)[1], torch.tensor([1, 3]))


def test_flops_count_mean_depth_and_params_count_unique_layers():
    c = _cfg(prelude=2, core=2, coda=2, R=2, s=1)
    plain = make_cfg("lookahead", 1); plain["main"] = dict(layers=8)
    assert main_block_passes(c["main"]) == 8 == main_block_passes(plain["main"])
    d = c["d_model"][-1]
    fp = flops_per_byte(c, 256, [4.0])["breakdown"]["main"]
    f8 = flops_per_byte(plain, 256, [4.0])["breakdown"]["main"]
    assert fp == pytest.approx(f8 + 4 * d * d / 4.0)                            # + injection and C per main position
    p6 = sum(p.numel() for p in HNetLM(c).net.inner.parameters())
    plain["main"] = dict(layers=6)
    assert p6 == sum(p.numel() for p in HNetLM(plain).net.inner.parameters()) + 3 * d + 2 * d * d


def test_injection_params_skip_weight_decay():
    m = _main(HNetLM(_cfg()))
    assert all(getattr(p, "_no_weight_decay", False) for p in (m.A_log, m.dt_bias, m.B))
    assert not getattr(m.C.weight, "_no_weight_decay", False)


@pytest.mark.parametrize("bad", [dict(layers=4), dict(loops=2)])
def test_parcae_rejects_plain_depth_keys(bad):
    c = _cfg(); c["main"].update(bad)
    with pytest.raises(ValueError, match="parcae"):
        HNetLM(c)


def test_ref_hnet_rejects_parcae():
    c = _cfg(); c["kind"] = "ref_hnet"
    with pytest.raises(ValueError, match="no looped main"):
        build_model(c)


def test_full_model_trains_one_step():
    torch.manual_seed(0)
    model = HNetLM(_cfg()).train()
    x = torch.randint(0, 256, (2, 128))
    out = model(x)
    loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1))
    loss.backward()
    assert math.isfinite(loss.item())
    assert _main(model).last_depth is not None
