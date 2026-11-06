"""hnet/mainread.py: zero at init, causal over chunks, and seen by the coarse-path probe."""
import math
import os

import pytest
import torch

from hnet.mainread import MainRead
from hnet.model import HNetLM
from hnet.probes import ablated
from tests.conftest import make_cfg, randomize

MR = dict(head_dim=16)


def test_zero_at_init():
    torch.manual_seed(0)
    m = HNetLM(make_cfg("lookahead", 2, main_read=MR)).eval()
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 128))
    st = m.stages()[1]
    e = torch.randn(2, 20, st.d)
    y = torch.randn(2, 8, st.d)
    r = st.main_read(e, y, torch.randint(-1, 8, (2, 20)))
    assert r.abs().max().item() == 0.0
    assert all(s.main_read is not None for s in m.stages())
    with torch.no_grad():
        m(x)   # builds and runs


def test_reads_only_seen_chunks():
    torch.manual_seed(0)
    mr = MainRead(32, 16)
    mr.init_weights(0.2)
    torch.nn.init.normal_(mr.o.weight, std=0.2)
    B, L, M = 2, 24, 10
    e, y = torch.randn(B, L, 32), torch.randn(B, M, 32)
    read_idx = torch.sort(torch.randint(-1, M, (B, L)), dim=1).values
    base = mr(e, y, read_idx)
    for j in range(M):
        y2 = y.clone()
        y2[:, j] += 1.0
        diff = (mr(e, y2, read_idx) - base).abs().amax(-1)
        assert (diff[read_idx < j] < 1e-6).all(), f"chunk {j} reached a position that has not seen it"
        assert (diff[read_idx >= j] > 1e-6).all(), f"chunk {j} did not reach a position that has seen it"
    assert (base[read_idx < 0].abs() == 0).all()


@pytest.mark.parametrize("router", ["eoc", "eoc_lag", "lookahead"])
def test_model_with_main_read_is_causal(router):
    torch.manual_seed(0)
    m = randomize(HNetLM(make_cfg(router, 2, main_read=MR)).eval(), seed=0)
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 256, (3, 256), generator=g)
    with torch.no_grad():
        base = m(x).logits
        for t in (0, 31, 130, 254):
            x2 = x.clone()
            x2[:, t + 1:] = torch.randint(0, 256, x2[:, t + 1:].shape, generator=g)
            err = (m(x2).logits[:, : t + 1] - base[:, : t + 1]).abs().max().item()
            assert err < 1e-4, f"LEAK {router} t={t}: {err}"


def test_probe_ablates_the_read():
    """'zero' removes r entirely (same logits as a dead read under 'zero'); 'shuffle' feeds row b the r of row b-1."""
    torch.manual_seed(0)
    m = randomize(HNetLM(make_cfg("lookahead", 2, main_read=MR)).eval(), seed=0)
    st = m.stages()[1]
    x = torch.randint(0, 256, (3, 256))
    with torch.no_grad():
        base = m(x).logits
        with ablated(st, "zero"):
            z_live = m(x).logits
        with ablated(st, "shuffle"):
            s_live = m(x).logits
        w = st.main_read.o.weight.clone()
        st.main_read.o.weight.zero_()
        with ablated(st, "zero"):
            z_dead = m(x).logits
        with ablated(st, "shuffle"):
            s_dead = m(x).logits
        st.main_read.o.weight.copy_(w)
    assert torch.allclose(z_live, z_dead, atol=1e-6)
    assert (s_live - s_dead).abs().max() > 1e-3          # the rolled r still reaches the decoder
    assert (base - s_live).abs().max() > 1e-3


@pytest.mark.parametrize("stage", [0, 1])
def test_flops_count_the_read(stage):
    """Keys are main outputs: L_s / prod(ratios[s:]) of them, whatever the stage."""
    from hnet.flops import flops_per_byte, mean_ctx
    plain = make_cfg("lookahead", 2)
    read = make_cfg("lookahead", 2)
    read["stages"][stage]["main_read"] = MR
    L, r = 512, [4.0, 2.0]
    d = read["d_model"][stage]
    L_s = L / math.prod(r[:stage])           # stage-s positions per row
    to_main = math.prod(r[stage:])           # stage-s positions per main position
    want = (4 * d * d + 4 * d * mean_ctx(L_s / to_main, None) + 4 * d * d / to_main) / math.prod(r[:stage])
    got = flops_per_byte(read, L, r)["fwd"] - flops_per_byte(plain, L, r)["fwd"]
    assert abs(got - want) < 1e-6 * want


def _stage0_read_model():
    cfg = make_cfg("lookahead", 2)
    cfg["stages"][0]["main_read"] = MR
    torch.manual_seed(0)
    return randomize(HNetLM(cfg).eval(), seed=0)


def test_stage0_reads_main_outputs():
    """At stage 0 the keys are the main network's outputs (not stage-1 outputs), masked by the composite index:
    byte t sees main chunk j iff j <= stage1.read_idx[stage0.read_idx[t]], the set the EMA path sees."""
    m = _stage0_read_model()
    st0, st1 = m.stages()
    assert st1.main_read is None and st0.main_read is not None
    seen = {}
    main_fwd = st1.inner.forward
    def main_spy(h, lengths, pos, pack=None, given=None):
        y, aux = main_fwd(h, lengths, pos, pack, given); seen["main"] = y; return y, aux
    st1.inner.forward = main_spy
    read_fwd = st0._read_fn
    def read_spy(e, y, idx):
        seen["y"], seen["idx"] = y, idx; return read_fwd(e, y, idx)
    st0._read_fn = read_spy
    plan_fwd = {}
    for k, st in enumerate((st0, st1)):
        pre = st._pre_fn
        def pre_spy(e, lengths, given=None, nxt=None, *, _pre=pre, _k=k):
            out = _pre(e, lengths, given, nxt); plan_fwd[_k] = out[4]; return out
        st._pre_fn = pre_spy
    x = torch.randint(0, 256, (2, 256))
    with torch.no_grad():
        out = m(x)
    assert torch.equal(seen["y"], seen["main"][..., : st0.d])
    r0, r1 = plan_fwd[0], plan_fwd[1]
    want = torch.where(r0 < 0, -1, torch.gather(r1, 1, r0.clamp(min=0)))
    assert torch.equal(seen["idx"], want)
    assert torch.equal(out.aux[0].main_idx, want) and torch.equal(out.aux[1].main_idx, r1)
    assert (want > 0).any()


def test_compose_read_index():
    from hnet.stage import compose_read_index
    read_idx = torch.tensor([[-1, 0, 0, 1, 2, 2]])
    inner = torch.tensor([[-1, 0, 3]])
    assert compose_read_index(read_idx, inner).tolist() == [[-1, -1, -1, 0, 3, 3]]


def test_read_only_ablation():
    """part="read": 'zero' equals the same model with a dead read; the EMA read is untouched."""
    m = _stage0_read_model()
    st = m.stages()[0]
    x = torch.randint(0, 256, (3, 256))
    with torch.no_grad():
        base = m(x).logits
        with ablated(st, "zero", part="read"):
            z = m(x).logits
        with ablated(st, "shuffle", part="read"):
            sh = m(x).logits
        w = st.main_read.o.weight.clone()
        st.main_read.o.weight.zero_()
        dead = m(x).logits
        with ablated(st, "zero"):
            dead_all = m(x).logits
        st.main_read.o.weight.copy_(w)
    assert torch.allclose(z, dead, atol=1e-6)
    assert (base - z).abs().max() > 1e-3 and (base - sh).abs().max() > 1e-3
    assert (dead - dead_all).abs().max() > 1e-3       # part "all" also removes the EMA read; "read" does not
    with pytest.raises(ValueError):
        with ablated(m.stages()[1], "zero", part="read"):
            pass


def test_params_of_the_read():
    m = HNetLM(make_cfg("lookahead", 2, main_read=MR))
    d = m.stages()[1].d
    assert m.stages()[1].main_read.describe()["params"] == 4 * d * d + 2 * d


@pytest.mark.skipif(os.environ.get("HNET_TEST_CUDA") != "1" or not torch.cuda.is_available(), reason="HNET_TEST_CUDA=1")
def test_compiled_read_has_gradients_on_gpu():
    """The training path compiles MainRead (dynamic shapes, bf16). Forward parity with eager lives in
    tests/test_gpu.py (test_compile_matches_eager_*[main_read]); here: the compiled read gets finite, nonzero grads."""
    from tests.test_gpu import _parity_pair
    _, comp, x = _parity_pair("lookahead", rand=False, main_read=dict(head_dim=64))
    with torch.autocast("cuda", torch.bfloat16):
        comp(x).logits.float().square().mean().backward()
    for st in comp.stages():
        g = st.main_read.o.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0
