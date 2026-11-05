"""GPU-only checks. Run with HNET_TEST_CUDA=1 (never on the local 5090; use the DGX Spark)."""
import os

import pytest
import torch

from hnet.blocks import Attention
from hnet.model import HNetLM, total_loss
from tests.conftest import make_cfg, randomize

pytestmark = pytest.mark.skipif(os.environ.get("HNET_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                                reason="set HNET_TEST_CUDA=1 on a GPU box")


@pytest.mark.parametrize("L,window", [(512, 128), (1024, 256), (384, 64)])
def test_flex_matches_dense_mask(L, window):
    torch.manual_seed(0)
    a = Attention(128, 64, window, "flex").cuda().to(torch.bfloat16)
    b = Attention(128, 64, window, "sdpa_mask").cuda().to(torch.bfloat16)
    b.load_state_dict(a.state_dict())
    from hnet.blocks import rope_cos_sin
    cos, sin = rope_cos_sin(L, 64, 10000.0, "cuda")
    x = torch.randn(2, L, 128, device="cuda", dtype=torch.bfloat16)
    err = (a(x, cos, sin, a.attn_mask(L, x.device)).float() - b(x, cos, sin, b.attn_mask(L, x.device)).float()).abs().max().item()
    assert err < 3e-2, err


@pytest.mark.parametrize("router", ["eoc", "eoc_lag", "lookahead"])
def test_model_flex_vs_mask(router):
    torch.manual_seed(0)
    cfg_f, cfg_m = make_cfg(router, 2, multiple=128, attn_impl="flex"), make_cfg(router, 2, multiple=128)
    for c in (cfg_f, cfg_m):
        c["d_model"] = [128, 128, 128]; c["head_dim"] = 64
    mf, mm = HNetLM(cfg_f).cuda(), HNetLM(cfg_m).cuda()
    randomize(mf); mm.load_state_dict(mf.state_dict())
    x = torch.randint(0, 256, (2, 512), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        of, om = mf(x), mm(x)
    # same routing decisions, close logits
    assert all(torch.equal(a.counts, b.counts) for a, b in zip(of.aux, om.aux))
    diff = (of.logits.float() - om.logits.float()).abs()  # bf16 noise on logits of scale ~2
    assert diff.mean().item() < 0.02 and diff.max().item() < 0.6
    with torch.autocast("cuda", torch.bfloat16):  # grad-enabled pass: flex backward
        loss, _ = total_loss(mf(x), x, torch.full((2,), 512, device="cuda"))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in mf.parameters() if p.grad is not None)


def _parity_pair(router, rand, seed=0, main_read=None, loops=1, main=None):
    """main_read: optional stage config (hnet/mainread.py). Its output projection starts at zero, so with default
    init it is set to a small random draw here, or the read would be idle and the parity would test nothing."""
    torch.manual_seed(seed)
    cfg = make_cfg(router, 2, multiple=128, main_read=main_read)
    cfg["d_model"] = [128, 128, 128]; cfg["head_dim"] = 64; cfg["main"]["loops"] = loops
    if main is not None:
        cfg["main"] = main
    eager, comp = HNetLM(cfg).cuda(), HNetLM(cfg).cuda()
    if rand:
        randomize(eager)
    elif main_read is not None:
        for st in eager.stages():
            torch.nn.init.normal_(st.main_read.o.weight, std=0.02)
    comp.load_state_dict(eager.state_dict())
    comp.compile_stacks()
    if main is not None and main.get("kind") == "parcae":   # train mode draws a random loop state and depth per call
        eager.eval(); comp.eval()
    return eager, comp, torch.randint(0, 256, (2, 512), device="cuda")


def _run(m, x, amp):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
        o = m(x)
    return o, total_loss(o, x, torch.full((x.shape[0],), x.shape[1], device="cuda"))


def _assert_parity(router, rand, amp, mean_tol, max_tol, loss_tol, main_read=None, loops=1, main=None):
    eager, comp, x = _parity_pair(router, rand, main_read=main_read, loops=loops, main=main)
    (oe, (le, ce_e)), (oc, (lc, ce_c)) = _run(eager, x, amp), _run(comp, x, amp)
    assert all(torch.equal(a.counts, b.counts) for a, b in zip(oe.aux, oc.aux)), "routing differs under compile"
    diff = (oe.logits.float() - oc.logits.float()).abs()
    assert diff.mean().item() < mean_tol and diff.max().item() < max_tol, (diff.mean().item(), diff.max().item())
    assert abs(le.item() - lc.item()) < loss_tol and abs(ce_e.item() - ce_c.item()) < loss_tol, (le.item(), lc.item())


MAIN_READ = [None, dict(head_dim=64)]


@pytest.mark.parametrize("main_read", MAIN_READ, ids=["plain", "main_read"])
@pytest.mark.parametrize("router", ["eoc", "lookahead"])
def test_compile_matches_eager_bf16(router, main_read):
    """Default init, bf16 autocast, one batch: compiled stacks vs eager. Same routing, logits and loss within bf16 noise."""
    _assert_parity(router, rand=False, amp=True, mean_tol=2e-3, max_tol=3e-2, loss_tol=2e-3, main_read=main_read)


@pytest.mark.parametrize("main_read", MAIN_READ, ids=["plain", "main_read"])
@pytest.mark.parametrize("router", ["eoc", "lookahead"])
def test_compile_matches_eager_fp32_random_weights(router, main_read):
    """Random weights put the routers near p=0.5, where bf16 noise flips hard boundaries (eager fp32 vs eager bf16 differs
    by 0.66 mean logit). fp32 removes that noise, so this isolates compile bugs: routing must match, logits to 1e-3."""
    _assert_parity(router, rand=True, amp=False, mean_tol=1e-3, max_tol=1e-2, loss_tol=1e-3, main_read=main_read)


@pytest.mark.parametrize("rand,amp", [(False, True), (True, False)], ids=["bf16_default", "fp32_random"])
def test_compile_matches_eager_looped_main(rand, amp):
    """Main.loops = 3: the compiled main blocks run three times per step (dynamic shapes); parity as the tests above."""
    tol = dict(mean_tol=2e-3, max_tol=3e-2, loss_tol=2e-3) if amp else dict(mean_tol=1e-3, max_tol=1e-2, loss_tol=1e-3)
    _assert_parity("lookahead", rand=rand, amp=amp, loops=3, **tol)


PARCAE = dict(kind="parcae", layers=None, loops=1, parcae=dict(prelude=1, core=1, coda=1, mean_recurrence=3,
                                                                backprop_depth=2, decay_init=0.4472135954999579, seed=0))


@pytest.mark.parametrize("rand,amp", [(False, True), (True, False)], ids=["bf16_default", "fp32_random"])
def test_compile_matches_eager_parcae_main(rand, amp):
    """Parcae main (hnet/parcae.py) at eval: prelude, 3 injected core steps, coda, compiled vs eager."""
    tol = dict(mean_tol=2e-3, max_tol=3e-2, loss_tol=2e-3) if amp else dict(mean_tol=1e-3, max_tol=1e-2, loss_tol=1e-3)
    _assert_parity("lookahead", rand=rand, amp=amp, main=PARCAE, **tol)


def test_parcae_no_grad_prefix_keeps_core_grads_under_autocast():
    """With a no-grad prefix of 3 steps and 1 step with grad under CUDA autocast, every core weight gets a gradient.
    Parcae turns the autocast weight cache off for the prefix (_no_grad_prefix), because a cast cached there has no
    grad_fn. On torch 2.14.1 this test also passes without that guard, so it checks the property, not the guard."""
    from hnet.parcae import ParcaeMain
    torch.manual_seed(0)
    cfg = make_cfg("lookahead", 1); cfg["main"] = PARCAE
    m = HNetLM(cfg).cuda().train().net.inner
    assert isinstance(m, ParcaeMain)
    m.sample_depths = lambda rows: (torch.full((rows,), 3), torch.ones(rows, dtype=torch.long))
    with torch.autocast("cuda", torch.bfloat16):
        out, _ = m(torch.randn(2, 16, m.d, device="cuda"), None, None)
    out.float().square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.core.parameters())
