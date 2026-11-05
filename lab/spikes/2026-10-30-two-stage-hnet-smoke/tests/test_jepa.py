"""Coarse JEPA (hnet/jepa.py) and its PairLoader (hnet/data.py): SIGReg, padding invariance, gradient reach, resumable stream."""
import numpy as np
import torch

from tests.conftest import make_cfg, randomize
from hnet.data import SEP, PairLoader
from hnet.jepa import CoarseJEPA, main_net, sigreg
from hnet.model import HNetLM


def _model():
    torch.manual_seed(0)
    return randomize(HNetLM(make_cfg("lookahead", 2, multiple=32)))


def _jepa(m):
    return CoarseJEPA(m, 64, 16, lam=0.02, slices=64, knots=17, seed=0)


def test_sigreg_separates_gaussian_from_collapse():
    g = torch.Generator().manual_seed(0)
    n, k = 256, 32
    gauss = torch.randn(n, k, generator=g)
    stats = {name: sigreg(z, torch.Generator().manual_seed(1), 128, 17).item() for name, z in
             [("gaussian", gauss), ("collapsed", torch.ones(n, k) + 0.01 * gauss), ("scaled", 4 * gauss)]}
    assert stats["gaussian"] < 0.05 * stats["collapsed"], stats
    assert stats["gaussian"] < 0.05 * stats["scaled"], stats


def test_embed_ignores_padding():
    m = _model().eval()
    j = _jepa(m)
    g = torch.Generator().manual_seed(1)
    n = 150
    seq = torch.randint(1, 256, (n,), generator=g)
    with torch.no_grad():
        ref = j.embed(m, torch.cat([seq, torch.zeros(10, dtype=torch.long)])[None, :160], torch.tensor([n]))
        for width, seed in [(192, 2), (256, 3)]:
            pad = torch.randint(0, 256, (width - n,), generator=torch.Generator().manual_seed(seed))
            z = j.embed(m, torch.cat([seq, pad])[None], torch.tensor([n]))
            assert (z - ref).abs().max().item() < 1e-4, width


def test_loss_reaches_main_and_projector_only():
    m = _model()
    j = _jepa(m)
    g = torch.Generator().manual_seed(4)
    xa, xb = torch.randint(1, 256, (4, 128), generator=g), torch.randint(1, 256, (4, 64), generator=g)
    loss, logs = j.loss(m, xa, torch.tensor([128, 100, 90, 70]), xb, torch.tensor([64, 40, 33, 50]), step=3)
    assert torch.isfinite(loss) and loss.item() > 0
    assert set(logs) == {"jepa/loss", "jepa/inv", "jepa/sigreg", "jepa/z_cos_same_pair", "jepa/z_cos_other_pair",
                         "jepa/top1_doc_given_code"}
    again, _ = j.loss(m, xa, torch.tensor([128, 100, 90, 70]), xb, torch.tensor([64, 40, 33, 50]), step=3)
    assert torch.equal(loss.detach(), again.detach()), "the SIGReg directions must depend on the step only"
    loss.backward()
    main = main_net(m)
    for name, mod in [("main", main), ("projector", j.proj)]:
        gsum = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        assert gsum > 0, f"no JEPA gradient reaches the {name}"
    main_ids = {id(p) for p in main.parameters()}
    outer = [n for n, p in m.named_parameters() if id(p) not in main_ids and p.grad is not None and p.grad.abs().sum() > 0]
    assert not outer, f"the JEPA gradient must stop at Main's input (encoders feed the routers): {outer[:4]}"


def test_hook_is_idle_outside_loss():
    m = _model()
    j = _jepa(m)
    with torch.no_grad():
        m(torch.randint(1, 256, (1, 64)))
    assert j._capture is None


def test_pair_loader_rows_and_resume(tmp_path):
    rng = np.random.default_rng(0)
    recs = {"code": [], "doc": []}
    for name in recs:
        for _ in range(7):
            recs[name].append(rng.integers(1, 256, rng.integers(5, 90)).astype(np.uint8))
        np.concatenate([np.append(r, SEP) for r in recs[name]]).astype(np.uint8).tofile(tmp_path / f"{name}.bin")
    pl = PairLoader(str(tmp_path), ("code", "doc"), (64, 32), 4, seed=5, device="cpu")
    assert pl.n == 7
    state = pl.state_dict()
    xa, la, xb, lb = pl.next()
    idx = np.random.default_rng(5).integers(0, 7, 4)
    for r, i in enumerate(idx):
        for x, lens, name, w in [(xa, la, "code", 64), (xb, lb, "doc", 32)]:
            rec = recs[name][i][:w]
            assert lens[r] == len(rec) and (x[r, : len(rec)].numpy() == rec).all() and (x[r, len(rec):] == 0).all()
    again = PairLoader(str(tmp_path), ("code", "doc"), (64, 32), 4, seed=99, device="cpu", state=state).next()
    assert all(torch.equal(a, b) for a, b in zip(again, (xa, la, xb, lb)))


def test_separate_backward_equals_summed_loss():
    """train.py backprops the JEPA term before the LM forward (peak memory); the gradient must equal that of the sum."""
    g = torch.Generator().manual_seed(7)
    x = torch.randint(1, 256, (2, 128), generator=g)
    xa, xb = torch.randint(1, 256, (4, 128), generator=g), torch.randint(1, 256, (4, 64), generator=g)
    la, lb = torch.tensor([128, 100, 90, 70]), torch.tensor([64, 40, 33, 50])

    def lm_loss(m):
        return m(x).logits[:, :-1].float().log_softmax(-1).gather(-1, x[:, 1:, None]).mean().neg()

    grads = []
    for separate in (False, True):
        m = _model()
        j = _jepa(m)
        torch.manual_seed(0)
        j.proj.apply(lambda mod: mod.reset_parameters() if hasattr(mod, "reset_parameters") else None)
        if separate:
            (0.7 * j.loss(m, xa, la, xb, lb, step=5)[0]).backward()
            lm_loss(m).backward()
        else:
            (0.7 * j.loss(m, xa, la, xb, lb, step=5)[0] + lm_loss(m)).backward()
        grads.append(torch.cat([p.grad.flatten() for p in list(m.parameters()) + list(j.parameters()) if p.grad is not None]))
    assert grads[0].shape == grads[1].shape
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-5, atol=1e-7)
