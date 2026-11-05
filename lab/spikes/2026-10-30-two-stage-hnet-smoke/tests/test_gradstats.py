"""hnet/gradstats.py: the AdamW update rms reads ~1 for a consistent gradient and the noise floor for pure noise."""
import torch

from hnet import gradstats


def _opt(params):
    return torch.optim.AdamW([dict(params=params, name="stage0_decay"), dict(params=[], name="stage1_nodecay")],
                             lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)


def _run(grad_fn, steps=2000):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(4096))
    opt = _opt([p])
    lv = gradstats.levels(opt)
    for _ in range(steps):
        p.grad = grad_fn()
        opt.step()
    return gradstats.update_rms(opt, lv)[0].item(), lv


def test_consistent_gradient_reads_one():
    r, _ = _run(lambda: torch.full((4096,), 0.3))
    assert abs(r - 1.0) < 1e-3


def test_noise_reads_the_floor():
    r, _ = _run(lambda: torch.randn(4096))
    assert abs(r - gradstats.NOISE_FLOOR_B1_09) < 0.02, r


def test_levels_and_grad_norms():
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([dict(params=[a], name="stage0_decay"), dict(params=[b], name="stage2_extra")])
    lv = gradstats.levels(opt)
    assert sorted(lv) == [0, 2]
    a.grad, b.grad = torch.full((3,), 2.0), torch.full((4,), 1.0)
    gn = gradstats.grad_norms(lv)
    assert torch.isclose(gn[0], torch.tensor(12.0).sqrt()) and torch.isclose(gn[2], torch.tensor(2.0))


def test_stats_do_not_change_training():
    def train(stats: bool):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4))
        opt = torch.optim.AdamW([dict(params=list(net[0].parameters()), name="stage0_decay"),
                                 dict(params=list(net[2].parameters()), name="stage2_decay")], lr=1e-2)
        lv = gradstats.levels(opt)
        g = torch.Generator().manual_seed(1)
        for _ in range(20):
            x = torch.randn(32, 8, generator=g)
            net(x).square().mean().backward()
            if stats:
                gradstats.grad_norms(lv)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if stats:
                gradstats.update_rms(opt, lv)
            opt.zero_grad(set_to_none=True)
        return [p.detach().clone() for p in net.parameters()]
    for a, b in zip(train(False), train(True)):
        assert torch.equal(a, b)
