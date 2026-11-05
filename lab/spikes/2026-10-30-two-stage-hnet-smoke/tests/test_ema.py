"""Gate 3: the parallel EMA scan equals the naive loop (values and gradients)."""
import pytest
import torch

from hnet.chunking import ema_naive, ema_scan


@pytest.mark.parametrize("M", [1, 2, 3, 16, 17, 33, 128, 200, 512])
def test_scan_matches_loop(M):
    torch.manual_seed(M)
    x = torch.randn(3, M, 24, dtype=torch.float32)
    p = torch.rand(3, M)
    p[0, : min(M, 5)] = 1.0          # clamps to 1 - 1e-4
    p[1, : min(M, 7)] = 0.0          # clamps to 1e-4
    assert (ema_scan(x, p) - ema_naive(x, p)).abs().max() < 1e-5


def test_gradients_match():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 8, dtype=torch.float32, requires_grad=True)
    p = torch.rand(2, 64).requires_grad_(True)
    w = torch.randn(2, 64, 8)
    g1 = torch.autograd.grad((ema_scan(x, p) * w).sum(), (x, p))
    g2 = torch.autograd.grad((ema_naive(x, p) * w).sum(), (x, p))
    for a, b in zip(g1, g2):
        assert (a - b).abs().max() < 1e-4


def test_matches_2025_recurrence_first_element():
    x = torch.ones(1, 3, 2)
    p = torch.tensor([[1.0, 0.5, 0.5]])
    out = ema_scan(x, p)
    # out_0 ~ p_0 (clamped) ; out_1 = .5 + .5*out_0 ; out_2 = .5 + .5*out_1
    assert torch.allclose(out[0, 1, 0], torch.tensor(0.5 + 0.5 * out[0, 0, 0]), atol=1e-6)
