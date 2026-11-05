"""Sliding-window attention (hnet/blocks.py Attention): a query at position q sees keys q - window + 1 .. q, no more.

Main network takes recall (lab/experiments) reads a small stage-1 window as "stage 1 cannot attend across a
document"; this test is what makes that reading safe, for both mask implementations.
"""
import os

import pytest
import torch

from hnet.blocks import Attention, rope_cos_sin

CUDA = os.environ.get("HNET_TEST_CUDA") == "1" and torch.cuda.is_available()
IMPLS = [("sdpa_mask", "cpu"), pytest.param("flex", "cuda", marks=pytest.mark.skipif(not CUDA, reason="HNET_TEST_CUDA=1"))]


@pytest.mark.parametrize("impl,device", IMPLS)
@pytest.mark.parametrize("window", [1, 8, 13])
def test_window_reach(impl, device, window):
    torch.manual_seed(0)
    L, d, hd, p = 128, 64, 16, 40
    att = Attention(d, hd, window, impl).to(device)
    cos, sin = rope_cos_sin(L, hd, 10_000.0, device)
    x = torch.randn(1, L, d, device=device)
    x2 = x.clone()
    x2[0, p] += 1.0
    mask = att.attn_mask(L, torch.device(device))
    with torch.no_grad():
        diff = (att(x2, cos, sin, mask) - att(x, cos, sin, mask)).abs().amax(-1)[0]
    changed = (diff > 1e-5).nonzero().flatten().tolist()
    assert changed == list(range(p, p + window)), f"window {window}: positions changed by input {p}: {changed}"
