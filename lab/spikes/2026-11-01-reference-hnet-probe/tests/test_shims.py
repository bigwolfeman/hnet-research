"""The flash_attn shim against direct float32 math (gate of the reference-checkpoint experiment)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import refhnet  # noqa: E402,F401  (puts shims/ and the reference repo on sys.path)

from flash_attn import (flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func,  # noqa: E402
                        flash_attn_kvpacked_func)
from flash_attn.ops.activations import swiglu  # noqa: E402
from flash_attn.ops.triton.layer_norm import RMSNorm  # noqa: E402
from flash_attn.ops.triton.rotary import apply_rotary  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def naive_attn(q, k, v, left):
    """q,k,v (B,S,H,D) float32; bottom-right causal with a left window."""
    Sq, Sk, D = q.shape[1], k.shape[1], q.shape[-1]
    s = torch.einsum("bihd,bjhd->bhij", q, k) / D ** 0.5
    i = torch.arange(Sq)[:, None] + (Sk - Sq)
    j = torch.arange(Sk)[None, :]
    ok = j <= i
    if left >= 0:
        ok &= j >= i - left
    s = s.masked_fill(~ok.to(s.device), float("-inf"))
    return torch.einsum("bhij,bjhd->bihd", s.softmax(-1), v)


@pytest.mark.parametrize("left", [-1, 7, 63, 200])
def test_qkvpacked_matches_naive(left):
    torch.manual_seed(0)
    qkv = torch.randn(2, 100, 3, 4, 32, device=DEV)
    ref = naive_attn(*qkv.unbind(2), left)
    out = flash_attn_qkvpacked_func(qkv.bfloat16(), causal=True, window_size=(left, -1)).float()
    assert (out - ref).abs().max() < 3e-2


def test_kvpacked_bottom_right_alignment():
    torch.manual_seed(1)
    q, kv = torch.randn(1, 5, 2, 16, device=DEV), torch.randn(1, 12, 2, 2, 16, device=DEV)
    ref = naive_attn(q, *kv.unbind(2), -1)
    out = flash_attn_kvpacked_func(q, kv, causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_varlen_matches_per_sequence():
    torch.manual_seed(2)
    lens = [30, 7, 50]
    cu = torch.tensor([0, 30, 37, 87], dtype=torch.int32, device=DEV)
    qkv = torch.randn(sum(lens), 3, 4, 16, device=DEV)
    out = flash_attn_varlen_qkvpacked_func(qkv, cu, max(lens), causal=True, window_size=(9, -1))
    for b in range(3):
        a, e = int(cu[b]), int(cu[b + 1])
        ref = naive_attn(*qkv[None, a:e].unbind(2), 9)[0]
        assert torch.allclose(out[a:e], ref, atol=1e-5)


def test_rmsnorm_fused_add():
    torch.manual_seed(3)
    n = RMSNorm(64, eps=1e-5, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        n.weight.copy_(torch.rand(64) + 0.5)
    x, r = torch.randn(3, 64, device=DEV, dtype=torch.bfloat16), torch.randn(3, 64, device=DEV)
    y, res = n(x, residual=r, prenorm=True, residual_in_fp32=True)
    s = x.float() + r
    ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5) * n.weight.float()
    assert res.dtype == torch.float32 and torch.equal(res, s)
    assert y.dtype == torch.bfloat16 and (y.float() - ref).abs().max() < 2e-2
    assert torch.equal(n(x, residual=r, prenorm=False, residual_in_fp32=True), y)


@pytest.mark.parametrize("interleaved", [False, True])
def test_rotary_matches_reference_torch_path(interleaved):
    from hnet.modules.rotary import apply_rotary_emb_torch
    torch.manual_seed(4)
    x = torch.randn(2, 20, 3, 48, device=DEV)
    theta = torch.randn(64, 16, device=DEV)
    cos, sin = theta.cos(), theta.sin()
    ref = apply_rotary_emb_torch(x, cos[:20], sin[:20], interleaved=interleaved)
    assert torch.allclose(apply_rotary(x, cos, sin, interleaved=interleaved), ref, atol=1e-5)
    back = apply_rotary(ref, cos, sin, interleaved=interleaved, conjugate=True)
    assert torch.allclose(back, x, atol=1e-5)
    ref_off = apply_rotary_emb_torch(x, cos[5:25], sin[5:25], interleaved=interleaved)
    assert torch.allclose(apply_rotary(x, cos, sin, seqlen_offsets=5, interleaved=interleaved), ref_off, atol=1e-5)


def test_swiglu():
    x, y = torch.randn(10), torch.randn(10)
    assert torch.allclose(swiglu(x, y), torch.nn.functional.silu(x) * y)
