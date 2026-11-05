"""Stack layouts (hnet/blocks.py parse_layout) and Mamba-2 outer stages (hnet/ssm.py).

The layout grammar and FLOPs tests run anywhere. The Mamba-2 tests need mamba_ssm, causal_conv1d and a GPU.
"""
import copy
import importlib.util

import pytest
import torch

from tests.conftest import make_cfg
from hnet.blocks import parse_layout
from hnet.flops import block_flops, flops_per_byte, layout_flops, mamba2_flops
from hnet.model import HNetLM, stage_layouts, total_loss

SSM = dict(d_state=16, d_conv=4, expand=2, headdim=16, chunk_size=64)


def test_parse_layout():
    assert parse_layout("T2") == [("attn", True)] * 2
    assert parse_layout("T1m4") == [("attn", True)] + [("mamba", False)] * 4
    assert parse_layout("M1t1") == [("mamba", True), ("attn", False)]
    assert parse_layout("T0") == [] and parse_layout("") == []
    for bad in ("x3", "T", "T1 m2", "2T"):
        with pytest.raises(ValueError):
            parse_layout(bad)


def test_default_layout_is_attention_blocks():
    st = make_cfg()["stages"][0]
    assert stage_layouts(st) == (f"T{st['enc_layers']}", f"T{st['dec_layers']}")
    assert stage_layouts(dict(st, enc_layout="M2", dec_layout="m3")) == ("M2", "m3")


def test_layout_flops_match_block_flops():
    d, h, L, w = 256, 704, 4096.0, 512
    assert layout_flops("T3", d, h, L, w, None) == pytest.approx(3 * block_flops(d, h, L, w))
    assert layout_flops("M2", d, h, L, w, SSM) == pytest.approx(2 * (mamba2_flops(d, SSM) + 6 * d * h))
    attn_only = block_flops(d, h, L, w) - 6 * d * h
    assert layout_flops("m1t1", d, h, L, w, SSM) == pytest.approx(mamba2_flops(d, SSM) + attn_only)


def test_explicit_T_layout_keeps_flops_and_params():
    cfg = make_cfg()
    cfg2 = copy.deepcopy(cfg)
    for st in cfg2["stages"]:
        st["enc_layout"], st["dec_layout"] = f"T{st['enc_layers']}", f"T{st['dec_layers']}"
    assert flops_per_byte(cfg, 512, [4.0, 2.0]) == flops_per_byte(cfg2, 512, [4.0, 2.0])
    a, b = HNetLM(cfg), HNetLM(cfg2)
    assert list(a.state_dict()) == list(b.state_dict())     # same parameter names: old checkpoints still load
    assert not any("_attn" in k for k in a.state_dict())


needs_mamba = pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None or not torch.cuda.is_available(),
    reason="Mamba-2 needs mamba_ssm, causal_conv1d and a GPU")


def mamba_cfg():
    cfg = make_cfg(router="lookahead", multiple=32)
    cfg["ssm"] = dict(SSM)
    cfg["stages"][0].update(enc_layout="M2", dec_layout="M2")
    cfg["stages"][1].update(enc_layout="T1m2", dec_layout="m2T1")
    return cfg


def fwd(m, x, lengths=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(x, lengths)


@needs_mamba
def test_mamba_layout_builds_and_counts():
    cfg = mamba_cfg()
    m = HNetLM(cfg)
    enc0 = m.stages()[0].encoder
    assert [b.mixer for b in enc0.blocks] == ["mamba", "mamba"] and all(b.mlp is not None for b in enc0.blocks)
    enc1 = m.stages()[1].encoder
    assert [(b.mixer, b.mlp is not None) for b in enc1.blocks] == [("attn", True), ("mamba", False), ("mamba", False)]
    f = flops_per_byte(cfg, 512, [4.0, 2.0])
    assert f["breakdown"]["stage0"] > 0 and f["fwd"] > 0


@needs_mamba
@torch.no_grad()
def test_mamba_model_is_causal_and_pad_safe():
    torch.manual_seed(0)
    m = HNetLM(mamba_cfg()).cuda().eval()
    for st in m.stages():                                  # a random router so chunking is not all-or-nothing
        st.residual_proj.weight.normal_(0, 0.1)
    x = torch.randint(0, 256, (2, 512), device="cuda")
    y = x.clone()
    y[:, 300:] = torch.randint(0, 256, (2, 212), device="cuda")
    a, b = fwd(m, x).logits[:, :300].float(), fwd(m, y).logits[:, :300].float()
    assert (a - b).abs().max() < 3e-2, (a - b).abs().max()    # bf16 kernels; a leak moves logits by O(1)
    lengths = torch.tensor([512, 320], device="cuda")         # right padding: positions >= length are pads
    full = fwd(m, x).logits[1, :320].float()
    padded = fwd(m, x, lengths).logits[1, :320].float()
    assert (full - padded).abs().max() < 3e-2, (full - padded).abs().max()


@needs_mamba
def test_mamba_train_step_is_finite_and_learns():
    torch.manual_seed(0)
    m = HNetLM(mamba_cfg()).cuda().train()
    x = torch.randint(0, 256, (4, 512), device="cuda")
    y = torch.roll(x, -1, dims=1)
    lengths = torch.full((4,), 512, device="cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(15):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce = total_loss(m(x, lengths), y, lengths)
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        mamba_grads = [p.grad for n, p in m.named_parameters() if ".mamba." in n]
        assert mamba_grads and all(g is not None for g in mamba_grads)
        opt.step()
        losses.append(ce.item())
    assert losses[-1] < losses[0]


def test_stack_height_counts_residual_adds():
    from hnet.blocks import Stack
    assert Stack(32, "T2", 16, None, 2.0, 16, 1e4, "sdpa_mask").height == 4          # T-only: 2 x layers, as before
    assert Stack(32, "t1T1", 16, None, 2.0, 16, 1e4, "sdpa_mask").height == 3


def test_init_scales_out_proj_by_height():
    """2025 _init_weights: std / sqrt(residual adds of this level and every enclosing level)."""
    cfg = make_cfg(stage_layers=[(2, 2), (1, 1)])
    cfg["stages"][0].update(enc_layout="t2", dec_layout="t2")       # 4 adds at stage 0 (old rule counted 8)
    torch.manual_seed(0)
    m = HNetLM(cfg)
    n0 = 4
    n1 = n0 + 2 * 2                                                    # stage 1: T1 + T1 -> 4 adds
    w0 = m.stages()[0].encoder.blocks[0].attn.out_proj.weight
    w1 = m.stages()[1].encoder.blocks[0].attn.out_proj.weight
    assert abs(w0.std().item() - 0.02 / n0 ** 0.5) < 0.1 * 0.02 / n0 ** 0.5
    assert abs(w1.std().item() - 0.02 / n1 ** 0.5) < 0.1 * 0.02 / n1 ** 0.5


def test_residual_proj_zero_weight_default_bias():
    m = HNetLM(make_cfg())
    for st in m.stages():
        assert torch.count_nonzero(st.residual_proj.weight) == 0
        assert st.residual_proj.bias.abs().max() > 0                   # nn.Linear default, as in the 2025 code


def test_partial_rope_rotates_only_the_first_dims():
    from hnet.blocks import apply_rope, rope_cos_sin
    x = torch.randn(2, 3, 50, 64)
    cos, sin = rope_cos_sin(50, 32, 1e4, x.device)
    y = apply_rope(x, cos, sin)
    assert torch.equal(y[..., 32:], x[..., 32:])
    assert not torch.allclose(y[..., :32], x[..., :32])
    cos_f, sin_f = rope_cos_sin(50, 64, 1e4, x.device)
    full = apply_rope(x, cos_f, sin_f)
    assert torch.allclose(full.norm(dim=-1), x.norm(dim=-1), atol=1e-4)   # a rotation keeps norms
    q = apply_rope(x[:, :, :1], cos[:1], sin[:1])
    assert torch.allclose(q, x[:, :, :1])                                  # position 0: identity
