"""Packed inner levels (hnet/packing.py): same math as the padded path at every valid position.

The round trip runs anywhere. The attention and model checks need a GPU (varlen flash attention) and, for the
model, mamba_ssm (tests/test_layouts.py mamba_cfg).
"""
import importlib.util

import pytest
import torch

from hnet.blocks import Stack
from hnet.model import HNetLM, total_loss
from hnet.packing import make_pack, pack_rows, unpack_rows

GPU = torch.cuda.is_available()
MAMBA = GPU and importlib.util.find_spec("mamba_ssm") is not None


def _counts(B=5, M=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(1, M + 1, (B,), generator=g)
    c[2] = M                                   # one full row
    return c


def test_pack_roundtrip_and_layout():
    B, M, d, mult = 5, 96, 8, 32
    c = _counts(B, M)
    p = make_pack(c, int(c.sum()), M, int(c.max()), mult)
    T = int(c.sum())
    assert p.idx.shape[0] % mult == 0 and p.idx.shape[0] >= T
    assert p.cu.tolist() == [0] + torch.cumsum(c, 0).tolist()[:-1] + [p.idx.shape[0]]
    assert p.max_len % mult == 0 and p.max_len >= int(c.max())
    seg = (p.cu[1:] - p.cu[:-1]).tolist()
    assert max(seg) <= p.max_len
    x = torch.randn(B, M, d)
    y = unpack_rows(pack_rows(x, p), p)
    valid = torch.arange(M)[None, :] < c[:, None]
    assert torch.equal(y[valid], x[valid])
    assert not y[~valid].any(), "pad slots must come back as 0"
    # row and position of every packed entry
    for b in range(B):
        s, e = int(p.cu[b]), int(p.cu[b]) + int(c[b])
        assert (p.seq_idx[0, s:e] == b).all() and torch.equal(p.pos[s:e], torch.arange(int(c[b])))


@pytest.mark.skipif(not GPU, reason="varlen flash attention needs a GPU")
@pytest.mark.parametrize("window", [None, 16, 64])
@torch.no_grad()
def test_packed_attention_stack_matches_padded(window):
    torch.manual_seed(0)
    B, M, d = 5, 96, 64
    st = Stack(d, "T2", 16, window, 2.0, 16, 10000.0, "sdpa_mask", rope_dim=8).cuda()
    c = _counts(B, M).cuda()
    x = torch.randn(B, M, d, device="cuda")
    p = make_pack(c, int(c.sum()), M, int(c.max()), 32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = st(x).float(), st(x, p).float()
    valid = torch.arange(M, device="cuda")[None, :] < c[:, None]
    err = (a - b)[valid].abs().max().item()
    assert err < 2e-2, err                     # bf16 kernels; a window or row-boundary error moves outputs by O(1)


def _clone_model(cfg, pack: bool, sd):
    m = HNetLM(dict(cfg, pack_inner=pack)).cuda()
    m.load_state_dict(sd)
    return m


@pytest.mark.skipif(not MAMBA, reason="needs mamba_ssm, causal_conv1d and a GPU")
@pytest.mark.parametrize("compile_", [False, True])
def test_packed_model_matches_padded(compile_):
    from tests.test_layouts import mamba_cfg

    torch.manual_seed(0)
    cfg = mamba_cfg()
    cfg["d_model"] = [64, 64, 128]   # Mamba-2 in_proj width 4d + 2 d_state + 2d/headdim must be a multiple of 8 (packing.py)
    base = HNetLM(cfg)
    with torch.no_grad():
        for st in base.stages():               # a random router so chunking is not all-or-nothing
            st.residual_proj.weight.normal_(0, 0.1)
    sd = base.state_dict()
    x = torch.randint(0, 256, (4, 512), device="cuda")
    y = torch.randint(0, 256, (4, 512), device="cuda")
    lengths = torch.tensor([512, 480, 256, 512], device="cuda")
    res = []
    for pack in (False, True):
        m = _clone_model(cfg, pack, sd)
        if compile_:
            m.compile_stacks()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(x, lengths)
        loss, _ = total_loss(out, y, lengths)
        loss.backward()
        res.append((float(loss), [a.counts.tolist() for a in out.aux],
                    {n: q.grad.float().norm().item() for n, q in m.named_parameters() if q.grad is not None}))
    (la, ca, ga), (lb, cb, gb) = res
    assert ca == cb, "routing changed"
    assert abs(la - lb) < 2e-3, (la, lb)
    assert set(ga) == set(gb)
    bad = {n: (ga[n], gb[n]) for n in ga if abs(ga[n] - gb[n]) > 1e-4 + 0.02 * abs(ga[n])}
    assert not bad, bad
