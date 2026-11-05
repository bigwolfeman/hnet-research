"""RefHNetLM adapter (reference H-Net behind our interface). Needs the reference environment and a GPU
(lab/spikes/2026-11-01-reference-hnet-probe/README.md); skipped elsewhere."""
import pytest
import torch

refmodel = pytest.importorskip("hnet.refmodel", reason="reference environment (refnet, mamba_ssm) not available",
                               exc_type=ImportError)
if not torch.cuda.is_available():
    pytest.skip("Mamba-2 kernels need CUDA", allow_module_level=True)

from hnet.evaluation import run_batches  # noqa: E402
from hnet.model import build_model, total_loss  # noqa: E402
from hnet.probes import ablated, coarse_use  # noqa: E402

CFG = dict(kind="ref_hnet", vocab_size=256, chunk_multiple=8, init_std=0.02, d_model=[64, 64, 128],
           stages=[{"router": {"target_ratio": 3.0, "loss_weight": 0.03}}] * 2,
           ref=dict(arch_layout=["m1", ["T1m1", ["T2"], "m1T1"], "m1"], d_intermediate=[0, 128, 256],
                    ssm_cfg=dict(chunk_size=256, d_conv=4, d_state=16, expand=2),
                    attn_cfg=dict(num_heads=[2, 2, 2], rotary_emb_dim=[16, 16, 16], window_size=[63, 63, -1]),
                    tie_embeddings=False))


def model(seed=0):
    torch.manual_seed(seed)
    m = build_model(CFG).cuda().eval()
    with torch.no_grad():                          # a random router, so boundaries are not all-or-nothing
        for st in m.stages():
            st.hnet.routing_module.q_proj_layer.weight.normal_(0, 0.3)
    return m


def fwd(m, x):
    """The reference attention requires bf16 activations: run as training and eval do (bf16 autocast)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(x)


def batch(B=4, L=512, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, L), generator=g).cuda()


@torch.no_grad()
def test_aux_is_consistent():
    m, x = model(), batch()
    out = fwd(m, x)
    assert out.logits.shape == (4, 512, 256) and len(out.aux) == 2
    a0, a1 = out.aux
    assert torch.equal(a1.lengths, a0.counts)
    for b in range(4):
        p0, p1 = a0.pos[b, : a0.counts[b]], a1.pos[b, : a1.counts[b]]
        assert p0[0] == 0 and bool((p0.diff() > 0).all()) and int(p0[-1]) < 512   # starts; byte 0 always starts
        assert set(p1.tolist()) <= set(p0.tolist())                                # stage-1 anchors are stage-0 anchors
    assert all(0 < float(a.frac) <= 1 for a in out.aux)


@torch.no_grad()
def test_ratio_loss_matches_reference_formula():
    from refnet.utils.train import load_balancing_loss
    m, x = model(), batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref_out = m.ref(x, mask=torch.ones_like(x, dtype=torch.bool))
    ours = fwd(m, x).aux[0].ratio_loss
    assert torch.allclose(ours, load_balancing_loss(ref_out.bpred_output[0], N=3.0).float(), atol=1e-5)


@torch.no_grad()
def test_causal():
    m, x = model(), batch(B=2, L=512)
    y = x.clone()
    y[:, 300:] = torch.randint(0, 256, (2, 212), device=x.device)
    a, b = fwd(m, x).logits[:, :300].float(), fwd(m, y).logits[:, :300].float()
    assert (a - b).abs().max() < 2e-2, (a - b).abs().max()   # bf16 kernels; a leak would move logits by O(1)


@torch.no_grad()
def test_ablation_handles():
    m, x = model(), batch()
    base = fwd(m, x).logits
    for st in m.stages():
        for mode in ("zero", "shuffle"):
            with ablated(st, mode):
                assert not torch.allclose(fwd(m, x).logits, base)
        assert st.ablate is None
    assert torch.equal(fwd(m, x).logits, base)
    batches = [(x, batch(seed=2))]
    r = coarse_use(m, batches, torch.device("cuda"), torch.bfloat16)
    assert set(r) == {"probe/coarse_base_bpb"} | {f"probe/coarse_{mo}_s{s}" for s in (0, 1) for mo in ("zero", "shuffle")}


def test_param_groups_cover_every_parameter_once():
    m = model()
    groups = m.stage_param_groups()
    assert [lvl for lvl, _ in groups] == [0, 1, 2]
    ids = [id(p) for _, ps in groups for _, p in ps]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    main_names = [n for n, _ in groups[2][1]]
    assert all(".main_network.main_network." in n for n in main_names)


def test_train_step_reduces_loss():
    m = model().train()
    x, y = batch(), batch(seed=3)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(20):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce = total_loss(m(x), y, torch.full((4,), 512, device="cuda"))
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
        losses.append(ce.item())
    assert losses[-1] < losses[0]
    run_batches(m.eval(), [(x, y)], torch.device("cuda"), torch.bfloat16)


@torch.no_grad()
def test_ref_state_maps_into_our_model():
    """The reference weights load into our HNetLM with the same configuration, and stage 0 (Mamba-2 only) agrees."""
    from hnet.refmodel import ref_state_to_ours
    ref = model()
    ours_cfg = dict(kind="hnet", vocab_size=256, chunk_multiple=8, d_model=[64, 64, 128], head_dim=32,
                    mlp_ratio=2.0, mlp_multiple_of=128, rope_theta=10000.0, init_std=0.02, attn_impl="sdpa_mask",
                    main=dict(layers=2), ssm=dict(CFG["ref"]["ssm_cfg"], headdim=64),
                    stages=[dict(enc_layers=1, dec_layers=1, enc_layout="m1", dec_layout="m1", window=64,
                                 router=dict(name="lookahead", target_ratio=3.0, loss_weight=0.03, kwargs={})),
                            dict(enc_layers=2, dec_layers=2, enc_layout="T1m1", dec_layout="m1T1", window=64,
                                 router=dict(name="lookahead", target_ratio=3.0, loss_weight=0.03, kwargs={}))])
    ours = build_model(ours_cfg).cuda().eval()
    ours.load_state_dict(ref_state_to_ours(ref, ours), strict=True)
    x = batch(B=2, L=256)
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e_ours = ours.stages()[0].encoder(ours.embed(x)).float()
        e_ref = ref.ref.backbone.encoder(ref.ref.embeddings(x), mask=mask).float()
    assert (e_ours - e_ref).abs().max() < 5e-2, (e_ours - e_ref).abs().max()
