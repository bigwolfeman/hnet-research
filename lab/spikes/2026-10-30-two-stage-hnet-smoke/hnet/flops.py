"""FLOPs per byte (Principle 2). One function, used by every run.

Counting rules (forward pass; 1 multiply-add = 2 FLOPs):
    per position, per block:  qkv + out proj = 8 d^2,  SwiGLU = 6 d h,
                              causal attention scores+values = 4 * d * mean_ctx
                              Mamba-2 block: mamba2_flops(d, ssm) (below) instead of the attention terms
    Stage layouts (enc_layout / dec_layout, hnet/blocks.py parse_layout): T = attention + MLP, t = attention,
    M = Mamba-2 + MLP, m = Mamba-2. A stage without layouts is T<enc_layers> / T<dec_layers>.
    mean_ctx for a causal window w over a length-L sequence:
        L <= w : (L + 1) / 2
        L >  w : (w (w + 1) / 2 + (L - w) w) / L
    Only causal attention work is counted (no masked-out half).
    Router: lookahead 2 d^2 * 2 (q and k projections), surprise lookahead + 2 d V (next-byte head), eoc 2 d. Residual proj 2 d^2. LM head 2 d_0 V.
    main_read (hnet/mainread.py) at stage s, per stage-s position: q and out proj 4 d^2 + attention 4 d * mean_ctx
    over the main positions (L_s / prod(ratios[s:]), causal, global); per main position: k and v proj 4 d^2.
    Main with loops R (weight-tied): R x its stack's FLOPs.
    Parcae main: prelude + R x core + coda block passes (R = mean recurrence), plus the injection and C (2 d^2 each).
    EMA scan, norms, RoPE, embedding lookup are not counted.
A level-s position costs 1 / prod(ratios[:s]) per byte, and its sequence length is
seq_len / prod(ratios[:s]). `ratios[s]` is the measured bytes-per-chunk of stage s (not the target).
Train FLOPs/byte = 3 x forward (backward = 2 x forward).
"""
from __future__ import annotations

import math


def mean_ctx(L: float, window: int | None) -> float:
    if window is None or window >= L:
        return (L + 1) / 2
    return (window * (window + 1) / 2 + (L - window) * window) / L


def block_flops(d: int, hidden: int, L: float, window: int | None) -> float:
    return 8 * d * d + 6 * d * hidden + 4 * d * mean_ctx(L, window)


def layout_flops(layout: str, d: int, hidden: int, L: float, window: int | None, ssm: dict | None) -> float:
    """Forward FLOPs per position of one stack. window: keys per query (None = global)."""
    from .blocks import parse_layout  # single source of truth for the layout grammar

    total = 0.0
    for mixer, mlp in parse_layout(layout):
        if mixer == "attn":
            total += 8 * d * d + 4 * d * mean_ctx(L, window)
        else:
            assert ssm, "Mamba-2 blocks need model.ssm"
            total += mamba2_flops(d, ssm)
        if mlp:
            total += 6 * d * hidden
    return total


def _hidden(d, ratio, multiple_of):
    from .blocks import swiglu_hidden  # single source of truth for the MLP width

    return swiglu_hidden(d, ratio, multiple_of)


def main_block_passes(m: dict) -> int:
    """Block passes per main position at eval: layers x loops (plain), prelude + R x core + coda (parcae, at the
    mean recurrence R; training samples the depth per row, hnet/parcae.py)."""
    if m.get("kind", "plain") == "parcae":
        p = m["parcae"]
        return int(p["prelude"]) + int(p["mean_recurrence"]) * int(p["core"]) + int(p["coda"])
    return int(m["layers"]) * int(m.get("loops", 1))


def flops_per_byte(cfg: dict, seq_len: int, measured_ratios: list[float]) -> dict:
    """cfg is the model config dict. Returns forward/train FLOPs per byte and a per-level breakdown."""
    if cfg.get("kind") == "ref_hnet":
        return ref_flops_per_byte(cfg, seq_len, measured_ratios)
    dims, stages = list(cfg["d_model"]), list(cfg["stages"])
    assert len(measured_ratios) == len(stages), "one measured ratio per stage"
    mlp_ratio, mult = cfg["mlp_ratio"], cfg["mlp_multiple_of"]
    V = cfg["vocab_size"]
    breakdown, per_byte_frac, L = {}, 1.0, float(seq_len)
    total = 2 * dims[0] * V  # LM head at byte level
    breakdown["head"] = total
    from .model import stage_layouts

    ssm = cfg.get("ssm")
    for s, st in enumerate(stages):
        d, h = dims[s], _hidden(dims[s], mlp_ratio, mult)
        enc_layout, dec_layout = stage_layouts(st)
        enc = layout_flops(enc_layout, d, h, L, st["window"], ssm)
        dec = layout_flops(dec_layout, d, h, L, st["window"], ssm)
        rname = st["router"]["name"]
        router = {"lookahead": 4 * d * d, "given": 0, "surprise": 4 * d * d + 2 * d * 256}.get(rname, 2 * d)
        extra = router + 2 * d * d
        r = measured_ratios[s]
        if st.get("main_read"):
            to_main = math.prod(measured_ratios[s:])   # stage-s positions per main position
            extra += 4 * d * d + 4 * d * mean_ctx(L / to_main, None) + 4 * d * d / to_main
        level = per_byte_frac * (enc + dec + extra)
        breakdown[f"stage{s}"] = level
        total += level
        per_byte_frac /= r
        L /= r
    d, h = dims[-1], _hidden(dims[-1], mlp_ratio, mult)
    main = per_byte_frac * main_block_passes(cfg["main"]) * block_flops(d, h, L, None)
    if cfg["main"].get("kind", "plain") == "parcae":
        main += per_byte_frac * 2 * (2 * d * d)            # the injection B e (once per position) and the output C
    breakdown["main"] = main
    total += main
    return dict(fwd=total, train=3 * total, breakdown=breakdown, main_seq_len=L)


def token_flops_per_byte(cfg: dict, seq_len: int, bytes_per_token: float) -> dict:
    """Forward/train FLOPs per BYTE of a token-level model (model.kind == "token": no stages, one isotropic net).

    One token costs `flops_per_byte(cfg, seq_len, [])["fwd"]` (blocks + LM head 2 d V; `seq_len` is the token window).
    A byte costs that divided by the measured bytes per token.
    """
    assert not cfg["stages"], "a token model has no stages"
    per_tok = flops_per_byte(cfg, seq_len, [])
    fwd = per_tok["fwd"] / bytes_per_token
    return dict(fwd=fwd, train=3 * fwd, fwd_per_token=per_tok["fwd"], breakdown=per_tok["breakdown"],
                main_seq_len=float(seq_len), bytes_per_token=bytes_per_token)


# ---- reference H-Net (model.kind == "ref_hnet", goombalab/hnet layouts) ----
#   layout string "m4T1": m = Mamba-2 block, t = attention block; UPPER case adds a SwiGLU MLP (d_intermediate[s],
#   rounded up to a multiple of 128 as in the reference SwiGLU). Same counting rules as above, plus Mamba-2:
#     in_proj 2 d (2 di + 2 G N + H) + causal conv 2 k (di + 2 G N) + scan 4 di N (recurrent form: state update and
#     readout, the analogue of counting only causal attention) + out_proj 2 di d ;  di = expand d, H = di / headdim, G = 1
#   attention window w (reference meaning): keys i-w..i, so the window in mean_ctx is w + 1.

def _ref_hidden(d_int: int) -> int:
    return (d_int + 127) // 128 * 128


def mamba2_flops(d: int, ssm: dict) -> float:
    di, N, k = ssm["expand"] * d, ssm["d_state"], ssm["d_conv"]
    H = di // ssm.get("headdim", 64)    # mamba_ssm Mamba2 default headdim 64 (the reference ssm_cfg leaves it out)
    return 2 * d * (2 * di + 2 * N + H) + 2 * k * (di + 2 * N) + 4 * di * N + 2 * di * d


def _ref_layout_flops(layout: str, d: int, d_int: int, ssm: dict, window: int, L: float) -> float:
    w = None if window is None or window < 0 else window + 1
    return layout_flops(layout, d, _ref_hidden(d_int), L, w, ssm)


def ref_flops_per_byte(cfg: dict, seq_len: int, measured_ratios: list[float]) -> dict:
    r = cfg["ref"]
    dims, d_int, ssm, windows = list(cfg["d_model"]), list(r["d_intermediate"]), r["ssm_cfg"], r["attn_cfg"]["window_size"]
    assert len(measured_ratios) == len(cfg["stages"]), "one measured ratio per stage"
    layout, breakdown, frac, L = r["arch_layout"], {}, 1.0, float(seq_len)
    total = 2 * dims[0] * cfg["vocab_size"]
    breakdown["head"] = total
    for s in range(len(cfg["stages"])):
        enc, inner, dec = layout
        d = dims[s]
        level = _ref_layout_flops(enc, d, d_int[s], ssm, windows[s], L) + _ref_layout_flops(dec, d, d_int[s], ssm, windows[s], L)
        level += 4 * d * d + 2 * d * d                                   # router q/k + residual proj
        breakdown[f"stage{s}"] = frac * level
        total += frac * level
        frac /= measured_ratios[s]
        L /= measured_ratios[s]
        layout = inner
    (main_layout,) = layout
    S = len(cfg["stages"])
    main = frac * _ref_layout_flops(main_layout, dims[S], d_int[S], ssm, windows[S], L)
    breakdown["main"] = main
    total += main
    return dict(fwd=total, train=3 * total, breakdown=breakdown, main_seq_len=L)
