"""Torch stand-in for the parts of `flash_attn` 2.8 that goombalab/hnet imports.

The flash-attn package has no build for torch 2.14 + CUDA 13 + sm_120, and the constitution keeps it out of the
project. Only the calls the reference H-Net makes are implemented, with flash-attn's semantics:

    qkv/kv packed attention, causal, window_size=(left, -1): key j is visible to query i iff i - left <= j <= i
    (left = -1: no window). Queries align to the END of the keys (bottom-right causal), as in flash-attn >= 2.1.
    varlen = the same per sequence, given cu_seqlens.
Anything else raises NotImplementedError instead of returning a wrong answer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__version__ = "2.8.0.post2+torchshim"


def _attend(q, k, v, softmax_scale, causal: bool, window_size):
    """q (B,Sq,H,D), k/v (B,Sk,H,D) -> (B,Sq,H,D)."""
    if not causal:
        raise NotImplementedError("non-causal attention is not used by the reference H-Net")
    left, right = window_size
    if right not in (-1, 0):
        raise NotImplementedError(f"window_size right={right}")
    Sq, Sk = q.shape[1], k.shape[1]
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    if Sq == Sk and (left < 0 or left >= Sk - 1):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=softmax_scale)
    else:
        i = torch.arange(Sq, device=q.device)[:, None] + (Sk - Sq)   # bottom-right aligned query positions
        j = torch.arange(Sk, device=q.device)[None, :]
        allowed = j <= i
        if left >= 0:
            allowed &= j >= i - left
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=softmax_scale)
    return out.transpose(1, 2)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
    assert dropout_p == 0.0 and not kw.get("return_attn_probs", False)
    q, k, v = qkv.unbind(dim=2)
    return _attend(q, k, v, softmax_scale, causal, window_size)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
    assert dropout_p == 0.0 and not kw.get("return_attn_probs", False)
    k, v = kv.unbind(dim=2)
    return _attend(q, k, v, softmax_scale, causal, window_size)


def _varlen(qf, kf, vf, cu_q, cu_k, softmax_scale, causal, window_size):
    outs = []
    for b in range(cu_q.numel() - 1):
        q0, q1, k0, k1 = int(cu_q[b]), int(cu_q[b + 1]), int(cu_k[b]), int(cu_k[b + 1])
        outs.append(_attend(qf[None, q0:q1], kf[None, k0:k1], vf[None, k0:k1], softmax_scale, causal, window_size)[0])
    return torch.cat(outs, dim=0)


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False,
                                     window_size=(-1, -1), **kw):
    assert dropout_p == 0.0
    q, k, v = qkv.unbind(dim=1)
    return _varlen(q, k, v, cu_seqlens, cu_seqlens, softmax_scale, causal, window_size)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                                    softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
    assert dropout_p == 0.0
    k, v = kv.unbind(dim=1)
    return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, causal, window_size)


def flash_attn_with_kvcache(*args, **kwargs):
    raise NotImplementedError("kv-cache decoding is not shimmed (the probe and training use full forward passes)")
