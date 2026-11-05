"""Token-level data helpers for the BPE control: byte lengths of tokens, eval windows, bpb math.

Pseudocode:
    token_nbytes[id]  = number of raw bytes the token stands for (GPT-2 byte<->unicode map); EOS = 1
                        (EOS stands for the 0 separator byte, so sum(nbytes[tokens]) == size of the .bin file)
    eval_windows(tokens, L, n):  n windows of L+1 tokens at evenly spaced, non-overlapping starts; x = w[:-1], y = w[1:]
    window_bytes(w)           = sum(nbytes[w[:, 1:]])                    # the bytes the targets stand for
    bpb                       = sum(token CE in nats) / (ln 2 * bytes)   # every byte of the text, exact
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
META = "meta.json"
NBYTES = "token_nbytes.npy"


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte -> printable unicode char map (the alphabet of byte-level BPE)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_nbytes_table(vocab: dict[str, int], vocab_size: int, special: set[str]) -> np.ndarray:
    """Raw byte length of every token id. Special tokens (EOS) count 1: they stand for the document separator byte."""
    inv = {c: b for b, c in bytes_to_unicode().items()}
    out = np.zeros(vocab_size, dtype=np.uint16)
    for tok, i in vocab.items():
        if tok in special:
            out[i] = 1
        else:
            assert all(c in inv for c in tok), f"token {tok!r} is not byte-level"
            out[i] = len(tok)  # one mapped char per byte
    assert (out > 0).all(), "every id needs a positive byte length"
    return out


def load_meta(token_dir: str | Path) -> dict:
    return json.loads((Path(token_dir) / META).read_text())


def load_nbytes(token_dir: str | Path) -> np.ndarray:
    return np.load(Path(token_dir) / NBYTES)


def eval_windows(tokens: np.ndarray, seq_len: int, n_windows: int, multiple_of: int = 1) -> np.ndarray:
    """(n, seq_len+1) int32 windows at evenly spaced starts that never overlap in their targets.

    n is cut to a multiple of `multiple_of` (the batch size) and to what fits. Same tokens give the same windows.
    """
    n_fit = (len(tokens) - 1) // seq_len
    n = min(n_windows, n_fit) // multiple_of * multiple_of
    assert n > 0, f"{len(tokens)} tokens cannot fill one batch of {multiple_of} windows of {seq_len}"
    stride = n_fit // n * seq_len                      # multiple of seq_len, so targets of two windows never overlap
    starts = np.arange(n) * stride
    return np.stack([np.asarray(tokens[s: s + seq_len + 1]) for s in starts]).astype(np.int32)


def window_bytes(windows: np.ndarray, nbytes: np.ndarray) -> int:
    """Raw bytes the targets (all but the first token) of these windows stand for."""
    return int(nbytes[windows[:, 1:]].astype(np.int64).sum())


def bpb(ce_sum_nats: float, n_bytes: int) -> float:
    """Bits per byte of a text: total token CE (nats) over ln2 times the text's bytes."""
    return ce_sum_nats / (LN2 * n_bytes)
