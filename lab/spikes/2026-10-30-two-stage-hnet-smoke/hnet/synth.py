"""Synthetic "recall" language: a byte stream where next-byte prediction needs a document-level memory.

Pseudocode:
    lexicon   = n_lex distinct random words, length uniform in [min_len, max_len], letters uniform over a-z
    document  (recall):    vocab = doc_vocab distinct lexicon words, drawn uniformly without replacement
                           each word slot: w ~ uniform(vocab)
              (no recall): each word slot: w ~ uniform(lexicon)           (doc_vocab = None; a memory is worthless)
    after each word: delimiter = SEP (document end) with prob p_end, else " "
    stream    = concatenated documents (the trainer's MixLoader takes random windows of it)

Exact oracle (the generating distribution, conditioned on the full document so far):
    S = distinct words seen in this document, k = |S|
    recall:    P(w) = 1/V for w in S ; q = (V - k) / (V * (n_lex - k)) for every other lexicon word
    no recall: P(w) = 1/n_lex for every lexicon word
    P(next byte c | word prefix u) = mass(u + c) / mass(u),   mass(u) = sum of P(w) over words w with prefix u
    P(" " | u) = (1 - p_end) * P(w = u) / mass(u) ; P(SEP | u) = p_end * P(w = u) / mass(u)
It is a lower bound for any model: a model's context window can start inside a document and miss words of S.
reach = R: S = distinct words of this document that start in the last R bytes. That is the same posterior for a
predictor that sees only those bytes (it ignores the cut word at the window's start), so it is the best bpb of a
model whose every path reaches R bytes back. reach 0 = no memory; reach >= the document = the full oracle.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import SEP

SPACE = 32
LETTERS = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)


@dataclass(frozen=True)
class SynthSpec:
    n_lex: int = 8192
    min_len: int = 3
    max_len: int = 8
    doc_vocab: int | None = 32      # None: no recall (every slot uniform over the lexicon)
    p_end: float = 1 / 150          # document end after a word


def make_lexicon(spec: SynthSpec, rng: np.random.Generator) -> list[bytes]:
    """n_lex distinct words. A word may be a prefix of another; the delimiter ends every word."""
    words: set[bytes] = set()
    while len(words) < spec.n_lex:
        n = spec.n_lex - len(words)
        lens = rng.integers(spec.min_len, spec.max_len + 1, n)
        letters = LETTERS[rng.integers(0, 26, (n, spec.max_len))]
        words.update(bytes(letters[i, : lens[i]]) for i in range(n))
    return sorted(words)[: spec.n_lex]


def write_lexicon(path, lexicon: list[bytes]) -> None:
    """One word per line, newline-terminated (synth/lexicon.txt, shared by both languages)."""
    Path(path).write_bytes(b"\n".join(lexicon) + b"\n")


def read_lexicon(path, spec: SynthSpec) -> list[bytes]:
    """Inverse of write_lexicon; fails loud if the word count does not match the spec."""
    lex = Path(path).read_bytes().split(b"\n")[:-1]
    assert len(lex) == spec.n_lex, f"{path} has {len(lex)} words, spec says {spec.n_lex}"
    return lex


def generate(spec: SynthSpec, lexicon: list[bytes], n_bytes: int, rng: np.random.Generator) -> np.ndarray:
    """A uint8 stream of at least n_bytes (whole documents; the last document is cut at n_bytes)."""
    assert len(lexicon) == spec.n_lex
    lens = np.array([len(w) for w in lexicon], dtype=np.int64)
    pad = np.zeros((spec.n_lex, spec.max_len), dtype=np.uint8)
    for i, w in enumerate(lexicon):
        pad[i, : len(w)] = np.frombuffer(w, dtype=np.uint8)
    mean_word = lens.mean() + 1
    out, have = [], 0
    while have < n_bytes:
        n_words = min(int((n_bytes - have) / mean_word * 1.05) + 1024, 8_000_000)   # ~50M bytes per chunk
        ends = rng.random(n_words) < spec.p_end                        # SEP after this word
        ends[-1] = True                                                # every chunk holds whole documents
        if spec.doc_vocab is None:
            idx = rng.integers(0, spec.n_lex, n_words)
        else:
            doc = np.concatenate([[0], np.cumsum(ends[:-1])])          # document id of every word slot
            n_docs = int(doc[-1]) + 1
            vocab = np.stack([rng.choice(spec.n_lex, spec.doc_vocab, replace=False) for _ in range(n_docs)])
            idx = vocab[doc, rng.integers(0, spec.doc_vocab, n_words)]
        wl = lens[idx] + 1                                             # word + delimiter
        starts = np.concatenate([[0], np.cumsum(wl)[:-1]])
        wid = np.repeat(np.arange(n_words), wl)
        cpos = np.arange(int(wl.sum())) - starts[wid]
        delim = np.where(ends, SEP, SPACE).astype(np.uint8)
        chunk = np.where(cpos < lens[idx][wid], pad[idx[wid], np.minimum(cpos, spec.max_len - 1)], delim[wid])
        out.append(chunk.astype(np.uint8))
        have += len(chunk)
    return np.concatenate(out)[:n_bytes]


class _Prefixes:
    """Prefix counts of a word multiset: count[u] = words with prefix u, exact[u] = words equal to u."""

    def __init__(self, words=()):
        self.count: Counter = Counter()
        self.exact: Counter = Counter()
        for w in words:
            self.add(w)

    def add(self, w: bytes, n: int = 1):
        for i in range(len(w) + 1):
            self.count[w[:i]] += n
        self.exact[w] += n

    def remove(self, w: bytes):
        self.add(w, -1)


def oracle_bits(spec: SynthSpec, lexicon: list[bytes], stream: np.ndarray, reach: int | None = None) -> np.ndarray:
    """-log2 P(stream[t] | stream[:t]) under the generator, for t >= the first document start in `stream`.

    Positions before the first SEP are NaN (their document began before the stream). The stream must hold whole
    words after that SEP (a cut last word is scored up to the cut).
    """
    lex = _Prefixes(lexicon)
    data = stream.tobytes()
    bits = np.full(len(data), np.nan)
    first = data.find(bytes([SEP]))
    assert first >= 0, "stream has no document start"
    V, n = spec.doc_vocab, spec.n_lex
    seen, seen_set, u = _Prefixes(), set(), b""
    window: deque = deque()          # (start, word) of this document's words, oldest first (reach only)
    in_window: Counter = Counter()

    def mass(count_lex: int, count_seen: int) -> float:
        if V is None:
            return count_lex / n
        k = len(seen_set)
        q = (V - k) / (V * (n - k))
        return count_seen / V + q * (count_lex - count_seen)

    for t in range(first + 1, len(data)):
        c = data[t]
        while reach is not None and window and window[0][0] < t - reach:
            _, w = window.popleft()
            in_window[w] -= 1
            if in_window[w] == 0:
                seen_set.discard(w)
                seen.remove(w)
        m_u = mass(lex.count[u], seen.count[u])
        if m_u <= 0:
            raise ValueError(f"position {t}: prefix {u!r} has zero mass (an earlier byte was impossible)")
        if c == SPACE or c == SEP:
            p_word = mass(lex.exact[u], seen.exact[u]) / m_u
            p = p_word * ((1 - spec.p_end) if c == SPACE else spec.p_end)
            if V is not None:
                if u not in seen_set:
                    seen_set.add(u)
                    seen.add(u)
                if reach is not None:
                    window.append((t - len(u), u))
                    in_window[u] += 1
            u = b""
            if c == SEP:
                seen, seen_set = _Prefixes(), set()
                window.clear()
                in_window.clear()
        else:
            uc = u + bytes([c])
            p = mass(lex.count[uc], seen.count[uc]) / m_u
            u = uc
        bits[t] = -math.log2(p) if p > 0 else math.inf               # inf: the stream is impossible here
    return bits


def anchors(spec: SynthSpec, lexicon: list[bytes], stream: np.ndarray) -> dict:
    """Fixed bpb anchors on a stream: the oracle, the same generator with no document memory, letters only."""
    oracle = np.nanmean(oracle_bits(spec, lexicon, stream))
    no_mem = np.nanmean(oracle_bits(SynthSpec(spec.n_lex, spec.min_len, spec.max_len, None, spec.p_end), lexicon, stream))
    return dict(oracle_bpb=float(oracle), no_memory_bpb=float(no_mem))
