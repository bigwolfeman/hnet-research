"""BPE reference compression: bytes per token of a tokenizer on a .bin eval file (Cartesia Figure 6 denominator).

Pseudocode:
    bytes_per_token(bin, encoding):
        cache = <bin>.bpe-<encoding>.json ; reuse if cache.file_bytes == size(bin)
        docs  = split bin on byte 0 ; text = utf8 decode (errors=replace) ; tokens = encode_ordinary(text)
        bpt   = sum(len(doc bytes)) / sum(len(tokens))                 # separators are not counted on either side
    hnet_over_bpe(bytes_per_chunk, bpt) = bytes_per_chunk / bpt         # > 1: the H-Net chunk is longer than a BPE token
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import SEP


def hnet_over_bpe(bytes_per_chunk: float, bytes_per_token: float) -> float:
    return float(bytes_per_chunk) / float(bytes_per_token)


def cache_path(bin_path: str | Path, encoding: str) -> Path:
    p = Path(bin_path)
    return p.with_name(f"{p.stem}.bpe-{encoding}.json")


def count_bytes_and_tokens(data: np.ndarray, encode_batch, docs_per_batch: int = 512) -> tuple[int, int]:
    """data: uint8 array, docs separated by SEP. encode_batch(list[str]) -> list[list[int]]."""
    cuts = np.flatnonzero(data == SEP)
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts, [len(data)]])
    n_bytes = n_tokens = 0
    for i in range(0, len(starts), docs_per_batch):
        raw = [data[s:e].tobytes() for s, e in zip(starts[i:i + docs_per_batch], ends[i:i + docs_per_batch]) if e > s]
        n_bytes += sum(len(r) for r in raw)
        n_tokens += sum(len(t) for t in encode_batch([r.decode("utf-8", errors="replace") for r in raw]))
    return n_bytes, n_tokens


def bytes_per_token(bin_path: str | Path, encoding: str) -> float:
    """Cached bytes-per-token of tiktoken `encoding` over the whole file. Cache sits next to the .bin."""
    cp = cache_path(bin_path, encoding)
    size = Path(bin_path).stat().st_size
    if cp.exists():
        c = json.loads(cp.read_text())
        if c["file_bytes"] == size:
            return float(c["bytes_per_token"])
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    data = np.memmap(bin_path, dtype=np.uint8, mode="r")
    n_bytes, n_tokens = count_bytes_and_tokens(
        data, lambda texts: enc.encode_ordinary_batch(texts, num_threads=1))
    out = dict(encoding=encoding, file_bytes=size, text_bytes=n_bytes, tokens=n_tokens, bytes_per_token=n_bytes / n_tokens)
    cp.write_text(json.dumps(out, indent=1))
    return out["bytes_per_token"]
