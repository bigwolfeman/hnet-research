"""BPE reference ratio math, cache, and the per-domain metric keys."""
import json

import numpy as np
import pytest

from hnet import bpe
from hnet.evaluation import domain_metrics, stage_stats
from hnet.model import LN2


def test_ratio_math():
    assert bpe.hnet_over_bpe(8.0, 4.0) == 2.0
    assert bpe.hnet_over_bpe(3.0, 4.0) == 0.75


def test_count_bytes_and_tokens_skips_separators():
    data = np.frombuffer(b"abcd\x00ef\x00\x00ghij", dtype=np.uint8)
    n_bytes, n_tokens = bpe.count_bytes_and_tokens(data, lambda texts: [list(t) for t in texts], docs_per_batch=2)
    assert (n_bytes, n_tokens) == (10, 10)
    n_bytes, n_tokens = bpe.count_bytes_and_tokens(data, lambda texts: [[0] * (len(t) // 2) for t in texts])
    assert (n_bytes, n_tokens) == (10, 2 + 1 + 2)


def test_cache_roundtrip_and_invalidation(tmp_path):
    pytest.importorskip("tiktoken")
    f = tmp_path / "v.bin"
    f.write_bytes(b"hello world, hello tokens\x00second doc here")
    bpt = bpe.bytes_per_token(f, "cl100k_base")
    cp = bpe.cache_path(f, "cl100k_base")
    c = json.loads(cp.read_text())
    assert c["bytes_per_token"] == bpt == c["text_bytes"] / c["tokens"] and c["file_bytes"] == f.stat().st_size
    c["bytes_per_token"] = 123.0
    cp.write_text(json.dumps(c))
    assert bpe.bytes_per_token(f, "cl100k_base") == 123.0          # served from cache
    f.write_bytes(f.read_bytes() + b" more")                        # size change invalidates
    assert bpe.bytes_per_token(f, "cl100k_base") != 123.0


def test_stage_stats_cumulative_and_relative():
    cum, rel = stage_stats([250, 100], 1000)         # 1000 bytes -> 250 chunks -> 100 chunks
    assert cum == [4.0, 10.0] and rel == [4.0, 2.5]


def test_domain_metrics_keys_and_values():
    m = domain_metrics("go", ce_mean=LN2, anchor_sums=[250, 100], total_bytes=1000, bpt=4.0)
    assert m["val/go/bpb"] == pytest.approx(1.0)
    assert m["val/go/bytes_per_chunk_s0"] == 4.0 and m["val/go/bytes_per_chunk_s1"] == 10.0
    assert m["val/go/hnet_over_bpe_s0"] == 1.0 and m["val/go/hnet_over_bpe_s1"] == 2.5
    assert m["val/go/ratio_s1"] == 2.5
    assert "val/go/hnet_over_bpe_s0" not in domain_metrics("go", 1.0, [250], 1000, None)
