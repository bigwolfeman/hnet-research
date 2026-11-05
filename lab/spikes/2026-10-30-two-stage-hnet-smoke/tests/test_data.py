"""Weighted multi-source loader, data config invariants, and parquet -> bin preparation."""
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from hnet.data import SEP, MixLoader, prepare

DATA_CFG = Path(__file__).resolve().parents[1] / "configs" / "data" / "bytes.yaml"


def make_sources(tmp_path, weights, size=4096):
    """One file per source, filled with a unique byte, so a row's first byte names its source."""
    srcs = []
    for i, (name, w) in enumerate(weights.items()):
        f = tmp_path / f"{name}.bin"
        np.full(size, 65 + i, dtype=np.uint8).tofile(f)
        srcs.append(dict(name=name, file=str(f), weight=w))
    return srcs


def test_source_frequencies_match_weights(tmp_path):
    weights = {"edu": 0.9, "go": 0.1}
    ld = MixLoader(make_sources(tmp_path, weights), seq_len=16, batch_size=64, seed=0, device="cpu")
    counts = {n: 0 for n in weights}
    n_batches = 400
    for _ in range(n_batches):
        x, y = ld.next()
        first = x[:, 0].tolist()
        assert (x == x[:, :1]).all() and (y == x[:, :1]).all()      # a row never mixes sources
        for b in first:
            counts[{65: "edu", 66: "go"}[b]] += 1
    n = n_batches * 64
    for name, w in weights.items():
        sigma = (w * (1 - w) / n) ** 0.5
        assert abs(counts[name] / n - w) < 4 * sigma, (name, counts[name] / n, w)


def test_weights_are_normalised_and_zero_weight_never_drawn(tmp_path):
    ld = MixLoader(make_sources(tmp_path, {"a": 3.0, "b": 1.0, "c": 0.0}), 16, 64, seed=1, device="cpu")
    seen = np.concatenate([(ld.next(), ld.last_sources)[1] for _ in range(100)])
    assert 2 not in seen
    assert abs((seen == 0).mean() - 0.75) < 0.03


def test_offsets_inside_file_and_y_is_shifted_x(tmp_path):
    f = tmp_path / "ramp.bin"
    np.arange(1000, dtype=np.int64).astype(np.uint8).tofile(f)
    ld = MixLoader([dict(name="r", file=str(f), weight=1)], 32, 8, seed=2, device="cpu")
    for _ in range(50):
        x, y = ld.next()
        assert x.shape == y.shape == (8, 32)
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert ((y[:, -1] - x[:, 0]) % 256 == 32).all()


def test_seed_reproducible(tmp_path):
    srcs = make_sources(tmp_path, {"a": 0.5, "b": 0.5})
    a = MixLoader(srcs, 16, 8, 7, "cpu").next()[0]
    b = MixLoader(srcs, 16, 8, 7, "cpu").next()[0]
    assert torch.equal(a, b)


def test_default_config_keeps_sql_out_of_train():
    cfg = OmegaConf.load(DATA_CFG)
    train = {d.name: d for d in cfg.train_sources}
    assert "sql" not in train and not any("sql" in d.file for d in cfg.train_sources)
    assert [d.name for d in cfg.val_sets] == ["edu", "go", "sql"]
    assert {d.name for d in cfg.val_sets} - set(train) == {"sql"}
    assert abs(sum(d.weight for d in cfg.train_sources) - 1.0) < 1e-9
    assert (train["edu"].weight, train["go"].weight) == (0.9, 0.1)
    assert cfg.primary_val in {d.name for d in cfg.val_sets}


def test_prepare_fills_targets_in_order_with_disjoint_docs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    docs = [f"doc{i} " + "x" * 20 + "\x00junk" for i in range(40)]
    pq.write_table(pa.table({"content": docs}), tmp_path / "a.parquet")
    stats = prepare(str(tmp_path / "*.parquet"), str(tmp_path / "out"), [("val.bin", 100), ("train.bin", 300)], "content")
    val = (tmp_path / "out" / "val.bin").read_bytes()
    train = (tmp_path / "out" / "train.bin").read_bytes()
    assert len(val) == stats["val.bin"]["bytes"] >= 100 and len(train) == stats["train.bin"]["bytes"] >= 300
    for blob in (val, train):
        assert blob.endswith(bytes([SEP])) and b"junk" in blob and blob.count(bytes([SEP])) == len(blob.split(bytes([SEP]))) - 1
    ids = lambda b: {d.split(b" ")[0] for d in b.split(bytes([SEP])) if d}
    assert not ids(val) & ids(train)
    with pytest.raises(RuntimeError):
        prepare(str(tmp_path / "*.parquet"), str(tmp_path / "o2"), [("v.bin", 10**6)], "content")


def test_starcoder_header_strip():
    import re
    from hnet.data import STARCODER_HEADER
    h = re.compile(STARCODER_HEADER)
    cases = {
        b"<reponame>a/b<filename>x.go<gh_stars>10-100\npackage main\n": b"package main\n",
        b"<reponame>a/b\n#sql\nSELECT 1;": b"#sql\nSELECT 1;",
        b"<filename>q.sql\nSELECT 1;": b"SELECT 1;",
        b"package main // <filename> is not a header here\n": b"package main // <filename> is not a header here\n",
    }
    for raw, want in cases.items():
        assert h.sub(b"", raw, count=1) == want


def test_prefetch_keeps_batch_order(tmp_path):
    """The prefetch thread must yield the exact batches inline reading yields for a given seed."""
    big = []
    for i, n in enumerate(("a", "b")):
        f = tmp_path / f"{n}.bin"
        np.random.default_rng(i).integers(0, 256, 200_000, dtype=np.uint8).tofile(f)
        big.append(dict(name=n, file=str(f), weight=0.5))
    inline = MixLoader(big, 512, 4, 7, torch.device("cpu"))
    pre = MixLoader(big, 512, 4, 7, torch.device("cpu"), prefetch=3)
    for _ in range(20):
        (xa, ya), (xb, yb) = inline.next(), pre.next()
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    pre.pf.close()


@pytest.mark.parametrize("prefetch", [0, 3])
def test_state_dict_continues_exact_stream(tmp_path, prefetch):
    """Resume: a loader built from state_dict() after k batches yields the batches k+1, k+2, ... of the original,
    also when the prefetch thread has already drawn batches past k."""
    big = []
    for i, n in enumerate(("a", "b")):
        f = tmp_path / f"{n}.bin"
        np.random.default_rng(i).integers(0, 256, 200_000, dtype=np.uint8).tofile(f)
        big.append(dict(name=n, file=str(f), weight=0.5))
    ref = MixLoader(big, 512, 4, 7, torch.device("cpu"), prefetch=prefetch)
    for _ in range(5):
        ref.next()
    sd = ref.state_dict()
    if prefetch:
        time.sleep(0.2)   # let the thread run ahead of the consumer
    resumed = MixLoader(big, 512, 4, 0, torch.device("cpu"), prefetch=prefetch, state=sd)   # seed is ignored
    for _ in range(8):
        assert torch.equal(ref.next()[0], resumed.next()[0])
    for ld in (ref, resumed):
        if ld.pf:
            ld.pf.close()


def test_prefetch_byteloader_order(tmp_path):
    from hnet.data import ByteLoader

    f = tmp_path / "v.bin"
    np.random.default_rng(3).integers(0, 256, 100_000, dtype=np.uint8).tofile(f)
    a, b = ByteLoader(str(f), 256, 3, 5, torch.device("cpu")), ByteLoader(str(f), 256, 3, 5, torch.device("cpu"), prefetch=2)
    for _ in range(10):
        assert torch.equal(a.next()[0], b.next()[0])
    b.pf.close()
