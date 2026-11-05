"""Same fixed batch, same seed: loss, grads and routing must match values saved from the code before the throughput work."""
import os

import pytest
import torch

from hnet.chunking import chunk_width, select_chunks
from tests import golden_util as g

CUDA = os.environ.get("HNET_TEST_CUDA") == "1" and torch.cuda.is_available()


def _compare(now: dict, ref: dict, loss_tol: float, grad_rtol: float, grad_atol: float):
    for a, b in zip(now["stages"], ref["stages"]):
        assert a["counts"] == b["counts"], "routing counts changed"
        assert a["anchors_sha1"] == b["anchors_sha1"], "anchor positions changed"
    assert abs(now["loss"] - ref["loss"]) < loss_tol, (now["loss"], ref["loss"])
    assert abs(now["ce"] - ref["ce"]) < loss_tol
    assert set(now["grad_norms"]) == set(ref["grad_norms"])
    bad = {n: (v, ref["grad_norms"][n]) for n, v in now["grad_norms"].items()
           if abs(v - ref["grad_norms"][n]) > grad_atol + grad_rtol * abs(ref["grad_norms"][n])}
    assert not bad, bad


def test_cpu_fp32_golden_eager():
    m, x, y, lengths = g.build_cpu()
    _compare(g.capture(m, x, y, torch.device("cpu"), amp=False, lengths=lengths), g.load("cpu_fp32.json"),
             loss_tol=1e-5, grad_rtol=1e-3, grad_atol=1e-6)


@pytest.mark.skipif(not CUDA, reason="set HNET_TEST_CUDA=1 on a GPU box")
@pytest.mark.parametrize("compile_", [False, True])
def test_gpu_bf16_golden(compile_):
    m, x, y = g.build_gpu(compile_=compile_)
    # bf16: loss within 2e-2, per-parameter grad norm within 10% (+ small floor)
    _compare(g.capture(m, x, y, torch.device("cuda"), amp=True), g.load("gpu_bf16.json"),
             loss_tol=2e-2, grad_rtol=0.10, grad_atol=1e-3)


def _sort_select(mask, multiple):
    """The original sort-based select_chunks, kept as the reference."""
    B, L = mask.shape
    counts = mask.sum(dim=1)
    M = min(L, -(-int(counts.max()) // multiple) * multiple)
    key = torch.arange(L)[None, :] + (~mask).long() * L
    idx = torch.sort(key, dim=1).values[:, :M]
    idx = torch.where(idx >= L, torch.zeros_like(idx), idx)
    return idx, counts, torch.arange(M)[None, :] < counts[:, None]


@pytest.mark.parametrize("density", [0.05, 0.3, 0.9])
def test_chunk_index_matches_sort(density):
    torch.manual_seed(1)
    mask = torch.rand(5, 256) < density
    for a, b in zip(select_chunks(mask, 32), _sort_select(mask, 32)):
        assert torch.equal(a, b)


def test_no_anchors_gives_valid_empty_width():
    mask = torch.zeros(2, 256, dtype=torch.bool)
    idx, counts, valid = select_chunks(mask, 32)
    assert idx.shape == (2, 32) and not valid.any() and chunk_width(counts, 256, 32) == 32
