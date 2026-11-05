# Agent Note: H-Net spike throughput changes

Status: implemented

## Problem

The 2-stage byte H-Net smoke trained at 240-360 kB/s on the 5090 with the GPU idle about half the time. Causes found:
stage 1 used dense-mask SDPA (full L^2), the chunk/EMA glue ran eager (about 1000 tiny launches per step), the
Hillis-Steele scan touched the whole tensor log2(M) times, and the loader did synchronous random reads on a slow disk.

## Decision

- Stage 1 uses `flex_attention` with dynamic shapes (config default; `attn_impl: sdpa_mask` stays available).
- `Stage.forward` is three compiled pieces (`_pre`, `_gather_in`, `_scatter_out`) around the single host sync that
  reads the padded anchor width M. One sync per stage remains; it cannot go because M sets the next stage's shape.
- `select_chunks` builds indices with cumsum plus scatter instead of a sort. Output is identical.
- `ema_scan` is two-level (chunks of 16, then a carry scan). Same math, fp32 summation order differs.
- `MixLoader`/`ByteLoader` take `prefetch=N`: one producer thread keeps N device batches ready. Batch order for a seed is unchanged.
- Fixed: a batch with no anchors gave M=0 and crashed the next gather. M is now at least `chunk_multiple`.
- Golden tests (`tests/test_golden.py`, `tests/golden/*.json`, made by `scripts/make_golden.py` from the pre-change code) pin loss, per-parameter grad norms and routing.

## Alternatives considered

- Drop the host sync with a fixed M = L: rejected, it multiplies inner-stage compute.
- `associative_scan` for the EMA: not tried, backward and dynamic-shape support is less certain than the plain two-level scan.
- CUDA graphs: not possible while M changes per step.

## Consequences

Compile time grows (about 4 minutes cold for 22 graphs; the inductor cache shortens reruns). Peak memory fell with the
fused glue. The first step of a run is slow; do not read speed from the first 300 steps. Numbers: see the PR report.

The compiled glue needs torch 2.14 or newer. On torch 2.11 (cu128), Inductor fails an assert in
`tiling_utils.analyze_memory_coalescing` while it compiles the stage-1 gather. The spike pins `torch>=2.14`.
Measured on the RTX 5090 (torch 2.14.1+cu130), 2-stage `lookahead`, seq 4096, steps 300-400:
1.13 MB/s at batch 16 (58 ms/step) and 1.06 MB/s at batch 24. Before the merge, the same arm ran at 0.84-0.90 MB/s.
Batch 16 stays the default.
