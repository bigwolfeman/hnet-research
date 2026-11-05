# Agent Note: Throughput: compiled stack segments and packed inner levels

Status: implemented

## Problem

The 2-stage reference layout (`reflayout_2stage`: m4 | T1m4 | T10 | m4T1 | m4, batch 16 x 4096 bytes) trained at
about 87 ms/step on the RTX 5090 with trained weights. Three costs were not part of the model math:

- Inner levels ran on `(B, M)` rows right-padded to the longest row. With the 0.9 / 0.1 edu / Go mix, 30-35% of
  stage-1 and main positions were padding.
- Each Mamba-2 call is a graph break (Dynamo cannot trace mamba_ssm's Triton autotuner). A graph break inside the
  block loop made Dynamo skip `Stack._body` and compile each `Block.forward` as its own frame. The residual stream
  `x` then entered two graphs per Mamba-2 block (norm, residual add), so autograd added its fp32 gradients with eager
  kernels, and autocast cast every Mamba-2 input from fp32 to bf16 with an eager copy (and back in backward).
- `make_pack` wrote `cu[-1] = T_pad`, a pageable host-to-device copy that syncs, and ran about 20 eager launches while
  the GPU sat idle right after the stage's one host sync.

## Decision

- `model.pack_inner: true` (default): inner levels run their stacks on valid positions only, as one `(1, T_pad)`
  sequence (`hnet/packing.py`). Mamba-2 gets `seq_idx`; attention runs `varlen_attn` with `cu_seqlens`.
  `ParcaeMain` keeps the padded layout (`uses_pack = False`).
- `Stack.forward` runs in segments, the code between two Mamba-2 calls (`hnet/blocks.py` `split_segments`,
  `run_segment`). Each segment compiles as one graph: the residual add fuses with the next norm, `x` flows from graph
  to graph, and the Mamba-2 input leaves the graph already cast to the autocast dtype. Mamba-2 runs eagerly between
  segments. All stacks share `run_segment`'s compile cache; segments of the same structure share a graph.
  `Stack.compile_blocks` raises Dynamo's `recompile_limit` to 64 (training uses about 14 entries).
- `pack_index` builds the packed layout with `nonzero_static` and no host sync; `Stage.compile_glue` compiles it.
  Its fill value is -1, then `where`: a fill of `B * M` specialized the graph on M (one recompile per M).
- `Block.finish` (residual add + MLP half) is the one definition of the post-mixer half of a block, used by
  `Block.forward` and by the segments.

Measured (RTX 5090, torch 2.14.1, trained weights `reflayout_2stage_s0/final.pt`, 80 warmup steps, mean of 30 steps):

| Change | ms/step | peak GB |
|---|---|---|
| padded inner levels, per-block compile | 86.8 | 10.3 |
| + packed inner levels | 74.0 | 7.9 |
| + no `cu[-1]` sync | 74.7 | 7.9 |
| + compiled segments | 70.8-71.4 | 8.0 |
| + compiled `pack_index` | 71.1-71.8 | 8.0 |

## Alternatives considered

- Compile `Stack._body` whole and let the Mamba-2 graph breaks split it: what the first version did; Dynamo skipped
  the frame and compiled per block (above).
- `torch._dynamo.config.nested_graph_breaks`: experimental in torch 2.14; not tried, the explicit segments give a
  fixed, documented graph structure.
- One compile entry point per segment kind (shorter Dynamo cache lookups after the host sync): no measurable gain in
  an A/B of 100 steps; reverted.
- Mamba-2 `use_mem_eff_path=False` (save activations instead of recomputing conv1d and the gated norm in backward):
  74.5 vs 74.7 ms, +1.3 GB. The autograd split/cat copies cost what the recompute saves.
- More row groups in mamba_ssm's gated-RMSNorm backward launch: at most 3% of that kernel; it already runs at about
  80% of DRAM bandwidth.
- Inductor `max_autotune_gemm` for the GEMMs inside compiled graphs (about 13 ms/step): median 71.8 vs 71.8 ms,
  cold compile 213 s vs 41 s, and some Triton templates fail on the 5090's 101 KB shared memory. Not enabled.
- Batch 24 or 32, CUDA graphs: batch changes are out of scope (batch 32 also runs out of memory); CUDA graphs need
  fixed shapes and M changes every step.

## Consequences

- Same math per valid position; pad slots of an unpacked inner output are 0 (the padded path left values nobody
  reads). Golden tests and `tests/test_packing.py` pin it.
- Two host syncs per step remain (one per stage, `chunk_stats`). After each, the GPU idles about 0.6 ms while the
  host runs the gather, the pack and the first segment's Dynamo and AOTAutograd wrappers.
- Recompiles stop after warmup except when M first crosses a power-of-two bound of `ema_scan`'s carry scan
  (`log2(M / 16)` steps): about one 2.7 s recompile per run per bucket, at most two buckets per level.
- The 5090 runs at its 500 W cap (throttle reason SW power cap); sustained step time can drift 5-7% between runs.
  Compare variants in the same session and the same step window.
- Remaining profile (71 ms/step): Mamba-2 kernels 31 ms, GEMMs 22 ms (162-194 TFLOP/s on the large ones), Inductor
  kernels 8.4 ms (bandwidth bound), attention 2.6 ms, AdamW + clipping 2.0 ms, mamba_ssm eager reductions and copies
  about 3 ms, GPU idle about 2.4 ms.

- Next steps that keep the math, with estimates: save `xBC_conv` in Mamba-2's backward instead of recomputing
  causal_conv1d (about 1.5 ms/step, +1.1 GB, needs a vendored copy of `MambaSplitConv1dScanCombinedFn`); drop
  mamba_ssm's eager `.sum` over size-1 split dims, flips and cumsums in the SSD backward (about 1 ms/step, needs a
  patched `_mamba_chunk_scan_combined_bwd`). A fused Mamba-2 block kernel is the large remaining step.

Earlier throughput work: [2026-10-31-hnet-spike-throughput.md](2026-10-31-hnet-spike-throughput.md).
