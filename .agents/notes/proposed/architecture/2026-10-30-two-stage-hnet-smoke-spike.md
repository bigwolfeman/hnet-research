# Agent Note: Two-stage byte H-Net smoke spike

Status: proposed

## Problem

Phase 1 of the [constitution](../../../../docs/constitution.md) needs a 2-stage byte H-Net with Transformer stages and a
router we can swap by config. We need to know that the code is causal, shape-stable, and trains, before any
real experiment. The spike lives in `lab/spikes/2026-10-30-two-stage-hnet-smoke/`.

## Proposal

Promote the spike package to `src/hnet/` after human review. Design choices made in the spike:

- One recursive `Stage(encoder, router, inner, decoder)`; `Main` is the leaf with the same interface.
- Router registry (`lookahead`, `eoc`). A router returns a `ChunkPlan` (anchors, p, STE confidence, read index).
  The chunking code does not know which router ran.
- EMA smoothing is an fp32 Hillis-Steele scan over (a, b) affine pairs. The 2025 code uses a Mamba-2 kernel for this.
- Chunk counts are right-padded to a multiple of 128. One host sync per stage.
- Sliding-window attention via `flex_attention` at stage 0 (fixed length). Deeper stages use dense-mask SDPA,
  because their lengths change every step and flex recompiles per shape.
- `eoc` has a `read_own_end` field. With `true` (the default), byte t reads the latest chunk that ends at or before t, so a chunk's last byte reads that chunk. With `false` (`eoc_lag`), every byte of chunk `k` reads chunk `k-1`, which wastes the freshest context at the byte that predicts the next chunk. `force_last=true` is rejected with `read_own_end=true`, because the forced boundary encodes the sequence end. Causality is tested for `eoc`, `eoc_lag`, and `lookahead` by perturbing future bytes and by comparing prefix runs.

- Train data is a weighted multi-source loader (`MixLoader`): per batch row pick a source by weight, then a random offset. Default mix by bytes: FineWeb-Edu 0.9, StarCoderData Go 0.1. Files are not re-mixed on disk, so the mix is a config change.
- Eval runs per domain (edu, go, sql). SQL is eval only. Each domain logs `val/{domain}/bpb`, bytes per chunk per stage, and `hnet_over_bpe_s{k}`: H-Net bytes per chunk at stage k divided by cl100k bytes per token on the same val file. This is the quantity behind constitution success criterion 4 (Cartesia Figure 6). Bytes per chunk is cumulative in raw bytes, so stage 1 is the product of the stage ratios. The BPE reference counts the whole val file once and caches it as JSON next to the `.bin`.
- Compile parity is a GPU test. Router decisions are hard thresholds, so random weights in bf16 flip them (eager fp32 vs eager bf16 differs by 0.66 mean logit). The test uses default init in bf16 and random weights in fp32.

## Alternatives considered

- Pre-mixed training file on disk: fixes the mix at build time and costs another 14 GB copy. Rejected for the loader-side mix.
- Per-batch (not per-row) source choice: a batch is then pure code or pure text, which makes the gradient noisier. Rejected.
- Per-stage ratio (relative to the stage input) as the BPE comparison: stage 1 would compare chunks of chunks. Cumulative raw bytes per chunk is the comparable quantity.

- Associative scan via `torch._higher_order_ops.associative_scan`: experimental API, skipped.
- Log-space cumulative product for the EMA: underflows when p is near 1 over long chunks. Hillis-Steele is stable and simple.
- Packed (cu_seqlens) layout like the 2025 code: needs varlen kernels. Right padding is simpler and causal attention makes it exact.
- flex_attention at every stage: one recompile per new chunk-length shape; the first 40-step Spark probe took minutes to compile.
- Eager execution without `torch.compile`: 0.70 s/step vs 0.44 s/step compiled at batch 16 on the GB10 (memory-bandwidth bound elementwise ops).

## Acceptance criteria

CPU gates pass (causality, padding, EMA, finite). GPU smoke runs for both routers show falling loss, ratios moving
toward targets without collapse, and probes logging. Numbers: see the spike README (`lab/spikes/2026-10-30-two-stage-hnet-smoke/README.md`, Smoke results).

## Risks

- `eoc` forced-last boundary makes the last chunk vector unused by any decoder position. Harmless for logits, counted in ratios.
- Ratio loss weight 0.03 may need to rise for `eoc` (constitution notes a "strong auxiliary loss").
- The `eoc` router is our reading of an unpublished design. A good smoke result does not validate that reading.
