# Reference: Cartesia "From Tokens to Bytes" (Parts I and II)

Reference doc. Notes in our own words. Raw text copies for private reading live in `ignored/blogs-raw/` (gitignored, not redistributed).

- Part I: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1 (Oct 2026)
- Part II: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2 (Oct 2026)
- Original H-Net paper: [arXiv 2507.07955](https://arxiv.org/abs/2507.07955), code: `goombalab/hnet`

## What the posts disclose

The posts use an internal "much improved" H-Net. They do not publish its equations. Everything below the "Inferred" line is our reading, not their claim.

### Architecture (2-stage, SmolLM3-matched run)

| Part | Layers | Resolution |
|---|---|---|
| Stage-0 encoder / decoder | 4 Mamba-2 each | bytes |
| Stage-1 encoder / decoder | 1 sliding-window attention block + 4 Mamba-2 each | ~4x compressed |
| Main network | 40-layer Transformer, global attention every layer | ~8x compressed (measured 7.6x) |

- Target ratios: 4x at stage 0, 2x at stage 1, set from prior work.
- Outer stages cost 43% of FLOPs (vs 8% for embed + LM head in the BPE model). Stage 0: 31% FLOPs, 6% params. Stage 1: 12% FLOPs, 9% params. Main: 57% FLOPs, 85% params.
- Matched within 2% FLOPs/byte. 3.67B params vs 3.08B.
- Optimizer for that run: AdamW (0.9, 0.95), wd 0.1, clip 1.0, peak LR 2e-4, WSD, 15% cooldown. No H-Net-specific tuning.

### 1-stage ladder (controlled comparison)

- Inner network = the BPE Transformer rung. Outer networks are FLOP- and param-matched to embed + LM head.
- A "strong auxiliary loss" pins compression to 4.57 bytes/chunk (the BPE ratio).
- Ladder rule: D/L ~ 75, heads = layers, head dim 128, SwiGLU 3.6D, Muon + WSD.
- FLOP counting: count only causal attention FLOPs (non-causal counting unfairly penalizes long byte sequences).
- Metric: bits-per-byte (bpb).

### Results worth reproducing in direction (not scale)

- 1-stage H-Net beats matched BPE at every rung: 1.28x compute on validation, ~2x on downstream suites, ~2x on code, ~3x median on unseen code languages.
- H-Net compresses code *more* than BPE does, and still models it better.
- Part II, loss-to-loss: a byte Transformer transfers validation loss to downstream best. BPE transfers worse (1.61x), BPE without pretokenization worse still (2.02x). H-Net matches the byte model (1.05x) while keeping BPE-level compute efficiency.
- End-to-end boundaries win: H-Net (0.7027 bpb at 1e20) beats frozen boundaries from an H-Net proxy (0.7057), which beats entropy-threshold boundaries (0.7138). Same architecture and ratio in all three.

### Qualitative behaviors (our "hierarchy works" signals)

- Stage-0 chunks split words at morphemes: `track|ers`, `wonder|ed`, `near|by`.
- Stage-1 chunks merge words: "a foot", "quite a few". Context-dependent, not pattern-based.
- CJK and other multi-byte scripts segment on character boundaries.
- Code: chunks cross newlines and indentation (`__init__(\n self,`).
- In-context tokenization: a new made-up word is split into many chunks at first, then fewer chunks each time it repeats. Repeated formats get very large chunks. Recurring motifs (`LURD`, `2526`) become single chunks.
- Some of these behaviors appear only late in training (stated for DNA, shown as an animation for English).

### Old vs new H-Net

The posts call the 2025 model "LH-Net" (lazy / lookahead). It decides a chunk boundary only after it sees the first byte of the next chunk. The posts say this hurts next-byte prediction and multi-stage hierarchies, and that LH-Net stage-0 chunks are worse (`r|hino`, `fro|m the fact`). Multi-stage LH-Net also fails to segment CJK characters.

## Inferred (our hypotheses, not their claims)

1. **Lookahead comes from the 2025 router.** That router sets `p_t = (1 - cos(q_t, k_{t-1})) / 2`: byte `t` *starts* a chunk, so the end of the previous chunk is known only after reading byte `t`.
2. **The new router most likely marks chunk *ends*.** `p_t` depends on `x_{<=t}` and means "byte t is the last byte of a chunk". Then the main network's output at chunk `j` summarizes a complete chunk and predicts chunk `j+1`. Byte `t` reads the latest chunk that ends at or before `t`. So the last byte of chunk `j` (which predicts the first byte of chunk `j+1`) reads main output `j`, the same layout as a tokenized LM. With this rule, stage-1 chunks are made of complete stage-0 chunks, which explains why it would help multi-stage hierarchies.
3. **"Strong auxiliary loss"** means the 2025 ratio loss with a larger weight, or a stricter penalty. It is not specified.

Experiments that test these live in [lab/experiments/](../../lab/experiments/).
