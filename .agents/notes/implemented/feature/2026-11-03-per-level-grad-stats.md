# Agent Note: per-level gradient statistics

Status: implemented

## Problem

The stage-1 fate (live or dead at 786M bytes) is a lottery in both our code and the reference code. Two readings remain: the main network gets a weak or noisy learning signal, or it learns and the decoders do not read it. Our runs logged only the global gradient norm, which cannot tell the two apart. Under AdamW the gradient scale of a level says little about its update size. What matters is whether the gradient keeps its direction over steps.

## Decision

`hnet/gradstats.py` logs two numbers per level (0 = stage 0 + embeddings + head, 1 = stage 1, 2 = main network) every `log_every` steps when `train.grad_stats` is true (the default in all train configs):

- `train/grad/norm_l{L}`: the L2 norm of the level's gradient before clipping.
- `train/grad/update_rms_l{L}`: the rms of the bias-corrected AdamW direction m_hat / (sqrt(v_hat) + eps), read from the optimizer state after the step. It is near 1 for a gradient that keeps its sign and low for zero-mean noise.

Levels come from the optimizer's group names (`stage{L}_*`), so the same code serves our model, the reference adapter, and the JEPA projector group. The statistics read tensors only: `tests/test_gradstats.py` checks bit-equal parameters with and without them, plus the two limits of the update rms. First use: [main gradient coherence](../../../../lab/experiments/refuted/2026-11-03-main-gradient-coherence.md).

## Alternatives considered

- **Gradient norm per level only.** Cheaper, but AdamW divides it out. A level with a tiny gradient still moves at the full LR, so the norm alone does not say whether the level learns.
- **Cosine between successive gradients.** Measures consistency directly, but needs a copy of the previous gradient per level (the main network alone is about 31M parameters) and reads one step pair, which is noisy. The AdamW state already holds the EMAs, for free.
- **Gradient at the decoder input (the dechunked inner signal against the residual).** Says what the decoder reads, but it needs a tensor hook inside the compiled scatter function. Left for a checkpoint-level diagnostic (`s1diag.py` in the coarse JEPA results) instead of every training step.

## Consequences

- Two extra host syncs per logged step (every 50 steps). No measurable change in step time in the smokes.
- The noise floor sqrt((1 − b1)/(1 + b1)) assumes a constant gradient scale. Early in training the update rms reads below it, so the numbers are for comparison across levels and seeds, not against the floor.
- The 200-step smokes for this change showed that the trainer is not bit-repeatable on the GPU (two identical runs end 0.0024 bpb apart). A seed does not fix a run's outcome.
