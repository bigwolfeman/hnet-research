# Agent Note: Parcae looped main network (main.kind parcae)

Status: implemented

## Problem

Phase 2 of the [constitution](../../../../docs/constitution.md) loops the coarsest network. The first loop, `Main.loops` ([looped main](./2026-11-03-looped-main.md)), fed each pass only the previous pass's normalized output. On text it erased its input: at R = 2 and R = 4 the main output was one vector for every row ([looped coarse main](../../../../lab/experiments/refuted/2026-11-03-looped-coarse-main.md)). On 2026-11-04 the user asked to borrow the Parcae implementation: prelude, core loop, coda.

## Decision

`lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/parcae.py` ports Parcae (Prairie et al. 2026, arXiv 2604.12946; github.com/SandyResearch/parcae at 4b86c2a, MIT). `ParcaeMain` subclasses `Main` and is selected by `model.main.kind: parcae` with a `main.parcae` block (prelude, core, coda, mean_recurrence, backprop_depth, decay_init, seed); `main.layers` must be null and `main.loops` 1.

- Prelude: a Stack ending in RMSNorm (Parcae `prelude_norm`). Coda: a Stack ending in RMSNorm (Parcae `ln_f`). Core: a Stack with no final norm (`Stack(final_norm=False)`); Parcae has no norm between iterations.
- Injection (Parcae `DiagonalInjection`): `x <- core(decay * x + dt * B e)`, `decay = exp(-softplus(dt_bias) exp(A_log))` in (0, 1). Init: A_log 0, decay sqrt(1/5), B identity. A_log, dt_bias, B skip weight decay (`_no_weight_decay`, read by `make_optimizer`). `dt * B e` is computed once per forward.
- State: truncated normal at the init std (Parcae `like-init`), from a fixed generator in eval mode so the coarse-path probe compares runs with one state.
- Depth (Parcae `poisson-truncated-full`, `per-sequence`): per row, total T ~ max(1, Poisson(R)); the last min(T, s) steps carry gradient, the rest run under no_grad with the autocast weight cache off (Parcae `_no_grad_prefix`). Eval runs R steps. The depth draws come from a generator seeded by `parcae.seed` and a draw counter kept as a buffer, so a resume continues the same sequence.
- `Main` gains `d` and `compile_blocks` so the model, the trainer, and the JEPA hook do not reach into `.stack`. FLOPs count prelude + R x core + coda block passes plus the injection and C (`hnet/flops.py main_block_passes`). The trainer logs `train/parcae/{depth_mean, depth_max, decay_max, decay_mean}`.
- Tests: `tests/test_parcae.py` (init, eval steps, per-row depth and truncated grad, sampler, FLOPs and params, weight decay, config errors); `tests/test_gpu.py` (compile parity of the Parcae main at eval; core grads with a no-grad prefix under CUDA autocast).

## Alternatives considered

- **Additive injection (x + e) or concat + linear (Huginn style).** Both are in Parcae's code. The paper reports concat as unstable and add as marginally stable, and picks the diagonal map. We take the paper's choice.
- **Fix `Main.loops` in place (add e at every pass).** That is a new design without the stability constraint, the depth sampling, or the truncated backprop, and it would not be the implementation the user asked for. Parcae's whole loop is the tested unit.
- **Fixed depth R in training.** Simpler and deterministic, but Parcae's sampled depth is part of what keeps the loop stable and lets eval depth vary. Kept as in Parcae.
- **Parcae's blocks, init (scaled-zero), value embeddings, and Muon.** They would make the main network differ from the R = 1 control in five ways at once. The port keeps our blocks, init rule, and AdamW, so the comparison is the loop structure.

## Consequences

- A Parcae main network has fewer parameters than a plain main network at the same eval FLOPs (prelude + coda + one core, not R cores). Training FLOPs are lower than 3 x forward because the no-grad steps have no backward. Wall time is higher than the FLOPs say: a batch runs to its deepest row.
- Train-mode forwards draw a random state and depth. Any eval must call `model.eval()` (the trainer and the probes do).
- `Main.loops` > 1 stays in the code for the record of the refuted run. Do not use it for the Phase 2 question.
- On torch 2.14.1, CUDA autocast does not reuse no-grad weight casts, so the no-grad-prefix guard is not exercised by the test; it stays as in Parcae.
- On text (1-stage, 786M bytes, AdamW), the Parcae main network carries document content in every seed (shuffle cost 0.055-0.90), where the naive loop carried none. It loses to the unlooped main at matched forward FLOPs (R = 2 mean 1.553, R = 6 mean 1.537, R = 1 mean 1.483), and R = 6 matches R = 2 at 41% fewer parameters ([Parcae coarse main](../../../../lab/experiments/refuted/2026-11-04-parcae-coarse-main.md)). It is a working looped main, not yet a better one.
