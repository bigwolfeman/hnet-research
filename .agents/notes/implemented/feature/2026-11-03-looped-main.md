# Agent Note: looped main network (Main.loops)

Status: implemented

## Problem

Phase 2 of the [constitution](../../../../docs/constitution.md) loops the coarsest network: the same weights run R times over the short coarse sequence. The spike's `Main` ran its stack once, so the bet had no code to test. On 2026-11-03 the user asked for the looped test now, on the 1-stage H-Net, as one of four ladders ([looped coarse main](../../../../lab/experiments/refuted/2026-11-03-looped-coarse-main.md)).

## Decision

`model.main.loops` (default 1) sets R. `Main.forward` (`lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/stage.py`) runs `h = stack(h)` R times with the same weights. Each pass ends in the stack's final RMSNorm, so every pass starts from a normalized stream. The init scales output projections by 1/sqrt(n_res + R x height), so a T2 x 4 main starts at the scale of a T8 main. `hnet/flops.py` counts R x the stack's FLOPs. Parameter counts do not change with R. `describe()` logs `loops`, so wandb records R in every run. The reference H-Net adapter (`kind: ref_hnet`) rejects loops other than 1. Tests: `tests/test_loops.py` (CPU), and `test_compile_matches_eager_looped_main` in `tests/test_gpu.py`.

## Alternatives considered

- **Input injection (add the main input to the stream at every pass, as in recurrent-depth LMs).** It helps loops keep the input in view. I declined it with the claim that at R up to 4 the residual stream keeps the input. The text run refuted that claim (see Consequences).
- **No final norm between passes (one norm at the end of all R passes).** The residual stream would grow over passes, and the init rule would not hold per pass. The stack already has the norm, so reuse it.
- **A separate LoopedMain class.** It would copy Main's interface for one integer. One class with `loops` keeps one code path (R = 1 is the old Main).

## Consequences

- R = 1 is the old model: the same keys, the same init, the same FLOPs. The full config of every run now has `model.main.loops`.
- Backward stores activations for every pass, so activation memory of the main network grows with R.
- A looped main network has fewer parameters at the same FLOPs. Comparisons at matched FLOPs and at matched parameters answer different questions.
- On text (1-stage, 786M bytes, 3 seeds per arm) the loop without input injection erases its input. At R = 2 and R = 4 the main output is the same vector for every row by the last pass (shuffle cost 0.0000), and R = 2 loses to R = 1 in every seed ([looped coarse main](../../../../lab/experiments/refuted/2026-11-03-looped-coarse-main.md)). `loops > 1` without input injection is not a test of looping. Do not use it for the Phase 2 question.
- The looped main is now the Parcae port: [Parcae main](./2026-11-04-parcae-main.md).
