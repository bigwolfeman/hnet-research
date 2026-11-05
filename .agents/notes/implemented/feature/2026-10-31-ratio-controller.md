# Agent Note: Ratio controller for matched-compression arms

Status: implemented

## Problem

The 2025 H-Net recipe adds `0.03 * ratio_loss` per stage. The LM loss pulls against it, so the measured ratio settles away from the target. The size of the drift depends on the architecture and on training time. With the `lookahead` router (target 4x), stage 0 settles at 3.0-3.1x in the 2-stage baseline, but at 3.8-3.9x when stage 1 has no blocks. The ratio sets the inner stages' sequence length, so arms that differ in ratio also differ in FLOPs/byte. In the [deep-main compute split](../../../../lab/experiments/refuted/2026-10-31-deep-main-compute-split.md) run 1, that drift was 14% of FLOPs, and two gates failed. The router experiment had the same problem (+10% FLOPs for `lookahead`).

Stage 0 always wants finer chunks than 4x. A controller must also hold a stage whose pull is near zero or changes sign, so it must not remove the fixed ratio loss. In the [compute ladder](../../../../lab/experiments/refuted/2026-11-01-compute-ladder-vs-bpe.md) attempt 1, the first controller drove the stage-1 weight to 0, and then stage 1 drifted from 2x to 4.8x. In attempt 2, with the fixed weight kept, the stage-1 integral term stayed at 0.000 ± 0.001 for 192,000 steps and the ratio held at 2.00. So the LM's pull on stage 1 is near zero. The drift came from removing the P term, not from a strong pull toward coarser chunks. I did not find why the old weight fell while the logged ratio was still at or below 2.0.

## Decision

`RatioController` in `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/chunking.py` adds a signed integral term per stage. The fixed-weight 2025 ratio loss stays as the proportional term (Cartesia's recipe is unchanged when the controller is off):

```
loss  = CE + sum_s (0.03 * ratio_loss_s + lam_s * G_s)    # G_s = mean boundary prob (with grad)
lam_s += lr * (N_s * F_s - 1)                            # F_s = anchor fraction (hard), N_s = target ratio
```

- The ratio loss's own gradient on G is `w N/(N-1) (N F - 1)`: a P controller, so it leaves an offset. The integral term removes the offset in both directions. Chunks too small (N F > 1): lam rises and pushes G down. Chunks too big: lam goes negative and pushes G up.
- Hydra key `train.ratio_integral_lr` (0 = off). Arm comparisons where the treatment can move the router use 0.002. Calibration on the blog split, seed 0, steps 500-4,400, train-mix stage-1 ratio (target 2): lr 0.01 gave 2.018 +- 0.112 with lag-50 autocorrelation +0.39, which is a slow wander that the integrator drives. lr 0.002 gave 2.005 +- 0.081 with autocorrelation +0.06. The old weight controller gave 1.999 +- 0.085. Stage 0: 4.006 +- 0.102 (old: 3.895). At a 10% ratio error, lr 0.002 moves lam by about 0.02 in 100 steps. The drift in ladder attempt 1 needed a force of that size and built over thousands of steps. The update stays on the device and adds no host sync. wandb logs `train/ratio_lambda_s{s}`.
- Resume checkpoints and WSD ladder branches save and restore `lam` (`state_dict` / `load_state_dict`).

## Alternatives considered

- **Dual ascent on the ratio-loss weight**, `log w_s += lr (N F - 1)` (the first design, used in deep-main run 2 and the first ladder attempt). Rejected after it failed: it is one-sided. It can lower w to zero, and then nothing holds the ratio. Measured in ladder attempt 1 (wandb `olotmyw6`): `ratio_weight_s1` fell from 0.03 to 0.000 by step 38,000, and the stage-1 ratio drifted from 2.0 to 4.8 (train mix), with edu FLOPs/byte down 17%. The same design held stage 0, where the pull is always toward finer chunks. It held both stages for 12,000 steps in deep-main run 2, so that run's gates hold.
- Tune `target_ratio` per arm until the measured ratios match. Rejected: this takes trial runs for every arm and seed, and the setting goes stale when anything else changes (the stage-1 pull reverses mid-run).
- Raise the fixed weight (for example 0.3) for every arm. Rejected: a stronger P term still leaves an architecture-dependent offset. It only makes the offset smaller.
- Match FLOPs by resizing the main network at the measured ratio. Rejected: the stage-0 chunk size would still differ, so that is a second confound.
- Accept the drift as an effect of the treatment and grade anyway. Rejected: the gate exists because a ratio change alone moves bpb. In run 1, the bpb curves crossed at the time the ratios separated.

## Consequences

- A controlled run trains at the configured ratio, not at the LM's preferred ratio. Its bpb is therefore not comparable with fixed-weight runs at a different ratio. Compare controlled with controlled.
- `lam` is a free variable. Its sign shows which way the LM pulls (positive: the LM wants finer chunks). If a router cannot reach its target, |lam| grows without bound and the ratio gate fails loudly. Read `train/ratio_lambda_s*` when a ratio gate fails.
- Measured over a full 192,000-step run (compute ladder attempt 2): blog split train-mix ratios 4.00-4.02 x 1.99-2.01 in every window; lam_s0 peaks at 0.011, lam_s1 stays at 0.000. Matched 1-stage: 4.00 ± 0.11, lam_s0 peaks at 0.041. Every rung gate held.
- Tests (`tests/test_ratio_controller.py`): update direction and fixed point; the loss term; and a toy plant where an LM term with graded per-position preferences pulls the ratio coarser or finer. In all four cases (targets 2x and 4x, pull either way) the ratio converges to target within 1%. A *uniform* pull is a bang-bang plant under Adam and limit-cycles. That is a property of the toy, not of a real LM.
