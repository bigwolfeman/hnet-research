# Experiment: deep-main compute split

Status: refuted

Date proposed: 2026-10-31
Algorithm: none (a reallocation of existing blocks; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

In the 2-stage H-Net, does moving stage 1's encoder and decoder blocks into the main network (same FLOPs/byte) make the main network do work and lower validation bpb?

Context: in every 2-stage checkpoint of the [router experiment](../refuted/2026-10-30-end-of-chunk-router.md), the main network's output can be swapped for another document's at a cost of +0.001 to +0.043 bpb (the spike's [coarse-path ablation](../../spikes/2026-10-30-two-stage-hnet-smoke/README.md#coarse-path-ablation-diagnostic-no-prior-prediction)). The 1-stage main network is used (+1.0 bpb). The [controls](../refuted/2026-10-31-byte-and-bpe-controls.md) put BPE 0.15 bpb ahead.

## Hypothesis

The stage-1 encoder and decoder (2 + 2 blocks, d 384, window 256 chunks, about 800 bytes of context) do the coarse work locally. The stage-1 decoder learns to predict from its own residual and stops reading the main network. Cartesia's 2-stage model puts thin outer stages around a deep (40-layer) main network. If stage 1 has no blocks, the only coarse computation is the main network, so the decoder must read it, and the main network's global context and depth start to pay.

## Predictions

Main prediction (only this one answers the Question):

- At the end of training, deep-main split has lower edu validation bpb than baseline split, by more than the seed spread of the baseline split runs in this experiment.

Other predictions (at most two):

- At the end of training, the main-network shuffle cost (`probe/coarse_shuffle_s1`) of deep-main split is larger than that of baseline split, in every seed.
- In baseline split, `probe/coarse_shuffle_s1` is larger at some early eval than at the end of training (the main network is used first and then dropped), rather than near zero from the first eval.

If supported, we do: make the deep-main split the 2-stage default (thin outer stages, deep main, as in Cartesia), write an Agent Note, and rerun the BPE comparison with it. Then the ratio-matched router rerun.

If not supported, we do: the compute split is not the cause. Next test the gradient path into the main network (main LR multiplier, stage-1 residual dropout), choosing the first by the time course from the second other prediction.

Gates (the run is broken if one fails; gates are not predictions):

- Causality tests pass for the deep-main shape (stage 1 with 0 blocks): `tests/test_causality.py` covers it.
- Code equivalence: baseline split seed 0 lands within the router experiment's `lookahead` 2-stage range (1.414 to 1.446 edu bpb). The perf merge and torch 2.14 must not change the model's behavior.
- Fwd FLOPs/byte of the two arms within 3% at measured ratios. Measured ratio per stage within 15% between the two arms, and no collapse (1x or one chunk). The `lookahead` stage-0 ratio settles near 3.1x, not the 4x target. Both arms share that router, so the gate compares arms, not targets.
- Finite loss for the whole run. Same training bytes (786M).

## Method

```
baseline split  (lookahead_2stage):            stage0 enc/dec 2/2 d256 w512 | stage1 enc/dec 2/2 d384 w256 | main 8  x d512
deep-main split (lookahead_2stage_deep_main):  stage0 enc/dec 2/2 d256 w512 | stage1 enc/dec 0/0 (router + chunk/dechunk only) | main 12 x d512
    FLOPs/byte fwd at ratios 3.1 x 2.0: 23.57M vs 22.94M (-2.7%).  Params 36.6M vs 42.5M.
for seed in 0, 1, 2:
    train baseline split, then deep-main split   (train=router: 12,000 steps x 16 x 4096 bytes, lr 8e-4, mult [1.5, 1.2, 1.0])
every eval (500 steps):
    val bpb per domain
    probes.coarse_use on 8 edu eval batches: for stage s in {0, 1}, mode in {zero, shuffle}:
        probe/coarse_{mode}_s{s} = bpb(stage s `up` ablated) - bpb
```

- Code: `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, master after commit aa12788 plus the probe commit. Local RTX 5090, torch 2.14.1+cu130, compiled.
- Launch: `scripts/run_arms.sh ignored/runs/deep-main-compute-split deep-main-compute-split lookahead_2stage:0 lookahead_2stage_deep_main:0 lookahead_2stage:1 lookahead_2stage_deep_main:1 lookahead_2stage:2 lookahead_2stage_deep_main:2`. wandb project `hnet-router`, group `deep-main-compute-split`.
- The baseline is rerun, not reused, because the probe time course and the code-equivalence gate need it.
- Risk: with no stage-1 encoder, the stage-1 router sees stage-0 encoder states directly. Its ratio may sit away from 2x. The ratio gate catches that.

### Amendment 2026-10-31 (during arm 1, no arm-vs-arm data)

- Param count correction: baseline split (`lookahead` 2-stage) has 37.1M params, not 36.6M (36.6M is the `eoc` 2-stage count; `lookahead` adds q/k projections). Deep-main split has 42.5M. Nothing else changes.

### Amendment 2026-10-31 (after run 1, before run 2): ratio controller

Run 1 failed two gates. Its numbers and diagnostics are in [run1.md](../results/2026-10-31-deep-main-compute-split/run1.md). The treatment moves the router: deep-main's stage-0 ratio drifts to 3.8-3.9x, while the baseline stays at 3.0-3.1x (gate: within 15%). The fwd FLOPs/byte then differ by 14% (gate: within 3%). With a fixed ratio-loss weight, any change to the inner stages can move the outer ratio. A matched comparison must hold the ratio.

Fix, applied to both arms in run 2:

```
RatioController (hnet/chunking.py), train.ratio_dual_lr = 0.01:
    every step, per stage s:  log w_s += 0.01 * (F_s * N_s - 1)     # F_s = anchor fraction, N_s = target ratio
    loss = CE + sum_s w_s * ratio_loss_s                              # w_s starts at loss_weight (0.03)
```

- Both arms run at the configured targets (4.0 x 2.0). At those ratios the fwd FLOPs/byte are 20.11M (baseline) and 19.56M (deep-main), which is -2.8%.
- Calibration before launch: 2,500 steps per arm, seed 0, wandb off. Edu val ratios were 3.91 x 1.96 for baseline and 3.93 x 2.01 for deep-main. Deep-main's stage-1 ratio moves between 1.86 and 2.14 from eval to eval.
- Run 2 uses wandb group `deep-main-compute-split-r2`, the same 6 arms and seeds, and the same launch with `-- train.ratio_dual_lr=0.01`.
- The Predictions do not change. "Seed spread of the baseline split runs in this experiment" means run 2's baseline.
- The code-equivalence gate cannot apply to run 2, because the controller changes the baseline. Run 1 covers it: run-1 baseline seed 0 got 1.4104. That is 0.004 below the old range of 1.414 to 1.446, on the good side. Seeds 1 and 2 got 1.4121 and 1.4693, so run 1's own baseline spread (0.059) is wider than the old range. The forward pass of the merged code on the old seed-0 checkpoint gives 1.4139, the old number exactly. I take the 0.004 as training noise from the torch 2.14 kernels, not a change in the model.
- Run 2's first arm (baseline seed 0) started at 09:47 on code that this amendment's commit adds. The wandb `env.git` field for that arm shows the parent commit. Its config has `train.ratio_dual_lr: 0.01`.

## Results

**Not supported.** With both arms held at the target ratios, deep-main split is *worse* than baseline split in every seed. Paired by seed (deep-main minus baseline): +0.133 (s0), +0.121 (s1), +0.070 (s2), mean +0.108 bpb. The baseline seed spread is 0.059.

Run 2 (ratio controller, `train.ratio_dual_lr=0.01`), wandb group `deep-main-compute-split-r2`. Per-run numbers, time courses and wandb links: [run2-summaries.json](../results/2026-10-31-deep-main-compute-split/run2-summaries.json). Samples (seed 0, both arms): [samples-s0.md](../results/2026-10-31-deep-main-compute-split/samples-s0.md). Run 1 (fixed weight, gates failed, not graded): [run1.md](../results/2026-10-31-deep-main-compute-split/run1.md).

| Arm | edu bpb (s0 / s1 / s2) | Go bpb | edu ratio s0 x s1 | Fwd FLOPs/byte | shuffle_s1 at end | shuffle_s0 at end | MB/s |
|---|---|---|---|---|---|---|---|
| baseline split | 1.434 / 1.438 / 1.492 | 0.982 / 1.000 / 1.008 | 3.93-3.96 x 1.99-2.06 | 20.0-20.4M | +0.008 / +0.004 / +0.009 | +0.97 / +1.00 / +0.53 | 1.19-1.21 |
| deep-main split | 1.567 / 1.559 / 1.562 | 1.068 / 1.068 / 1.073 | 3.92-3.99 x 1.93-1.99 | 19.6-20.0M | +0.012 / +0.032 / +0.054 | +0.15 / +0.26 / +0.09 | 1.20-1.25 |

Gates (run 2): measured ratios match within 2% per stage, and none collapsed. Fwd FLOPs/byte are -1.6% for deep-main on the arm means, with a worst pair of -2.9% (s0); all are inside 3%. Loss is finite, and both arms saw the same 786M bytes. Causality tests pass for the deep-main shape. Code equivalence comes from run 1 (see the second amendment). All gates hold.

The controller raised the stage-0 weight from 0.03 to 0.9-1.6 in the baseline, and to 0.2-0.4 in deep-main. The baseline's LM loss pulls harder toward small chunks. At 4x, the baseline is +0.022 bpb (paired mean) worse than its run-1 self at 3.1x, with 14% fewer FLOPs.

Edu val bpb through training (seed 0):

| step | 1000 | 2000 | 3000 | 6000 | 9000 | end |
|---|---|---|---|---|---|---|
| baseline split | 2.064 | 1.801 | 1.734 | 1.602 | 1.538 | 1.434 |
| deep-main split | 2.077 | 1.802 | 1.749 | 1.676 | 1.645 | 1.567 |

The arms are level through step 2000. The gap opens after step 3000 and grows to the end, now at matched ratios.

### Other predictions

- Deep-main `coarse_shuffle_s1` larger in every seed: supported (0.012 vs 0.008, 0.032 vs 0.004, 0.054 vs 0.009). But every value is small. A main network that costs 0.01-0.05 bpb when it gets another document's output is still close to dead.
- Baseline `coarse_shuffle_s1` larger early than at the end: not supported. It is 0.001-0.005 at the first evals and 0.004-0.009 at the end. The main network is never used. It is not "used, then dropped".

## Verdict

**Not supported.** At matched ratios and FLOPs, moving stage 1's blocks into the main network costs +0.108 bpb (paired mean, 3 seeds), about 2x the baseline seed spread. Deep main does not wake up the main network. It also takes away the stage-1 blocks, which were the only coarse compute doing work.

As planned: the compute split is not the cause. Next, test the gradient path into the main network.

## Updated hypothesis

The stage-1 blocks were never stealing the main network's job. They do the useful coarse work: stage-0 shuffle costs 0.5-1.0 bpb in the baseline and only 0.09-0.26 without them. In both arms the main network carries almost no document-specific information (shuffle_s1 below 0.06). Its output acts as a learned constant: zeroing it costs 0.03-0.34, while shuffling it costs almost nothing. In both arms the main network is never used from the first eval on. So the failure happens at initialization and early training, not in a later competition. The 1-stage main network (+1.0 bpb when shuffled) gets its gradient through one dechunk (EMA + STE + residual). The 2-stage main network gets it through two, and it sees 512 positions per row.

Candidate causes, in test order:

1. Gradient into the main network is too small. Measure the per-level gradient norm per parameter over the first 2,000 steps, for 1-stage and 2-stage, then test a main LR multiplier.
2. The stage-1 residual path wins at once. Test stage-1 residual dropout, or zero the stage-1 residual early in training.

A cheap measurement comes first: per-level gradient norms and update/weight ratios, logged from step 1. That needs no new arm.

## Related

- [End-of-chunk router experiment](../refuted/2026-10-30-end-of-chunk-router.md)
- [Byte and BPE controls](../refuted/2026-10-31-byte-and-bpe-controls.md)
- [Constitution](../../../docs/constitution.md), Success criterion 1.
