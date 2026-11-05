# Deep-main compute split, run 1 (fixed ratio-loss weight): two gates failed, not graded

Experiment: [deep-main compute split](../../refuted/2026-10-31-deep-main-compute-split.md). Run 2 repeats the arms with the ratio controller.

Per-run numbers, wandb links and probe time courses: [run1-summaries.json](run1-summaries.json). wandb group `deep-main-compute-split`.

| Arm | edu bpb (s0 / s1 / s2) | edu ratio s0 x s1 | Fwd FLOPs/byte | shuffle_s1 at end | shuffle_s0 at end |
|---|---|---|---|---|---|
| baseline split | 1.410 / 1.412 / 1.469 | 3.0-3.1 x 2.0 | 23.5-24.2M | +0.088 / +0.022 / +0.003 | +1.44 / +1.43 / +0.84 |
| deep-main split | 1.556 / 1.558 / 1.564 | 3.8-3.9 x 2.0 | 19.9-20.3M | +0.059 / +0.048 / +0.017 | +0.28 / +0.26 / +0.25 |

Gates: the FLOPs gate failed (deep-main -14%), and the ratio gate failed (stage 0 +22%). Loss was finite, and the training bytes were equal. Deep-main is worse by 0.09 to 0.15 bpb, but it also had 14% less compute and coarser stage-0 chunks. So run 1 does not answer the Question, and the verdict waits for run 2.

Diagnostics from run 1 (wandb time courses, not graded):

- The main network is not "used, then dropped". In all 6 runs, `coarse_shuffle_s1` starts near 0 (0.000 to 0.009 at step 500) and grows for the whole run. In baseline seed 0 it goes 0.035 at step 6000, 0.057 at 9000, and 0.088 at 12000, and it still rises at the end. The main network is a slow learner, not a locked-out one.
- Deep-main uses the coarse path less over time. `coarse_shuffle_s0` falls from 0.39-0.55 at step 500 to 0.25-0.28 at the end. In the baseline it rises from 0.63-0.83 to 0.84-1.44. With no stage-1 blocks, the stage-0 decoder learns to predict locally.
- The bpb curves cross. Deep-main is level with or ahead of the baseline up to step 2000 (seed 1 at step 1000: 1.966 vs 2.116). It falls behind after step 3000, while its stage-0 ratio climbs from 3.0 to 3.8.
- `coarse_zero_s1` (main output set to 0) costs 0.22-0.38 in the baseline, against 0.003-0.088 for shuffle. The main output acts mostly as a constant bias. Its document-specific part is small.
