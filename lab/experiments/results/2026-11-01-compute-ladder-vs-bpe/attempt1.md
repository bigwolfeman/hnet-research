# Compute ladder vs BPE: attempt 1 (broken, gates failed)

Plan: [2026-11-01-compute-ladder-vs-bpe](../../refuted/2026-11-01-compute-ladder-vs-bpe.md), amendment 2026-11-01 03:00.

- Arm 1 only (blog split 2-stage, seed 0), wandb `olotmyw6` (state: crashed). No BPE or 1-stage data exists from this attempt.
- Launched 01:20 CDT. A power loss stopped it at 03:32 CDT (02:32 CST; the clocks fell back at 02:00), at step 141,550 of 192,000. Speed was 45 ms/step (1.45 MB/s) after the first 10,000 steps.
- Cause of the gate failures: the first `RatioController` design (dual ascent on the ratio-loss weight) is one-sided. Detail: [ratio controller note](../../../../.agents/notes/implemented/feature/2026-10-31-ratio-controller.md).

## Controller trace (train mix, wandb)

| step | ratio s0 | ratio s1 | weight s0 | weight s1 |
|---:|---:|---:|---:|---:|
| 3,500 | 4.05 | 1.97 | 0.47 | 0.026 |
| 14,000 | 3.83 | 2.00 | 1.11 | 0.021 |
| 28,000 | 4.10 | 1.96 | 2.15 | 0.006 |
| 38,500 | 3.99 | 2.56 | 2.98 | 0.000 |
| 66,500 | 4.02 | 3.27 | 5.37 | 0.000 |
| 105,000 | 3.93 | 4.52 | 8.32 | 0.000 |
| 141,550 | 3.94 | 4.85 | 11.42 | 0.000 |

## Rungs (edu val)

| rung | bytes | bpb | ratio s0 | ratio s1 | fwd FLOPs/byte | vs BPE 22.27M | gates |
|---|---:|---:|---:|---:|---:|---:|---|
| b12000 | 0.79B | 1.5092 | 3.99 | 1.98 | 22.61M | +1.5% | hold |
| b48000 | 3.15B | 1.3433 | 4.00 | 2.44 | 20.28M | −8.9% | ratio s1 +22% (gate 15%), FLOPs (gate 5%): fail |

Raw rung metrics: [attempt1-ladder_12000.json](attempt1-ladder_12000.json), [attempt1-ladder_48000.json](attempt1-ladder_48000.json). The main run's edu FLOPs/byte fell to 17.7-18.5M (−17 to −20%) from step 54,000.
