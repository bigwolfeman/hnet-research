# Experiment: reference layout vs BPE

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (an existing arm on the existing compute ladder; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does our 2-stage H-Net, with the reference configuration in our code, beat the BPE Transformer on edu at the 12.58B-byte top rung of the compute ladder?

Context: the blog-split 2-stage lost to BPE there by 0.179 bpb, with its main network unused ([compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)). With the reference configuration (Mamba-2 outer stages, paper Eq. 11 LR, targets 3 x 3, the 2025 init, partial RoPE), our code brings the inner hierarchy online in every seed tested by 3.15B bytes, and ties BPE there on edu (+0.008 to +0.034) while beating it on Go and SQL ([dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md)). The main network's content reaches the stage-1 decoder in one of those three seeds.

## Hypothesis

The hierarchy pays more as training bytes grow. With the inner hierarchy online, the gap to BPE keeps falling past 3.15B bytes, and at 12.58B bytes our 2-stage H-Net is ahead of BPE on edu.

## Predictions

Main prediction (only this one answers the Question):

- At the 12.58B-byte top rung, the mean edu val bpb of our two seeds is lower than BPE's 1.1156. Grading uses our design's own seed spread at 3.15B bytes, 0.026 (seeds 1, 3, 4: 1.1852 to 1.2111), because BPE has one seed. Lower by more than 0.026: supported. Inside ±0.026: can't tell. Higher by more than 0.026: not supported.

Other predictions (at most two):

- At the top rung, Go and SQL bpb are below BPE's (0.654 and 1.337) in both seeds.
- At the top rung, the stage-1 shuffle − zero is above zero in both seeds (the main network's content reaches the stage-1 decoder).

If supported, we do: Phase 1's goal holds at this scale. The reference configuration becomes our 2-stage default (Agent Note). Phase 2 (the looped LM over the coarse level) builds on this model. If other prediction 2 fails, the main network is not yet read at stage 1, and Phase 2 starts with that problem: bring the stage-1 design options to the user first.

If not supported, we do: read the rung trend against BPE and against the matched 1-stage (1.1581 at the top rung). If the gap still shrinks at every rung, plan a larger budget or model. If it does not, the 2-stage design loses to the 1-stage at matched FLOPs, and we bring that to the user with the stage-1 design options.

If can't tell, we do: the next test adds seeds (and BPE seeds), or goes past 12.58B bytes, whichever is cheaper per unit of spread.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`: 147 passed, 1 skipped).
- Edu ratios within 15% of 3 x 3 at every rung; edu fwd FLOPs/byte within 5% of BPE's 22.27M at every rung. Training bytes 786,432,000, 3,145,728,000, and 12,582,912,000 at the rungs. Finite loss.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
arm: reflayout_2stage (our code: stage0 m4/m4 d256, stage1 T1m4/m4T1 d256, main T10 d512; targets 3 x 3,
     RatioController 0.002, lr 6.25e-4 x [3.03, 1.75, 0.72], 2025 init, rope_dim 32; 41.8M params, ~22.6M fwd FLOPs/byte)
seeds: 0 and 1 (the first two; seed 0 came online at 786M bytes, seed 1 was the slowest at 3.15B bytes)
budget: configs/train/ladder.yaml as is: max_steps 192,000 (12.58B bytes), ladder_fracs [1/16, 1/4]
        rungs at 12,000 and 48,000 steps branch off the stable phase at 0.8 b; the main run decays over its last 20%
every eval (2,000 steps): val bpb per domain, ratios, FLOPs, coarse-path probe (8 edu batches)
grade: mean over seeds of val/edu/bpb at step 192,000, against BPE 1.1156
controls (not rerun): BPE Transformer and matched 1-stage from compute-ladder-vs-bpe (same data, batch bytes,
                      schedule, and rungs); the reference code, seed 0, at the 3.15B rung (ref-hnet-at-our-scale-r2)
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `TRAIN=ladder scripts/run_arms.sh ignored/runs/reference-layout-vs-bpe reference-layout-vs-bpe reflayout_2stage:0 reflayout_2stage:1`
- wandb project `hnet-router`, group `reference-layout-vs-bpe`.
- Cost: 204,000 steps per seed (192,000 plus the 2,400- and 9,600-step branches) at about 112 ms/step with evals, about 7 h per seed, about 14 h in total.
### Amendment 2026-11-02 16:00 (stopped before any result)

- The user stopped seed 0 at step 6,250 (15:59, signal 9) to free the GPU for the synthetic recall and JEPA tests. `run_arms.sh` then started seed 1, and I stopped it at its start. No eval of either seed was read, and no rung was reached. Neither run dir has a `resume.pt` (the first is written at step 8,000), so a rerun starts from step 0 with the same command.
- The Predictions do not change. This plan waits in `planned/` until it is rerun.

### Amendment 2026-11-03 18:00 (rerun queued)

- The user asked for this run as the last of four ladders ("Lets do all 4, save A for last"). The queue `ignored/runs/next-ladder/chain.sh` runs it after grown hierarchy keeps main, looped coarse main, and thin outer stages, with the same launch command. The code adds `Main.loops` (default 1) and `train.grow_from` (default off); the reference layout's model, init, and FLOPs do not change. The suite gate now has 186+ tests, not 147.
- The Predictions do not change.
- 2026-11-04 01:00: the Parcae port lands while seed 0 runs, so seed 1 starts on the newer commit. For the plain `Main` the change is interface only (`Main.d`, `Main.compile_blocks`, `stage_params` over the same parameters, a `final_norm` flag that defaults on, a weight-decay opt-out no plain parameter sets). The model, init, optimizer groups, and FLOPs of the reference layout do not change; the CPU suite (186 passed) and the GPU suite gate both cover the plain path.

- Readings, not predictions: the H-Net − BPE gap at each rung, and the 3.15B rung of seed 1 against the same seed's 3.15B result in [dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md) (the same training up to step 38,400).

## Results

Chain `ignored/runs/next-ladder/chain.sh` (copy in results/), launch command as in Method. Seed 0 23:12 to 05:14 (code c5b2cae), seed 1 05:14 to 10:56 on 2026-11-04 (code 5e4cba7, see the amendment). wandb pvw7vrh2, tq1xn0ik. Artifacts in [results/](../results/2026-11-04-reference-layout-vs-bpe/): `summary.txt` and `val-curves.csv` (`pull.sh`), `gap.png` and `gap.txt` (`plot_gap.py`: both seeds and BPE on one bytes axis, and the gap at each eval).

### Gates

- Suite: 198 passed, 1 skipped (chain start, c5b2cae). Seed 1's commit passed 211 tests in the next chain's gate.
- Params 41,801,344. Bytes 786,432,000, 3,145,728,000, and 12,582,912,000 at the rungs. Finite loss.
- Edu ratios: seed 0 3.02 x 2.93, 3.18 x 2.70, 3.16 x 2.70; seed 1 3.05 x 2.99, 3.05 x 2.89, 3.16 x 2.64 (rungs 0.79B, 3.15B, 12.58B). The largest miss is −12% (seed 1 stage 1 at the top rung); the gate is 15%.
- Edu fwd FLOPs/byte: 22.69M, 22.68M, 22.75M (seed 0) and 22.40M, 22.68M, 22.96M (seed 1), +0.6% to +3.1% against 22.27M.

### Main prediction: mean edu val bpb at 12.58B bytes

| rung | seed 0 | seed 1 | mean | BPE | mean − BPE | matched 1-stage |
|---|---:|---:|---:|---:|---:|---:|
| 0.79B | 1.3185 | 1.3587 | 1.3386 | 1.2731 | +0.066 | 1.5532 |
| 3.15B | 1.1694 | 1.2253 | 1.1974 | 1.1769 | +0.020 | 1.2742 |
| **12.58B** | **1.0905** | **1.0979** | **1.0942** | **1.1156** | **−0.021** | 1.1581 |

The mean is 0.021 below BPE. The band is ±0.026, so the result is inside it.

### Other predictions

- Go and SQL below BPE (0.654, 1.337) in both seeds at the top rung: holds. Go 0.581 and 0.587, SQL 1.189 and 1.207.
- Stage-1 shuffle − zero above 0 in both seeds at the top rung: holds. +0.382 and +0.306 (shuffle cost 0.857 and 0.670, zero cost 0.475 and 0.364).

### Readings (planned, not predictions)

- **The H-Net − BPE gap per rung** falls in both seeds: +0.045, −0.008, −0.025 (seed 0) and +0.086, +0.048, −0.018 (seed 1).
- **Seed 1 at 3.15B** is 1.2253 here and 1.2111 in [dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md), +0.014, inside the 0.026 spread. The two runs are not the same past step 9,600: here the 0.79B branch takes 2,400 steps of batches from the stream first.

### Observations (not predictions)

- **The curves cross, then the gap stops growing.** On one bytes axis (`gap.png`), BPE is ahead early, by 0.07-0.20 bpb in the first 1B bytes. Seed 0 crosses BPE at 2.1B bytes, seed 1 at 6.8B. After the crossing each gap grows for a few billion bytes and then holds: seed 0 at −0.024 to −0.030 from 6B to 12.58B, seed 1 at −0.015 to −0.018 from 10B to 12.58B. In the LR decay (the last 2.5B bytes, the same schedule in both arms), BPE takes back up to 0.005 in seed 0. On this data the gaps do not widen near the end.
- **The seeds differ in when they cross, not in the slope after.** The seed that crossed late (seed 1, the slowest seed to come online at 3.15B) ends 0.007 behind the early one.
- **The 2-stage model beats the matched 1-stage by 0.060 and 0.068** at the top rung (one 1-stage seed). At 0.79B it was 0.19-0.23 ahead too, and 0.05-0.10 at 3.15B.
- **The main network is in use at stage 1 in both seeds.** The shuffle costs (0.86, 0.67) are 4-5x the best 786M-byte seed (0.158).

## Verdict

**Can't tell.** The mean edu val bpb of the two seeds at 12.58B bytes is 1.0942, 0.021 below BPE's 1.1156. That is inside the ±0.026 band from our seed spread. BPE has one seed, so its own spread is not measured.

## Updated hypothesis

The reference configuration in our code reaches BPE and passes it on edu, in both seeds, and beats it on code by 0.07 (Go) and 0.13-0.15 (SQL). The lead on edu is small, 0.018-0.025, and it does not grow over the last 4B bytes, so a longer run at this model size is not the cheap way to resolve it. The test was too small in its control: one BPE seed. The plan's can't-tell rule says to add seeds or train past 12.58B, whichever is cheaper. Adding seeds is cheaper (the train set holds about 14.5B bytes, only 15% more). The next test adds H-Net seed 2 and BPE seed 1, and grades an ordering (every H-Net seed below every BPE seed), which needs no band: [reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md).

## Related

- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
- [Dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md)
- [Reference H-Net at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md)
