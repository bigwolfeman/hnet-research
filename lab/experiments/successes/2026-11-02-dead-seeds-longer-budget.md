# Experiment: dead seeds, longer budget

Status: success

Date proposed: 2026-11-02
Algorithm: none (three existing seeds at a 4x budget; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Are our code's dead seeds slow or stuck? At 786M bytes, our code with the reference configuration ends with the stage-0 signature below zero in seeds 1, 3, and 4 ([main-network seed rate](../refuted/2026-11-02-main-network-seed-rate.md)). With 4x the training bytes, does it come online in those seeds?

Context: in that experiment, edu bpb followed the stage-0 sign (shuffle − zero of the dechunked inner output at stage 0) in all 10 runs of both codes. The sign starts negative in every run and climbs. The runs that end negative are still climbing when the LR decay starts at step 9,600. The reference code at 3.15B bytes (seed 0) tied BPE on edu and beat it on Go and SQL ([reference H-Net at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md)).

## Hypothesis

The inner network's content becomes useful to the stage-0 decoder at a speed that varies by seed. The 786M-byte budget ends before the slow seeds get there. It is not a stuck state that needs a design change. With the 3.15B-byte budget, every seed of our code reads the inner network's content at stage 0.

## Predictions

Main prediction (only this one answers the Question):

- Our code (`reflayout_2stage`) has the stage-0 shuffle − zero above zero at the end of the 3.15B-byte run (step 48,000) in all three seeds, 1, 3, and 4. All three: supported. Any at or below zero: not supported.

Other predictions (at most two):

- Each of the three seeds' edu bpb at 3.15B bytes ties the reference code's 1.1708 (seed 0, the same budget) or is below it. "Ties" means inside the 0.059 seed spread used in that experiment.
- The stage-1 shuffle − zero is above zero at step 48,000 in all three seeds.

If supported, we do: the 786M-byte budget is too short to judge a seed, and our code needs no design change to bring the hierarchy online. Then pick our 2-stage design (the reference configuration in our code) and rerun the BPE comparison at 3.15B bytes and more, with several seeds. Speed (our code is about 8% faster than the reference code) stays a measurement.

If not supported, we do: some seeds stay stuck at 4x the budget. Bring the design options to the user before building one: a longer stable phase, a main-network warm start, a larger main-network FLOPs share, or an auxiliary loss that rewards use of the inner network's content. The stuck seeds become the test set for the fix.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`: 147 passed, 1 skipped).
- Edu ratios within 15% of 3 x 3 at the 786M rung and at the end; edu fwd FLOPs/byte within 5% of 22.27M. Training bytes 3,145,728,000 per arm. Finite loss.
- The 786M rung (step 12,000, branched at step 9,600) has the stage-0 sign below zero in each seed, as the 12,000-step runs did. If a rung is above zero, the long run is not the same training as the short one up to the branch, and the comparison to the 786M result does not hold.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file (same model code as 8fbd1ba)
arms: our code, reflayout_2stage (faithful init, rope_dim 32), seeds 1, 3, 4
budget: max_steps 48,000 (3.146B bytes), ladder_fracs [0.25] -> rung at 12,000 steps (786.4M bytes), branched off
        the stable phase at step 9,600, the same decay as the 12,000-step runs
recipe: as the 12,000-step runs (configs/train/ladder.yaml, RatioController 0.002 from the experiment config,
        WSD linear decay over the last 20%); the stable LR runs to step 38,400
every eval (2,000 steps, stable LR): val bpb per domain, ratios, FLOPs, coarse-path probe (8 edu batches)
grade: probe/coarse_shuffle_s0 - probe/coarse_zero_s0 at step 48,000
control: the reference code, seed 0, same budget, schedule, and rung (at-our-scale rerun, wandb group
         ref-hnet-at-our-scale-r2); not rerun
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `TRAIN=ladder scripts/run_arms.sh ignored/runs/dead-seeds-longer-budget dead-seeds-longer-budget reflayout_2stage:1 reflayout_2stage:3 reflayout_2stage:4 -- train.max_steps=48000 "train.ladder_fracs=[0.25]"`
- wandb project `hnet-router`, group `dead-seeds-longer-budget`.
- Cost: about 95 ms/step, about 1.4 h per arm, about 4.3 h in total.
- The stable-phase evals from step 10,000 to step 38,000 also show whether the stage-0 sign keeps climbing at the stable LR. That is a reading, not a prediction.

### Amendment 2026-11-02 15:27 (seed 4 stopped, resumed)

- Seed 4's process stopped at step 42,600 (15:23) with no traceback. The user stopped it: a compilation started on the CPU with the GPU at full load, over the power budget. The training logs show no mid-run recompile, so the compile was not from this run. `run_arms.sh` logged it as "FAILED (exit 0)": the `$(date)` in the log line reset `$?` before the line read it. The script now saves the exit code first and names the signal for a code above 128.
- Fix: the same launch command again. Seeds 1 and 3 skip (summary.json). Seed 4 resumes from `resume.pt` at step 40,000, with model, AdamW, controller, loader RNG, and wandb id (`5ydcoweh`) restored. Steps 40,001 to 42,600 ran twice; the logged curve is the second pass.
- What I saw before this amendment: the final results of seeds 1 and 3 (stage-0 sign +0.086 and +0.200; edu bpb 1.2111 and 1.1852), and seed 4's stable-phase train loss. The Predictions do not change.

## Results

All three arms ran to step 48,000 (2026-11-02 11:00 to 15:44; commit 7db5d7a for the model code). Log dir `ignored/runs/dead-seeds-longer-budget`. wandb: seed 1 `3glrxsv3`, seed 3 `v8l7hqsl`, seed 4 `5ydcoweh`. Artifacts: [results/](../results/2026-11-02-dead-seeds-longer-budget/) (`*-summary.json`, `*-ladder_12000.json`, `eval-curves.csv`).

### Gates

| gate | seed 1 | seed 3 | seed 4 | result |
|---|---|---|---|---|
| our suite (`HNET_TEST_CUDA=1`) | 147 passed, 1 skipped (run at 11:00 for this model code; the later `run_arms.sh` change does not touch training) | | | pass |
| edu ratios, 786M rung / end (3 x 3 ±15%) | 3.07 x 3.02 / 3.11 x 2.83 | 3.01 x 2.95 / 3.12 x 2.80 | 3.05 x 2.95 / 3.15 x 2.84 | pass |
| edu fwd FLOPs/byte, 786M rung / end (22.27M ±5%) | −0.4% / +1.4% | +1.8% / +1.6% | +1.1% / +0.6% | pass |
| 48,000 x 65,536 = 3,145,728,000 bytes; finite loss | yes | yes | yes | pass |
| 786M rung has the stage-0 sign below zero | −0.412 | −0.219 | −0.163 | pass |

At the 786M rung, edu bpb is 1.3651, 1.3488, and 1.3524. The 12,000-step runs of the same seeds got 1.3536, 1.3522, and 1.3573. The two are the same training up to the branch, and they differ by up to 0.012. That is bf16 and compile-order noise; the sign agrees in every seed.

### At the end (3.15B bytes, step 48,000)

| run | stage-0 shuffle − zero | stage-1 shuffle − zero | edu bpb | Go bpb | SQL bpb |
|---|---:|---:|---:|---:|---:|
| **ours, seed 1** | **+0.086** | **−0.120** | **1.2111** | 0.690 | 1.440 |
| **ours, seed 3** | **+0.200** | **−0.045** | **1.1852** | 0.657 | 1.329 |
| **ours, seed 4** | **+0.052** | **+0.026** | **1.2015** | 0.660 | 1.355 |
| reference code, seed 0 (control) | +0.319 | +0.139 | 1.1708 | 0.653 | 1.297 |
| BPE Transformer (control) | | | 1.1769 | 0.732 | 1.460 |

Stage-0 shuffle − zero by eval (stable LR to step 38,400, then the decay):

| seed | 2k | 6k | 10k | 14k | 18k | 22k | 26k | 30k | 34k | 38k | 42k | 46k | 48k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | −0.431 | −0.758 | −0.367 | −0.111 | +0.039 | +0.073 | +0.115 | +0.126 | +0.157 | +0.228 | +0.194 | +0.134 | +0.086 |
| 3 | −0.464 | −0.446 | −0.135 | −0.049 | −0.019 | +0.012 | +0.036 | +0.058 | +0.061 | +0.211 | +0.210 | +0.215 | +0.200 |
| 4 | −0.423 | −0.260 | −0.062 | +0.000 | −0.022 | +0.024 | +0.039 | +0.107 | +0.140 | +0.152 | +0.114 | +0.073 | +0.052 |

- Main prediction (stage-0 sign above zero at step 48,000 in seeds 1, 3, and 4): +0.086, +0.200, +0.052. **Supported.**
- Other prediction 1 (edu bpb ties the reference's 1.1708 or is below it, inside 0.059): holds. The gaps are +0.040, +0.014, and +0.031. All three also tie BPE (+0.034, +0.008, +0.025).
- Other prediction 2 (stage-1 sign above zero at the end in all three): does not hold. Only seed 4 is above zero (+0.026); seeds 1 and 3 are at −0.120 and −0.045.

### What the runs show (observations after the fact)

- **The dead seeds were slow.** At the stable LR, the stage-0 sign climbs in every seed and first crosses zero between steps 14,000 and 20,000 (0.9B to 1.3B bytes). Seed 3 dips back below zero at steps 24,000 and 28,000 and stays above from step 30,000. Seeds that come alive at 786M bytes cross between steps 4,000 and 8,000. The 786M-byte budget ends inside the slow seeds' climb.
- **The LR decay lowers the stage-0 sign.** From the decay start (step 38,400) to the end, it falls in seeds 1 and 4 (+0.228 → +0.086, +0.152 → +0.052) and holds in seed 3. The 12,000-step runs showed the same fall. The decay improves bpb (−0.11 over the last 9,600 steps) and still lowers the stage-0 decoder's reliance on inner content.
- **Our code is still behind the reference at stage 1.** At the same budget, the reference's seed 0 has stage-1 sign +0.139 and `shuffle_s1` 0.460. Our three seeds reach `shuffle_s1` 0.12 to 0.29, and two have the sign below zero. The inner hierarchy is in use. The main network's content is not yet read by the stage-1 decoder in most seeds.
- **On code, our model beats BPE at 3.15B bytes in all three seeds:** Go 0.657 to 0.690 against 0.732, SQL 1.329 to 1.440 against 1.460.

## Verdict

**Supported.** All three of our code's dead seeds have the stage-0 sign above zero at 3.15B bytes (+0.086, +0.200, +0.052). Their edu bpb ties the reference code and BPE.

## Updated hypothesis

In our code, with the reference configuration, the inner hierarchy comes online in every seed tested. The speed varies by seed, and 786M bytes is too short to judge a seed. Stage 0 needs no design change. Stage 1 is the open part: the main network's content reaches the stage-1 decoder in one of three seeds at 3.15B bytes, and in the reference code's one seed. The same slow-start pattern may hold one level up. The next run trains our design to the 12.58B-byte top rung of the BPE comparison with two seeds. It answers the Phase 1 question (does our 2-stage H-Net beat BPE) and shows whether stage 1 comes online with more bytes: [reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md).

## Related

- [Main-network seed rate](../refuted/2026-11-02-main-network-seed-rate.md)
- [Reference H-Net at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md)
- [Faithful init and RoPE](../refuted/2026-11-02-faithful-init-and-rope.md)
