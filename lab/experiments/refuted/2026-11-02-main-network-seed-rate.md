# Experiment: main-network seed rate

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (more seeds of two existing arms; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does the reference code bring the main network online for every init draw, or is it a lottery in the reference code too?

Context: with the reference configuration and a matched init distribution, our code was live in seed 0 and dead in seed 1 ([faithful init and RoPE](../refuted/2026-11-02-faithful-init-and-rope.md)). Over six runs of the reference layout in our code, 2 were live. The reference code was live in both of its seeds (0 and 1). Two seeds cannot separate "the reference code is more robust" from "the reference code drew two good inits".

## Hypothesis

A remaining code difference makes our code less robust: the fp32 residual adds inside the reference's fused norm (our blocks add residuals in the activation dtype under autocast). The reference code brings the main network online for every init draw at this configuration and budget.

## Predictions

Main prediction (only this one answers the Question):

- The reference code (`ref_hnet_2stage` with the RatioController) has the live stage-1 signature (`coarse_shuffle_s1` above `coarse_zero_s1` at 12,000 steps) in all three new seeds, 2, 3, and 4. All three: supported. Any at or below: not supported.

Other predictions (at most two):

- Our code (`reflayout_2stage`, faithful init and partial RoPE) is dead (shuffle at or below zero) in at least one of seeds 2, 3, and 4.
- Every live run, in either code, has a positive stage-1 shuffle − zero already at step 2,000, and every dead run a negative one (the fate shows by the first eval, as in F, G, and H).

If supported, we do: the reference code is robust (5 of 5 seeds) and ours is not. Find the last code difference: make our residual adds fp32 as in the reference's fused add-norm, then test it on the seeds where our code died.

If not supported, we do: the main network's fate is a lottery in both codes, at this budget. Code parity is done. The work becomes a design that brings the main network online for any draw. Candidates: a longer stable phase before decay, a main-network warm start, a larger share of FLOPs in the main network, or an auxiliary loss that rewards use of the main network's content. We take that to the user before building one.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (147 passed, 1 skipped); the reference venv's adapter tests pass (7).
- Edu ratios within 15% of 3 x 3 at the end; edu fwd FLOPs/byte within 5% of 22.27M. Bytes 786,432,000 per arm. Finite loss.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
arms, 12,000 steps each, no branches, RatioController 0.002, probe every 2,000 steps:
  reference code  ref_hnet_2stage   seeds 2, 3, 4   (reference venv; train.ratio_integral_lr=0.002 on the command line,
                                                     as in its seeds 0 and 1)
  our code        reflayout_2stage  seeds 2, 3, 4   (spike venv; faithful init, rope_dim 32)
seed s sets the init draw and the data order; the two codes draw different inits from the same seed
existing seeds (not rerun): reference code 0, 1 (live, live); our code 0, 1 (live, dead; faithful-init-and-rope)
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 12,000
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, two calls in sequence:
  - `PY=../2026-11-01-reference-hnet-probe/.venv/bin/python TRAIN=ladder scripts/run_arms.sh ignored/runs/main-network-seed-rate/ref main-network-seed-rate ref_hnet_2stage:2 ref_hnet_2stage:3 ref_hnet_2stage:4 -- train.max_steps=12000 "train.ladder_fracs=[]" train.ratio_integral_lr=0.002`
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/main-network-seed-rate/ours main-network-seed-rate reflayout_2stage:2 reflayout_2stage:3 reflayout_2stage:4 -- train.max_steps=12000 "train.ladder_fracs=[]"`
- wandb project `hnet-router`, group `main-network-seed-rate`.
- Cost: about 21 min per arm, about 2.1 h in total.

## Results

All six arms ran to 12,000 steps (2026-11-02 08:48 to 10:51, commit 8fbd1ba). Log dirs `ignored/runs/main-network-seed-rate/{ref,ours}`. wandb: reference code seed 2 `97q7mup2`, seed 3 `gu0lejtk`, seed 4 `24afwxy9`; our code seed 2 `s27d9kck`, seed 3 `hd9kqj1n`, seed 4 `21qy39rj`. Artifacts: [results/](../results/2026-11-02-main-network-seed-rate/) (`*-summary.json`, `eval-curves.csv`).

### Gates

| gate | result |
|---|---|
| our suite (`HNET_TEST_CUDA=1`): 147 passed, 1 skipped; reference venv `test_refmodel.py`: 7 passed | pass |
| edu ratios at the end (3 x 3 ±15%): 2.99 to 3.09 x 2.82 to 3.00 | pass |
| edu fwd FLOPs/byte (22.27M ±5%): reference code +1.8% to +2.3%, our code −0.4% to +2.4% | pass |
| 12,000 steps x 65,536 bytes = 786,432,000 bytes per arm; finite loss | pass |

### Every run of the reference configuration at 12,000 steps (seeds 0 and 1 from earlier experiments)

| code | seed | stage-1 shuffle − zero | stage-0 shuffle − zero | edu bpb | Go bpb | SQL bpb |
|---|---:|---:|---:|---:|---:|---:|
| reference | 0 | +0.021 | +0.124 | 1.3178 | 0.884 | 1.922 |
| reference | 1 | +0.032 | +0.028 | 1.3185 | 0.890 | 1.933 |
| reference | **2** | **−0.060** | **−0.026** | **1.3315** | 0.899 | 1.906 |
| reference | **3** | **+0.048** | **+0.107** | **1.3156** | 0.907 | 1.940 |
| reference | **4** | **−0.018** | **+0.200** | **1.3186** | 0.880 | 1.876 |
| ours | 0 | +0.023 | +0.125 | 1.3208 | 0.917 | 2.003 |
| ours | 1 | −0.127 | −0.304 | 1.3536 | 0.896 | 1.958 |
| ours | **2** | **+0.004** | **+0.206** | **1.3111** | 0.870 | 1.785 |
| ours | **3** | **−0.117** | **−0.329** | **1.3522** | 0.915 | 1.965 |
| ours | **4** | **−0.034** | **−0.198** | **1.3573** | 0.923 | 1.990 |

Bold rows are this experiment. Seeds 0 and 1: reference code from the [difference ablation](2026-11-01-reference-hnet-difference-ablation.md) (seed 0 is the 0.79B rung of the at-our-scale rerun); our code from [faithful init and RoPE](2026-11-02-faithful-init-and-rope.md).

Stage-1 shuffle − zero by eval (steps 2,000 to 12,000):

| run | 2k | 4k | 6k | 8k | 10k | 12k |
|---|---:|---:|---:|---:|---:|---:|
| reference seed 2 | +0.011 | −0.024 | −0.034 | −0.045 | −0.176 | −0.060 |
| reference seed 3 | +0.030 | +0.033 | +0.044 | +0.042 | +0.052 | +0.048 |
| reference seed 4 | −0.006 | −0.010 | −0.014 | −0.006 | −0.009 | −0.018 |
| ours seed 2 | +0.013 | +0.018 | +0.011 | +0.024 | +0.023 | +0.004 |
| ours seed 3 | −0.017 | −0.024 | −0.037 | −0.063 | −0.052 | −0.117 |
| ours seed 4 | +0.004 | +0.004 | −0.003 | −0.008 | −0.013 | −0.034 |

Stage-0 shuffle − zero by eval:

| run | 2k | 4k | 6k | 8k | 10k | 12k |
|---|---:|---:|---:|---:|---:|---:|
| reference seed 1 | −0.482 | −0.588 | −0.185 | +0.050 | +0.136 | +0.028 |
| reference seed 2 | −0.460 | −0.347 | −0.195 | −0.047 | +0.002 | −0.026 |
| reference seed 3 | −0.228 | −0.120 | +0.029 | +0.105 | +0.128 | +0.107 |
| reference seed 4 | −0.132 | +0.032 | +0.095 | +0.165 | +0.201 | +0.200 |
| ours seed 0 | −0.224 | −0.155 | +0.000 | +0.084 | +0.132 | +0.125 |
| ours seed 1 | −0.446 | −0.880 | −0.679 | −0.387 | −0.247 | −0.304 |
| ours seed 2 | −0.231 | +0.005 | +0.109 | +0.189 | +0.208 | +0.206 |
| ours seed 3 | −0.524 | −0.770 | −0.540 | −0.227 | −0.140 | −0.329 |
| ours seed 4 | −0.403 | −0.418 | −0.318 | −0.199 | −0.131 | −0.198 |

- Main prediction (reference code live at stage 1 in seeds 2, 3, and 4): seed 3 yes; seeds 2 and 4 no. **Not supported.**
- Other prediction 1 (our code dead in at least one of seeds 2 to 4): holds. Seeds 3 and 4 are dead; seed 2 is live by +0.004.
- Other prediction 2 (the stage-1 sign at step 2,000 already shows the fate): does not hold. Reference seed 2 is +0.011 at step 2,000 and ends at −0.060. Our seed 4 is +0.004 and ends at −0.034.

### What the runs show (observations after the fact; they grade nothing)

- **The stage-1 signature is a lottery in both codes.** Over seeds 0 to 4, the reference code is live in 3 and our code in 2. The stage-1 sign is also noisy from eval to eval: reference seed 2 goes −0.045, −0.176, −0.060 over the last three evals.
- **Edu bpb follows the stage-0 sign, not the stage-1 sign.** The 6 runs with stage-0 shuffle above zero are all at 1.3111 to 1.3208. The 4 runs below zero are all at 1.3315 to 1.3573. Reference seed 4 is dead at stage 1 and still gets 1.3186, because its stage 0 is live (+0.200). At this budget, what matters for bpb is whether the stage-0 decoder reads the content of the whole inner network (stage 1 plus main). It matters less whether the stage-1 decoder reads the main network. Go and SQL do not split this cleanly (our seed 1 has the second-best Go bpb of our five), so the claim is for edu only.
- **At stage 0, every run starts dead and most come alive later.** All 9 runs with a step-2,000 eval are below zero there (−0.13 to −0.52). The sign then rises, and 6 runs cross zero between steps 4,000 and 8,000. The 4 runs that end below zero are still rising at step 10,000 (reference seed 2 reaches +0.002). Then they fall back during the LR decay (steps 9,600 to 12,000). Reference seed 1 was at −0.588 at step 4,000, as deep as two of our dead seeds, and ended live.
- **So "dead" at stage 0 may mean "slow".** The fate is not set at init or by step 2,000. It looks like a race between the inner network becoming useful and the end of the run. Our code loses the race more often (3 of 5 runs against 1 of 5) and by more (−0.20 to −0.33 against −0.026). Five seeds per code cannot separate that gap from chance.
- **Mean edu bpb over seeds 0 to 4:** reference code 1.3204 (1.3156 to 1.3315), our code 1.3390 (1.3111 to 1.3573). Our two best runs match or beat the reference's best.

## Verdict

**Not supported.** The reference code is live at stage 1 in seed 3 only. Seeds 2 (−0.060) and 4 (−0.018) are dead. Over seeds 0 to 4 it is 3 of 5, and our code is 2 of 5.

## Updated hypothesis

The stage-1 signature does not separate the two codes. Both are lotteries at stage 1, and stage-1 use does not decide bpb at this budget. The signal that tracks bpb is the stage-0 sign. That sign starts negative in every run and climbs during training. The runs that end below zero look slow, not stuck: they are still climbing when the LR decay starts. If that is right, the 786M-byte budget is too short to judge a seed, and a longer budget brings every seed online in our code with no design change. The next run trains our three dead seeds at the 3.15B-byte budget where the reference code tied BPE: [dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md). The design options (a longer stable phase, a main-network warm start, a larger main-network FLOPs share, an auxiliary content loss) wait for that answer, because a slow start needs a different fix than a stuck one.

## Related

- [Faithful init and RoPE](../refuted/2026-11-02-faithful-init-and-rope.md)
- [Reference init in our code](../refuted/2026-11-02-reference-init-in-our-code.md)
- [Reference H-Net difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md)
