# Experiment: faithful init and RoPE

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (our model with the 2025 init rule and partial RoPE; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

With the reference init rule and partial RoPE fixed in our code, does our model give the reference's live main network on fresh init, in both seeds?

Context: from the reference's exact step-0 weights, our code was live in seed 0 (stage-1 shuffle − zero +0.050) and just short in seed 1 (−0.010) ([reference init in our code](../refuted/2026-11-02-reference-init-in-our-code.md)). With our old init it was −0.020 and −0.061. Our init now follows the 2025 rule (per-block residual count, default residual bias), and its step-0 statistics match the reference's. `rope_dim` 32 matches the reference's partial RoPE ([note](../../../.agents/notes/implemented/bug-fix/2026-11-02-init-rule-and-partial-rope.md)).

## Hypothesis

Our implementation now matches the reference in every part that acts on training, so our code with fresh init trains like the reference code: a live stage-1 main network in both seeds.

## Predictions

Main prediction (only this one answers the Question):

- At 12,000 steps, both seeds (0 and 1) of `reflayout_2stage` with fresh init have `probe/coarse_shuffle_s1` above `probe/coarse_zero_s1`. Both above: supported. Either at or below: not supported.

Other predictions (at most two):

- Both seeds' edu bpb is within the 2-stage seed spread (0.059) of the reference code's for the same seed (1.3178, 1.3185).
- The two seeds' `coarse_shuffle_s1` differ by less than 3x (the reference code: 1.02x; our old init: 8.4x; reference init in our code: 2.3x).

If supported, we do: our code is fixed. The next step is the bisection from the reference layout toward the blog split (widths, stage-0 blocks, stage-1 depth and compute split, targets), to choose our 2-stage design. Then the BPE ladder reruns with it.

If not supported, we do: test partial RoPE and the init separately (two arms: init fix with full RoPE, and the reverse), and look at the fp32 residual adds in the reference's fused norm.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`: 147 passed, 1 skipped), and the reference venv's adapter tests pass (7). The step-0 std of every tensor class is within 3% of the reference's (56 classes, 4 seeds).
- Edu ratios within 15% of 3 x 3 at the end; edu fwd FLOPs/byte within 5% of 22.27M. Bytes 786,432,000 per arm. Finite loss.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
  init: Stack.height = sum(2 if MLP else 1); out_proj/fc2 std 0.02 / sqrt(sum of heights over enclosing levels);
        residual_proj weight 0, bias nn.Linear default
  RoPE: rope_dim 32 of head_dim 64 (reflayout_2stage.yaml)
arm: reflayout_2stage, fresh init, seeds 0 and 1; RatioController; 12,000 steps, no branches; probe every 2,000
controls (not rerun): reference code seeds 0/1 (+0.021, +0.031); our code with old init (F: −0.020, −0.061) and
  with the reference's weights (G: +0.050, −0.010)
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 12,000, both seeds
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `TRAIN=ladder scripts/run_arms.sh ignored/runs/faithful-init-and-rope faithful-init-and-rope reflayout_2stage:0 reflayout_2stage:1 -- train.max_steps=12000 "train.ladder_fracs=[]"`. wandb project `hnet-router`, group `faithful-init-and-rope`.
- Cost: about 20 min per arm.

## Results

Both arms ran to 12,000 steps (commit 59a2cff). Log dir `ignored/runs/faithful-init-and-rope`; wandb seed 0 `wmo21v4e`, seed 1 `yh1nptqe`. Artifacts: [results/](../results/2026-11-02-faithful-init-and-rope/) (`*-summary.json`, `eval-curves.csv`).

### Gates

| gate | seed 0 | seed 1 | result |
|---|---|---|---|
| suites; step-0 std per tensor class within 3% of the reference | 147 passed, 1 skipped; 7 passed; 56 of 56 | | pass |
| edu ratios at the end (3 x 3 ±15%) | 3.04 x 2.93 | 3.04 x 2.99 | pass |
| edu fwd FLOPs/byte (22.27M ±5%) | 22.59M | 22.44M | pass |
| training bytes, finite loss | 786,432,000, yes | same | pass |

### Stage-1 signature and bpb at the end (12,000 steps), with every run of this configuration so far

| run | seed 0: shuffle − zero | seed 0 edu bpb | seed 1: shuffle − zero | seed 1 edu bpb |
|---|---:|---:|---:|---:|
| **our code, faithful init + partial RoPE (this run)** | **+0.023** | **1.3208** | **−0.127** | **1.3536** |
| our code, reference weights (G) | +0.050 | 1.3105 | −0.010 | 1.3288 |
| our code, old init (F) | −0.020 | 1.3248 | −0.061 | 1.3797 |
| reference code | +0.021 | 1.3178 | +0.031 | 1.3185 |

This run, by eval (steps 2,000 to 12,000): seed 0 shuffle − zero +0.011, +0.021, +0.019, +0.024, +0.024, +0.023; seed 1 −0.022, −0.054, −0.066, −0.081, −0.105, −0.127 (`shuffle_s1` 0.0017 → 0.0080).

- Main prediction (both seeds live): seed 0 yes, seed 1 no. **Not supported.**
- Other prediction 1 (edu bpb within 0.059 of the reference's same seed): holds, +0.003 and +0.035.
- Other prediction 2 (the two seeds' `shuffle_s1` within 3x): does not hold, 20x (0.159 against 0.0080).

### What the arms show

- **Seed 0 now matches the reference code** on the signature (+0.023 against +0.021), main-network use (0.159 against 0.135), stage-0 balance, and bpb (1.3208 against 1.3178).
- **Seed 1 is dead from the first eval** (shuffle 0.0017, `shuffle_s0` 0.073) and never recovers. Its init is a fresh draw from the same distribution as the reference's. It is not the reference's seed-1 weights: our code builds modules in a different order, so the same seed gives different draws. With the reference's own seed-1 weights (G), our code was near the line (−0.010).
- **So the outcome follows the particular init draw, at matched init distribution.** Across the six runs of the reference layout in our code (F, G, H), 2 are live and 4 are not. The reference code is 2 of 2. Two seeds cannot tell whether the reference code is more robust or got two good draws.

## Verdict

**Not supported.** With the 2025 init rule and partial RoPE, our code with fresh init is live in seed 0 (+0.023, matching the reference) and dead in seed 1 (−0.127).

## Updated hypothesis

Two explanations remain. (a) A difference in the code still acts on training and makes our code less robust. The fp32 residual adds inside the reference's fused norm are the last known one. (b) The main network's fate is decided by the init draw in both codes, at this budget, and the reference's two seeds were both good draws. The two call for different work. (a) needs more code parity. (b) needs a design that makes the main network come online for any draw. The next run counts live seeds in both codes on fresh seeds: [main-network seed rate](2026-11-02-main-network-seed-rate.md).

## Related

- [Reference init in our code](../refuted/2026-11-02-reference-init-in-our-code.md)
- [Reference layout in our code](../refuted/2026-11-02-reference-layout-in-our-code.md)
- [Init rule and partial RoPE note](../../../.agents/notes/implemented/bug-fix/2026-11-02-init-rule-and-partial-rope.md)
