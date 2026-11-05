# Experiment: reference layout in our code

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (a configuration of our model; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Is the dead main network of our 2-stage H-Net caused by its configuration (widths, block layout, compute split, targets, LR) or by our implementation?

Context: the reference H-Net trained at our scale has a live main network (stage-1 shuffle − zero +0.021 and +0.031 at 786M bytes, two seeds; [at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md), [difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md)). Every configuration of our model so far has a dead one: −0.092 (blog split), −0.176 (paper LR), −0.248 (Mamba-2 + paper LR) ([our H-Net with Mamba-2](../refuted/2026-11-02-our-hnet-with-mamba2.md)). Our router, residual path, and EMA dechunk follow the 2025 reference code. Stack layouts now let our model build the reference configuration exactly: the same layout, widths, MLP widths, windows, targets, and LR, with the same FLOPs/byte (22.576M at 3 x 3) and the same parameter count (41,801,344).

## Hypothesis

The cause is the configuration. Our implementation is a faithful copy of the reference modules, so with the reference's configuration our model trains like the reference and its main network is used.

## Predictions

Main prediction (only this one answers the Question):

- At the end of the run (12,000 steps, 786.4M bytes), the reference-layout arm (seed 0) has the live stage-1 signature: `probe/coarse_shuffle_s1` above `probe/coarse_zero_s1` (anchor: 0 for shuffle − zero). Above: supported. At or below: not supported.

Other predictions (at most two):

- The seed-1 arm also has shuffle above zero at stage 1 at the end.
- Both arms' edu val bpb at the end is within the 2-stage seed spread (0.059) of the reference arm's (1.3178 seed 0, 1.3185 seed 1).

If supported, we do: the configuration is the cause, and our code is sound. Bisect the configuration in our code at this budget, starting from the reference layout and moving one group at a time toward the blog split: widths (256/384/512), stage-0 blocks (`m4` vs `M2` vs `T2`), stage-1 depth and compute split, then targets. The result picks our 2-stage design, and the BPE ladder reruns with it.

If not supported, we do: the cause is an implementation detail. Audit our modules against the reference line by line (block norm and residual order, partial RoPE, init of each module, the router's handling of padding, the dechunk EMA and its STE, the residual projection's dtype) and test the differences one at a time, starting with the ones that touch the decoder input.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`, 143 passed, 1 skipped). Fwd FLOPs/byte at 3 x 3 and the parameter count equal the reference arm's (22.576M, 41,801,344).
- Edu ratios within 15% of the targets (3 x 3) at the end; edu fwd FLOPs/byte within 5% of BPE's 22.27M.
- Training bytes 786,432,000 per arm. Finite loss.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file; our model (model.kind hnet)
arm: reflayout_2stage (configs/experiment/reflayout_2stage.yaml), seeds 0 and 1
  layout   stage 0 enc m4, dec m4 ; stage 1 enc T1m4, dec m4T1 ; main T10      (= ref_hnet_2stage)
  widths   256 / 256 / 512 ; head_dim 64 ; SwiGLU 768 (d 256), 1408 (d 512)   (mlp_ratio 2.75, multiple of 128)
  windows  1024 keys (reference window_size 1023), main global
  Mamba-2  model.ssm = d_state 128, d_conv 4, expand 2, headdim 64, chunk_size 256 (the reference values)
  recipe   targets 3 x 3, weight 0.03, LR 6.25e-4 x [3.03, 1.75, 0.72], RatioController 0.002
controls (not rerun): the reference arm ref_hnet_2stage + RatioController at 12,000 steps, seed 0 (0.79B rung of
  wandb 2lf7edo2) and seed 1 (u8hqkojb)
schedule: configs/train/ladder.yaml, max_steps 12,000, no branches; eval every 2,000 steps with the probe
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 12,000, seed 0
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `TRAIN=ladder scripts/run_arms.sh ignored/runs/reference-layout-in-our-code reference-layout-in-our-code reflayout_2stage:0 reflayout_2stage:1 -- train.max_steps=12000 "train.ladder_fracs=[]"`. wandb project `hnet-router`, group `reference-layout-in-our-code`.
- Differences from the reference arm that stay: our module code; full RoPE on all 64 head dims (the reference rotates 32); the encoder and decoder stacks of stage 0 hold an unused MLP width (no MLP blocks there, so no parameters).
- Cost: about 20 min per arm.

## Results

Both arms ran to 12,000 steps (commit 057e817). Log dir `ignored/runs/reference-layout-in-our-code`; wandb seed 0 `3vtqdaxq`, seed 1 `gjsjr4tl`. Artifacts: [results/](../results/2026-11-02-reference-layout-in-our-code/) (`*-summary.json`, `eval-curves.csv`).

### Gates

| gate | seed 0 | seed 1 | result |
|---|---|---|---|
| suite; FLOPs/byte at 3 x 3 and params equal to the reference arm | 143 passed, 1 skipped; 22.576M and 41,801,344 both | | pass |
| edu ratios at the end (3 x 3 ±15%) | 3.02 x 2.99 | 3.10 x 3.01 | pass |
| edu fwd FLOPs/byte at the end (22.27M ±5%) | 22.53M | 22.13M | pass |
| training bytes, finite loss | 786,432,000, yes | same | pass |

### Coarse-path probe and bpb at the end (12,000 steps)

| arm | `shuffle_s1` | `zero_s1` | shuffle − zero | `shuffle_s0` | `zero_s0` | edu bpb | Go bpb | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference layout, our code, seed 0 | 0.0656 | 0.0856 | **−0.020** | 1.040 | 1.041 | 1.3248 | 0.918 | 91 |
| reference layout, our code, seed 1 | 0.0078 | 0.0690 | −0.061 | 0.427 | 0.888 | 1.3797 | 0.935 | 94 |
| reference code, seed 0 (control) | 0.135 | 0.114 | +0.021 | 1.041 | 0.917 | 1.3178 | 0.884 | 113 |
| reference code, seed 1 (control) | 0.132 | 0.101 | +0.031 | 1.048 | 1.021 | 1.3185 | 0.890 | 104 |

`shuffle_s1` / `zero_s1` by eval (steps 2,000 to 12,000):

- seed 0: 0.019/0.018, 0.029/0.029, 0.038/0.048, 0.047/0.058, 0.057/0.068, 0.066/0.086
- seed 1: 0.0016/0.010, 0.0029/0.016, 0.0033/0.025, 0.0048/0.036, 0.0067/0.046, 0.0078/0.069

- Main prediction: shuffle − zero at stage 1 is −0.020 for seed 0. **Not supported.**
- Other prediction 1 (seed 1 also live): does not hold (−0.061).
- Other prediction 2 (edu bpb within 0.059 of the reference): seed 0 holds (+0.007); seed 1 does not (+0.061).

### What the arms show

- **With the reference configuration, our code moves most of the way to the reference, but not all of it.** Seed 0 gets half the reference's main-network use (0.066 against 0.13), reaches the reference's stage-0 balance (shuffle 1.040, zero 1.041), and matches its bpb within 0.007. No configuration of our model came this close before.
- **Our code is seed-sensitive where the reference code is not.** The two reference seeds agree to 0.003 in `shuffle_s1` and 0.0007 in bpb. Our two seeds differ 8x in `shuffle_s1` and by 0.055 in bpb. The split is visible at the first eval: at step 2,000 seed 0 has shuffle equal to zero at stage 1 and `shuffle_s0` 0.29; seed 1 has 0.0016 against 0.010 and `shuffle_s0` 0.058. Whatever decides the outcome happens in the first 2,000 steps.
- **The single-seed results in our code carry this uncertainty.** The paper-LR arm (0.112) and the Mamba-2 arms (0.0026, 0.00002) each had one seed. The 13x gap between the blog-split control (0.0107) and the ladder rung (0.0008) is the same effect.

## Verdict

**Not supported.** With the reference's exact configuration (layout, widths, MLP widths, windows, targets, LR, FLOPs, parameter count), our implementation does not give the live stage-1 signature: −0.020 for seed 0 and −0.061 for seed 1. Configuration explains most of the gap, but not all of it. The rest is in our implementation, and it shows up as a coin flip in the first 2,000 steps.

## Updated hypothesis

An implementation difference makes early training fragile, so the main network comes online in some seeds and not in others. The reference code comes online in both seeds. A difference that acts in the first steps is most likely in initialization: per-stack output-projection scaling, Mamba-2 in_proj and conv init, embedding and head init, or partial RoPE. A difference in the forward pass is the other option. Next, two checks split them. (1) Forward equivalence: copy the reference arm's weights into our model and compare the logits. Equal logits mean the forward passes are the same, and the gap is in initialization. (2) Init transplant: train our model from the reference model's step-0 weights, for two seeds. If both seeds come out live like the reference, initialization is the cause.

## Related

- [Our H-Net with Mamba-2](../refuted/2026-11-02-our-hnet-with-mamba2.md)
- [Paper LR in our H-Net](../refuted/2026-11-02-paper-lr-in-our-hnet.md)
- [Reference H-Net difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md)
- [Stack layouts note](../../../.agents/notes/implemented/feature/2026-11-02-stack-layouts-mamba2.md)
