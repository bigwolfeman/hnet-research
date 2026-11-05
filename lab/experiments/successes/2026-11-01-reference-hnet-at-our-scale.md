# Experiment: reference H-Net at our scale

Status: success

Date proposed: 2026-11-01
Algorithm: none (the reference H-Net code, scaled down; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Is our dead 2-stage main network caused by our scale, or by our model and recipe?

Context: the released `hnet_2stage_L` uses its main network heavily (shuffle cost 1.35 bpb), and our 2-stage models never do (at most 0.004 bpb, at any scale up to 12.58B bytes) ([reference checkpoint main-network use](../successes/2026-11-01-reference-checkpoint-main-use.md)). That checkpoint is about 30x larger than our models and saw about 30x more data. So the gap could come from scale, or from what our code does differently: Transformer outer stages where the reference uses Mamba-2, a flatter per-stage LR (ours [1.5, 1.2, 1.0], the paper's Eq. 11 about [3.0, 1.5, 0.7]), and targets 4 x 2 where the paper uses 3 x 3. This run trains the reference code at our size, data, batch, schedule, and budget. Only the model code and the paper's recipe change.

## Hypothesis

The dead main network is not a scale effect. Something in our model or recipe lets the stage-1 decoder ignore the main network. The paper's own ablation shows the encoder type matters most: Transformer-only encoders and decoders are "by far the worst" and compress poorly. In the reference design, the Mamba-2 outer stages give each chunk boundary a state that summarizes the whole chunk. That makes the main network's input informative early in training, so the decoder learns to read the main network's output.

## Predictions

Main prediction (only this one answers the Question):

- Over the run up to 3.15B training bytes, the reference 2-stage H-Net at our scale reaches a `probe/coarse_shuffle_s1` above the largest value our blog-split 2-stage showed at any eval in the same byte range (0.0041, step 6,000 of [ladder attempt 2](../refuted/2026-11-01-compute-ladder-vs-bpe.md)), and it is still above that value at the 3.15B rung. Above at the rung: supported. At or below: not supported.

Other predictions (at most two):

- At the 3.15B rung, the reference arm's coarse path at stage 1 has the healthy signature: shuffle rise above zero rise.
- At the 3.15B rung, the reference arm has lower edu val bpb than our blog split's 3.15B rung (1.4235), by more than the 2-stage seed spread (0.059).

If supported, we do: scale is not the cause. Find which reference difference wakes the main network: in our code, swap one at a time at the 0.79B rung (Mamba-2 outer stages, the Eq. 11 LR multipliers, targets 3 x 3), read with this probe. Then rerun the BPE comparison with the fixed design.

If not supported, we do: even the reference design leaves its main network unused at our scale. That points to scale. Phase 1's 2-stage criterion is then out of reach on one 5090 at this model size. We take it back to the user with the fallback options: a 1-stage base for Phase 2, or the JEPA training method to cut the data needed.

Gates (the run is broken if one fails; gates are not predictions):

- Adapter tests pass (`tests/test_refmodel.py`, reference venv: aux consistency, the ratio loss equals the reference `load_balancing_loss`, causality, probe ablation, parameter groups, a training step). Our suite passes unchanged (136 tests).
- No ratio collapse: each stage's edu ratio stays between 1.5x and 6x at every eval. Fwd FLOPs/byte is reported at each rung and must be within 10% of BPE's 22.27M. The reference recipe has no ratio controller, so the ratios float. The smoke run measured 23.1M (+3.5%).
- Training bytes per rung equal our blog split's (786.4M and 3.146B; same steps, batch, and sequence length). Finite loss.

## Method

```
model (configs/experiment/ref_hnet_2stage.yaml, hnet/refmodel.py wraps goombalab/hnet @ 3673fe1 as `refnet`):
    arch_layout  m4 | T1m4 | T10 | m4T1 | m4     (hnet_2stage_L's layout; L has T26)
    d_model      256 | 256 | 512                 (L: 1024 | 1024 | 1536)
    d_intermediate 0 | 768 | 1408 ; Mamba-2 d_state 128, expand 2, headdim 64 ; attention windows 1023 | 1023 | global
    41.8M params ; fwd FLOPs/byte at targets 3 x 3: 22.58M (+1.4% vs BPE) ; split 44/22/34% (L: 34/17/49%)
recipe (H-Net paper): targets 3 x 3, ratio-loss weight 0.03, no ratio controller,
    peak LR 6.25e-4 x per-stage multipliers [3.03, 1.75, 0.72] (paper Eq. 11, N_GPT 4.6, D = 256, 256, 512),
    reference init (std 0.02, residual-scaled out_proj/fc2, zero residual_proj, identity router q/k, std 1 embeddings)
everything else = our ladder arm (configs/train/ladder.yaml): data mix, batch 16 x 4096 bytes, AdamW (0.9, 0.95),
    wd 0.1 on >= 2-D params, clip 1.0, warmup 500, WSD linear decay over the last 20%, bf16 autocast
budget: max_steps 48,000 (3.146B bytes), ladder_fracs [0.25] -> rung at 12,000 steps (786.4M bytes)
    (the same branch steps as our blog split's 12,000 and 48,000 rungs)
every eval (2,000 steps): val bpb per domain, ratios, FLOPs; coarse-path probe on the 8 edu probe batches:
    probe/coarse_{zero,shuffle}_s{0,1} = bpb rise with that stage's dechunked inner output zeroed / rolled across rows
```

- Launch: `PY=../2026-11-01-reference-hnet-probe/.venv/bin/python TRAIN=ladder scripts/run_arms.sh ignored/runs/ref-hnet-at-our-scale ref-hnet-at-our-scale ref_hnet_2stage:0 -- train.max_steps=48000 "train.ladder_fracs=[0.25]"`, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`. wandb project `hnet-router`, group `ref-hnet-at-our-scale`.
- Control: our blog-split 2-stage, ladder attempt 2 (wandb `5rvc25h5`), same data, batch, schedule, and branch steps. Its rungs are taken from that run. They are not rerun.
- Speed (smoke run): 102 ms/step, about 1.5 h for the run. Peak memory 16.8 GB.
- Differences from the reference recipe that stay: our batch (65k bytes per step, not 2.1M), our schedule (linear decay, not inverse square root), weight decay on 2-D parameters only (the reference also decays Mamba's 1-D A_log, D, and dt_bias), ratio loss averaged over valid positions.

### Amendment 2026-11-01 15:55 (attempt 1 broken: FLOPs gate)

- Attempt 1 (wandb `b7nn8d09`, log dir `ignored/runs/ref-hnet-at-our-scale`, code at commit cc081e6) ran to the end, but it failed the FLOPs gate. With no ratio controller, both ratios drifted below the 3 x 3 targets: edu 2.85 x 3.10 at step 2,000, 2.68 x 2.94 at the 0.79B rung, and 2.53 x 2.55 at the end. Edu fwd FLOPs/byte went over +10% at step 14,000 (24.70M) and ended at 26.79M (+20.3%). The 0.79B rung held every gate (24.37M, +9.4%). Artifacts: [results/](../results/2026-11-01-reference-hnet-at-our-scale/) (`attempt1-*`).
- Fix: the rerun turns on our RatioController (`train.ratio_integral_lr=0.002`, the value our blog split used), which pins each stage's train-mix ratio to 3. Everything else is unchanged. This adds one piece of our recipe to the reference arm. The blog-split control used the same controller, so the two arms now share it.
- Rerun: log dir `ignored/runs/ref-hnet-at-our-scale-r2`, wandb group `ref-hnet-at-our-scale-r2`, same command plus `train.ratio_integral_lr=0.002`.
- What I saw before this amendment: all of attempt 1. Its `coarse_shuffle_s1` was above 0.0041 at every eval (0.039 at step 2,000, 0.129 at the 0.79B rung, 0.527 at the end), shuffle rose above zero at both stages, and its edu bpb at the end (1.159) was below our blog split's 1.4235. These come from a run that spent up to 20% more FLOPs per byte, so they do not grade any prediction. The Predictions do not change.

## Results

Attempt 2 (wandb `2lf7edo2`, log dir `ignored/runs/ref-hnet-at-our-scale-r2`, code at commit 7a7e80d, RatioController on). It ran to the end with no restart. Artifacts: [results/](../results/2026-11-01-reference-hnet-at-our-scale/) (`attempt2-summary.json`, `attempt2-ladder_12000.json`, `attempt2-eval-curves.csv` with every eval).

### Gates

| gate | value | result |
|---|---|---|
| adapter tests, our suite | 6 pass (reference venv), 136 pass | pass |
| edu ratios at every eval (1.5x to 6x) | s0 2.99 to 3.16, s1 2.73 to 3.07 | pass |
| edu fwd FLOPs/byte at every eval (BPE 22.27M, 10%) | 22.35M to 23.24M (+0.4% to +4.4%); 0.79B rung 22.70M, 3.15B rung 22.70M (+1.9%) | pass |
| training bytes | 786,432,000 and 3,145,728,000, the same as the blog split | pass |
| finite loss | yes | pass |

The controller held the train-mix ratios at 3: `train/ratio_lambda_s0` rose to 0.015 (stage 0 pulled toward finer chunks), and `lambda_s1` stayed within 0.007 of 0. Go val runs coarser than edu at stage 1 (6.47 at the end); the gate is on edu only.

### Main prediction: stage-1 coarse-path use

| step (bytes) | `coarse_shuffle_s1` reference arm | blog split (control) |
|---|---:|---:|
| 2,000 (131M) | 0.033 | under 0.0005 |
| 6,000 (393M) | 0.070 | 0.0041 (its maximum) |
| 12,000 rung (786M) | 0.135 | 0.0008 |
| 24,000 (1.57B) | 0.242 | under 0.0005 |
| **48,000 rung (3.15B)** | **0.460** | **0.000014** |

The reference arm is above 0.0041 at every eval from step 2,000. It rises at every eval. At the 3.15B rung it is 0.460, 112x the control's maximum and about 30,000x the control at the same rung.

### Other predictions

- Healthy signature at the 3.15B rung: shuffle 0.460 > zero 0.321 at stage 1, and 1.648 > 1.329 at stage 0. Holds. Shuffle was above zero at stage 1 at every eval.
- Edu bpb at the 3.15B rung: 1.1708 against the blog split's 1.4235, lower by 0.253 (seed spread 0.059). Holds.

### Against the other ladder arms at 3.15B bytes (measurement, one seed each)

| arm | edu bpb | Go bpb | SQL bpb (eval only) | params |
|---|---:|---:|---:|---:|
| reference arm | **1.1708** | **0.6534** | **1.2974** | 41.8M |
| BPE Transformer | 1.1769 | 0.732 | 1.460 | 62.5M |
| our matched 1-stage | 1.2742 | 0.766 | 1.523 | 26.0M |
| our blog split 2-stage | 1.4235 | | | 49.4M |

- On edu the reference arm ties BPE (−0.006, inside the 0.059 spread). On Go it is 0.079 below BPE, and on SQL 0.163 below. Our 1-stage arm needed 12.58B bytes to get within 0.043 of BPE on edu.
- Attempt 1 (no controller, up to +20% FLOPs/byte) ended at 1.1592 edu bpb and `coarse_shuffle_s1` 0.527. Holding the FLOPs cost 0.012 bpb and did not change the picture.
- Speed: 113 ms/step against 52 ms/step for the blog split at the same FLOPs/byte; peak memory 23.1 GB. The GPU drew 550-575 W.

## Verdict

**Supported.** At our scale, data, batch, schedule, and FLOPs/byte, the reference 2-stage H-Net uses its main network from step 2,000 on, and the use grows through the run (`coarse_shuffle_s1` 0.460 at 3.15B bytes, against at most 0.0041 for our blog split). Scale is not why our main network is dead. The cause is in our model or recipe.

## Updated hypothesis

Something our 2-stage design does differently lets the stage-1 decoder ignore the main network. The reference design fixes it and, at 3.15B bytes, ties BPE on edu and beats it on code, which our blog split did not do at 12.58B bytes. The next step is to find which difference matters: [reference H-Net difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md) removes one reference feature at a time (Mamba-2 outer stages, the paper's per-stage LR, 3 x 3 targets) at the 0.79B budget, where the reference arm's main network is already in use (0.135 at that rung).

## Related

- [Reference checkpoint main-network use](../successes/2026-11-01-reference-checkpoint-main-use.md)
- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
- [Reference probe harness spike](../../spikes/2026-11-01-reference-hnet-probe/README.md)
