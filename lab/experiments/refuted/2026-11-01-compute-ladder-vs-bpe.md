# Experiment: compute ladder vs BPE

Status: refuted

Date proposed: 2026-11-01
Algorithm: none (WSD compute ladder; pseudocode in Method and in `train.py`)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does the 2-stage byte H-Net close the gap to the BPE Transformer, and pass it, when compute grows 16x past our earlier runs, at matched FLOPs/byte and matched training bytes?

Context: at 786M bytes (about 5e16 training FLOPs), every H-Net lost to BPE by 0.14 bpb or more ([controls](../refuted/2026-10-31-byte-and-bpe-controls.md)), and the 2-stage main network stayed unused ([deep-main split](../refuted/2026-10-31-deep-main-compute-split.md)). Cartesia's Part II compute chart shows the byte H-Net ahead of BPE on code at every point from 1e18 FLOPs up. Their chunking graph shows the routers in an off-target plateau until about 13B bytes; our 786M-byte runs end inside that plateau. Our earlier runs never reached the regime that Cartesia reports.

## Hypothesis

The H-Net loses at 5e16 FLOPs because the hierarchy has not come online yet: the router and the main network need many more bytes than a fixed tokenizer, which is good from step 0. With 16x the bytes, the main network starts to carry document-specific information, and the hierarchy's advantage (fewer main-network positions per byte, chunks learned end to end) shows in bpb.

## Predictions

Main prediction (only this one answers the Question):

- At the top rung (12.58B training bytes), the blog-split 2-stage H-Net has lower edu validation bpb than the BPE Transformer. Grading uses the 2-stage seed spread measured at 786M bytes (0.059, deep-main run 2 baseline), because each arm here has one seed. H-Net lower by more than 0.059: supported. Inside +-0.059: can't tell. H-Net higher by more than 0.059: not supported.

Other predictions (at most two):

- The H-Net minus BPE edu bpb gap shrinks from rung to rung: 0.79B > 3.15B > 12.58B bytes.
- The blog-split main-network use (`probe/coarse_shuffle_s1`) is larger at the top rung than at the 0.79B rung.

If supported, we do: the hierarchy pays at about 1e18 training FLOPs at this model size. The blog split becomes the 2-stage default (Agent Note). Phase 2 (the looped LM) builds on a model trained at this scale or more. Rerun the router comparison (`eoc` vs `lookahead`) at a ladder rung where the main network is in use.

If not supported, we do: read the trend. If the gap shrinks at every rung, the crossing is past 12.6B bytes. Plan a larger run on more data (the train set holds about 14.5B bytes) or a larger model. If the gap does not shrink, scale is not the cause. Return to the gradient path into the main network (per-level gradient norms, main LR multiplier).

Gates (the run is broken if one fails; gates are not predictions):

- Fwd FLOPs/byte at each rung's measured ratios: the blog split and the matched 1-stage are within 2% of the BPE Transformer's 22.27M (Cartesia's standard). The ratio controller holds the ratio, so this gate also checks the controller.
- Measured edu ratios at each rung are within 15% of the targets (4x and 2x). No collapse.
- Training bytes per rung are equal across arms within 1%. Finite loss for the whole run.
- Ladder restore is exact. A GPU check (snapshot, 3 steps, restore) showed bit-exact parameters and AdamW state. The smoke run showed each rung decaying to a lower bpb than the stable state at the same step.

## Method

```
arms (one seed each, in this order, local RTX 5090, torch 2.14.1+cu130, compiled, bf16):
  blog_split_2stage  stage0 enc/dec 2/2 d256 w512 | stage1 enc/dec 1/1 d384 w256 | main 13 x d512
                     ratios held at 4 x 2 by RatioController (ratio_dual_lr 0.01)
                     fwd FLOPs/byte 22.44M (+0.8% vs BPE). Split stage0/stage1/main 39/10/50% (Cartesia 31/12/57). 49.4M params.
  bpe_transformer    the controls' BPE Transformer: vocab 32k, d512, 9 layers, 1024 tokens. 22.27M fwd FLOPs/byte. 62.5M params.
  matched_1stage     stage0 enc/dec 2/2 d256 w512 | main 7 x d512, ratio held at 4x. 21.99M (-1.2%). 26.0M params.
budget: 12.58B bytes = 16 x 786.4M  (H-Net 192,000 steps x 16 x 4096; BPE 180,080 steps x 16 x 1024 tokens)
        training FLOPs at the top rung ~ 3 x 22.3M x 12.58B ~ 8.4e17
ladder (train=ladder, ladder_fracs [1/16, 1/4]): WSD with a shared stable phase
  at step 0.8 b for each rung b in {1/16, 1/4} x max_steps:
      snapshot model + AdamW (+ controller weights); decay linearly to b; eval; save ladder_<b>.pt; restore; continue
  the main run decays over its own last 20% to the top rung
eval every 2000 steps: val bpb per domain (edu, go, sql), ratios, FLOPs, coarse_use probe; probes every 16,000
```

- Launch: `lab/spikes/2026-10-30-two-stage-hnet-smoke/scripts/run_ladder.sh ignored/runs/compute-ladder-vs-bpe`. wandb project `hnet-router`, group `compute-ladder-vs-bpe`. Rung results are logged as `ladder_b<steps>/*`, saved as `ladder_<steps>.json` in the run dir, and stored in `summary.json` under `ladder`.
- The BPE rungs (11,255 and 45,020 steps) hold the same bytes as the H-Net rungs (12,000 and 48,000 steps).
- Data: the main run plus its two branches draw about 13.4B bytes: 0.93 epoch of the edu train set (12.88B bytes) and 0.83 epoch of Go (1.61B bytes). The loader samples random offsets, so no window repeats in order.
- Each rung's branch sees new batches from the stream, so the main run after a branch sees different batches than it would without the branch. Batches are random samples of the same mix, so the effect is noise only.
- Cost: about 3.7 h (blog split) + 2.7 h (BPE) + 3.5 h (1-stage), about 10 h in total.
- The ratio controller departs from Cartesia's recipe. Their routers go through an off-target plateau early, and the controller does not allow that. The controller is needed to hold FLOPs/byte within 2%, which is Cartesia's own matching standard.
- The 1-stage arm is not in a prediction. It gives the 1-stage vs 2-stage ordering at each rung (Cartesia Part I: 2 stages overtake 1 stage late).

### Amendment 2026-11-01 01:35 (step 2,000 of arm 1, no rung or arm-vs-arm data)

- The 2% FLOPs gate cannot hold on a single text domain, so it changes to 5% on edu val, the standard the [controls](../refuted/2026-10-31-byte-and-bpe-controls.md) used. Reason: the ratio controller holds the *train-mix* ratio at 4x. Edu text chunks finer than Go (edu val 3.86x vs Go 4.3x at step 2,000), so on edu val the blog split runs at 23.13M FLOPs/byte (+3.9%). The BPE side moves the other way: on the train mix, its bytes per token are 4.2647 instead of 4.5273, so it spends 23.64M per byte. At 4 x 2 the blog split spends 22.44M there, which is -5.1%. The choice of text domain moves the comparison by about 9%, more than the 2% gate.
- Gate now: edu-val fwd FLOPs/byte within 5% of BPE's 22.27M at each rung. I also report the train-mix ratio of each arm vs BPE's 23.64M, as a measurement.
- Nothing else changes. The run continues.

### Amendment 2026-11-01 03:00 (attempt 1 broken; arm-1 data only, no BPE or 1-stage data)

- Attempt 1 is a broken run: two gates failed in arm 1, and a power loss stopped it at step 141,550 of 192,000. Record: [results/attempt1.md](../results/2026-11-01-compute-ladder-vs-bpe/attempt1.md).
- Cause: the first RatioController design (dual ascent on the ratio-loss weight) is one-sided. From about step 35,000, the LM pulled stage 1 toward *coarser* chunks. The controller drove the stage-1 weight to 0, and the stage-1 ratio drifted from 2.0 to 4.8 on the train mix. At the 3.15B rung, the edu stage-1 ratio was 2.44 (+22%, gate 15%) and fwd FLOPs/byte was 20.28M (−8.9%, gate 5%). The 0.79B rung held every gate.
- Fix: the controller is now a PI controller. The fixed 0.03 ratio loss is the P term, plus a signed integral term `lam_s * G_s` with `lam_s += lr (N_s F_s - 1)`. Hydra key `train.ratio_integral_lr` replaces `ratio_dual_lr`. Its value is 0.002, from a 6,000-step calibration (0.01 let the integrator drive a slow stage-1 wander). Why and tests: [ratio controller note](../../../.agents/notes/implemented/feature/2026-10-31-ratio-controller.md).
- Power safety: `train.resume_every = 8000` writes `resume.pt` (model, AdamW, controller, loader rng state, wandb id, rung results). `run_arms.sh` runs each arm in a fixed dir and resumes it when the same command runs again. Note: [resumable runs](../../../.agents/notes/implemented/feature/2026-11-01-resumable-runs.md).
- Rerun: all three arms from step 0 on the fixed code, log dir `ignored/runs/compute-ladder-vs-bpe-r2`, wandb group `compute-ladder-vs-bpe-r2`. The BPE arm does not use the controller, but it reruns too, so that all arms share one code version.
- What I saw before this amendment: arm 1 only. I saw its bpb at both rungs and in the main run, and its coarse-path probe (`coarse_shuffle_s1` 0.0105 at 0.79B, 0.0059 at 3.15B). Those values come from a run whose stage-1 ratio was off target, so they do not grade other prediction 2. I saw no BPE value at any rung. The Predictions do not change.

## Results

Attempt 2 (log dir `ignored/runs/compute-ladder-vs-bpe-r2`, wandb group `compute-ladder-vs-bpe-r2`): blog split `5rvc25h5`, BPE `dnpo8u01`, matched 1-stage `a60y5spl`. Code at commit 919acbe. All three arms ran to the end with no power loss. Artifacts: [results/](../results/2026-11-01-compute-ladder-vs-bpe/) (`*-summary.json`, `*-ladder_*.json`, `r2-eval-curves.csv` with every eval of every arm).

### Gates (all hold)

| rung | arm | edu ratio s0 | edu ratio s1 | edu fwd FLOPs/byte | vs 22.27M | train bytes |
|---|---|---:|---:|---:|---:|---:|
| 0.79B | blog split | 4.01 | 2.01 | 22.35M | +0.4% | 786,432,000 |
| 0.79B | 1-stage | 4.02 | | 21.90M | −1.7% | 786,432,000 |
| 0.79B | BPE | | | 22.13M | −0.6% | 786,423,551 |
| 3.15B | blog split | 4.06 | 1.99 | 22.27M | 0.0% | 3,145,728,000 |
| 3.15B | 1-stage | 4.04 | | 21.83M | −2.0% | 3,145,728,000 |
| 3.15B | BPE | | | 22.13M | −0.6% | 3,145,694,204 |
| 12.58B | blog split | 4.05 | 1.99 | 22.29M | +0.1% | 12,582,912,000 |
| 12.58B | 1-stage | 4.12 | | 21.56M | −3.2% | 12,582,912,000 |
| 12.58B | BPE | | | 22.13M | −0.6% | ~12,582,800,000 |

- BPE spends 22.13M on edu val (4.556 bytes/token), slightly below the plan's 22.27M anchor. Against 22.13M, the blog split is +0.7% at the top rung and the 1-stage is −2.6%.
- Train-mix FLOPs/byte (measurement): the blog split spends 22.39-22.42M at its train-mix ratios (4.00-4.01 x 2.00-2.01), and the 1-stage spends 21.98-22.00M. That is −5.2% and −7.0% vs BPE's 23.64M.
- Losses are finite in all three arms.

### Controller (blog split, train mix, mean ± sd over a window)

| window (steps) | ratio s0 | ratio s1 | lam s0 | lam s1 |
|---|---:|---:|---:|---:|
| 500-4,000 | 4.007 ± 0.117 | 2.008 ± 0.069 | 0.002 | 0.000 |
| 30,000-38,400 | 4.010 ± 0.133 | 1.998 ± 0.060 | 0.007 | 0.000 |
| 100,000-104,000 | 4.017 ± 0.123 | 1.998 ± 0.052 | 0.011 | 0.000 |
| 188,000-192,000 | 4.002 ± 0.115 | 2.006 ± 0.033 | 0.008 | 0.000 |

- The stage-1 integral term stayed at 0.000 ± 0.001 for the whole run. The fixed 0.03 ratio loss alone held stage 1 at 2.00. The LM did not pull stage 1 coarser in this run. This corrects the attempt-1 diagnosis (amendment 03:00): attempt 1's drift needed the old controller to remove the P term first. See the [ratio controller note](../../../.agents/notes/implemented/feature/2026-10-31-ratio-controller.md).
- The 1-stage arm's lam s0 rose to 0.041 by step 100,000: there the LM pulls stage 0 toward finer chunks.

### Main prediction: edu val bpb per rung

| rung (bytes) | blog split 2-stage | BPE | matched 1-stage | blog − BPE | 1-stage − BPE |
|---|---:|---:|---:|---:|---:|
| 0.79B | 1.5498 | 1.2731 | 1.5532 | +0.277 | +0.280 |
| 3.15B | 1.4235 | 1.1769 | 1.2742 | +0.247 | +0.097 |
| **12.58B** | **1.2944** | **1.1156** | 1.1581 | **+0.179** | +0.043 |

At the top rung the blog split is 0.179 bpb *above* BPE, 3x the 0.059 spread.

### Other predictions

- Gap shrinks rung to rung: 0.277 > 0.247 > 0.179. Holds.
- Blog-split main-network use grows: `coarse_shuffle_s1` is 0.0008 at 0.79B, 0.00001 at 3.15B, and 0.0003 at 12.58B. The top rung is *below* the 0.79B rung. Does not hold. Every main-run eval from step 2,000 to 192,000 is under 0.0005, except one eval at 0.004 (step 6,000).

### Observations (not predictions)

- **The 2-stage main network is dead at every scale.** It is 13 x d512, 50% of the arm's FLOPs. Shuffling its output across documents costs nothing (above). Zeroing it costs 0.25-0.37 bpb in the stable phase, so it acts as a learned constant. In the final LR decay the model stops using even the constant: `coarse_zero_s1` falls from 0.19 at step 154,000 to 0.017 at step 192,000. The stage-0 coarse path is live and grows: `coarse_shuffle_s0` 0.26 → 0.52 → 1.17.
- **The 1-stage arm uses its main network, and nearly catches BPE.** Its `coarse_shuffle_s0` grows to 1.71. It beats the blog split by 0.149 at 3.15B and by 0.136 at 12.58B with half the parameters (26.0M vs 49.4M) at the same FLOPs. On edu it is 0.043 behind BPE at the top rung, inside the seed spread. The 1-stage − BPE gap shrank 0.280 → 0.097 → 0.043.
- **On code the 1-stage arm passes BPE.** Top-rung bpb: Go 0.598 vs BPE 0.654 (−0.056), SQL (eval only) 1.245 vs 1.337 (−0.092). At 3.15B it was behind on both (Go 0.766 vs 0.732, SQL 1.523 vs 1.460). On Go, BPE gets 2.89 bytes/token, so it spends more FLOPs per Go byte than on edu, and it still loses. The blog split is far behind on both (Go 0.808, SQL 1.549).
- Speed (measurement): blog split 54.1 ms/step (1.21 MB/s), 1-stage 53.0 ms/step (1.24 MB/s), BPE 48.3 ms/step (1.45 MB/s).

## Verdict

**Not supported.** At 12.58B training bytes (about 8.4e17 training FLOPs), the blog-split 2-stage H-Net has edu val bpb 1.2944 and BPE has 1.1156: the H-Net is 0.179 worse, 3x the 2-stage seed spread. All gates hold, so this is a clean result.

The plan says: if the gap shrinks at every rung, plan a larger run. The gap does shrink, but I do not follow that branch. The hypothesis needed the main network to come online with more bytes. Other prediction 2 shows that it does not: at 16x the bytes, the main network still carries no document information. A larger run would scale a model that spends half its compute on a dead module. The plan's other branch ("return to the gradient path into the main network") matches the evidence, and so does the deep-main record's open next step.

## Updated hypothesis

Scale is not what keeps the 2-stage H-Net behind. In our 2-stage model the main network never learns to carry document information, at 0.79B bytes or at 12.58B. Three experiments now show this: deep-main split (both arms), ladder attempt 1, and this run. The 2-stage arm is in effect a stage-0 H-Net with a 2x-strided stage-1 block stack and a dead 13-layer tail. The tail costs half the FLOPs. That explains why the 2-stage arm loses to a 1-stage arm with half the parameters at matched FLOPs.

The 1-stage result says the byte-level idea itself works at this compute. The 1-stage H-Net uses its main network, closes the edu gap to within seed noise, and passes BPE on Go and SQL by 12.58B bytes. That matches Cartesia's Part I ordering on code, where byte models gain the most.

What stops the 2-stage main network is a property of the second dechunk path, not of compute. Candidate causes, cheapest test first:

1. Gradient into the 2-stage main network is too small (two EMA dechunks + STE in series). Measure per-level gradient norms and update/weight ratios from step 1, for the 1-stage and 2-stage configs. No new arm is needed.
2. The stage-1 residual path wins at once. The stage-1 decoder gets the stage-1 encoder output through the residual, and that path can do the job alone. Test stage-1 residual dropout, or zero the residual early in training.
3. Main LR multiplier. Cartesia scales LR per stage. Test a larger main-network multiplier.

Next planned experiment: **2-stage main-network gradient path** (`lab/experiments/planned/`, to be written before any run).

## Related

- [Deep-main compute split](../refuted/2026-10-31-deep-main-compute-split.md)
- [Byte and BPE controls](../refuted/2026-10-31-byte-and-bpe-controls.md)
- [Ratio controller note](../../../.agents/notes/implemented/feature/2026-10-31-ratio-controller.md)
- [Constitution](../../../docs/constitution.md), Success criterion 1.
