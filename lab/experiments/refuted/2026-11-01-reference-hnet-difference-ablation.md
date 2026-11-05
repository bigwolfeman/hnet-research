# Experiment: reference H-Net difference ablation

Status: refuted

Date proposed: 2026-11-01
Algorithm: none (config swaps in the reference H-Net; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Which difference between the reference H-Net and our 2-stage H-Net decides whether the main network is used?

Context: in [reference H-Net at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md) attempt 1, the reference design trained on our data, batch, schedule, and budget, and its main network was in use from step 2,000 (`coarse_shuffle_s1` 0.039, rising to 0.53). Our blog split never passed 0.0041. Attempt 2 (with the RatioController, FLOPs matched) confirmed it: supported, `coarse_shuffle_s1` 0.135 at the 0.79B rung and 0.460 at 3.15B.

Three recipe differences can be swapped inside the reference code, one at a time:

1. Outer stages: the reference encoders and decoders are Mamba-2 (`m4`, `T1m4`); ours are Transformer blocks.
2. Per-stage LR: the reference uses 6.25e-4 x [3.03, 1.75, 0.72] (the main network gets the lowest LR); ours is 8e-4 x [1.5, 1.2, 1.0] (the main network gets 1.8x the reference value, the outer stages less).
3. Chunk targets: the reference uses 3 x 3; ours uses 4 x 2.

The remaining differences (router form, dechunk, norms, residual path, init) live in our model code and cannot be swapped by config. They are what remains if none of the three swaps kills the main network.

## Hypothesis

The Mamba-2 outer stages make the main network useful. A recurrent encoder writes a summary of the whole chunk into the state at each chunk boundary, so the main network gets an informative input from the first steps of training, and the decoder learns to read the main network's output. A windowed Transformer encoder gives each boundary position only a local attention mix. Then the decoder can predict almost as well from its own byte-level residual, and the main network becomes a constant. The H-Net paper's ablation points the same way: Transformer-only encoders and decoders were "by far the worst".

## Predictions

Main prediction (only this one answers the Question):

- At the end of the run (12,000 steps, 786.4M bytes), the Transformer-outer arm has `probe/coarse_shuffle_s1` at or below 0.0041, the largest value our blog split showed at any eval up to 3.15B bytes (its main network was dead). At or below: supported. Above: not supported.

Other predictions (at most two):

- The Transformer-outer arm has the lowest `coarse_shuffle_s1` of the four arms at the end of the run.
- The our-LR arm and the 4 x 2 arm both end above 0.0041 (their main networks stay in use).

If supported, we do: the outer-stage block type is the cause. Port Mamba-2 encoders and decoders into our Stage code (or make the reference model our 2-stage base, after a review with the user), then rerun the BPE comparison at 3.15B bytes with the fixed design.

If not supported, we do: read the other two arms. If one of them ends at or below 0.0041, that difference is the cause, and the next run confirms it in our code. If every arm stays above 0.0041, none of the three config differences is the cause. Then the cause is in our model code (router, dechunk, norms, residual path, init), and the next experiment moves reference pieces into our code one at a time.

Gates (the run is broken if one fails; gates are not predictions):

- The control arm (reference config, seed 1) ends above 0.0041. If it does not, the 12,000-step run is too short to show a live main network, and the test cannot tell.
- Each arm's edu ratios are within 15% of its targets at the end. Fwd FLOPs/byte at the end is within 10% of BPE's 22.27M.
- Training bytes are equal across arms (786,432,000). Finite loss.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file; model.kind ref_hnet (hnet/refmodel.py)
every arm = ref_hnet_2stage (configs/experiment/ref_hnet_2stage.yaml) + RatioController (train.ratio_integral_lr=0.002),
            then exactly one swap:
  control  ref_hnet_2stage, seed 1                                   (seed 0 = the 0.79B rung of the at-our-scale rerun)
  touter   ref_hnet_2stage_touter: layout T2 | T1T2 | T10 | T2T1 | T2, d_intermediate [640, 768, 1408]
           (each m4 -> T2: attention + SwiGLU at the reference windows); 22.73M fwd FLOPs/byte at 3 x 3
  ourlr    ref_hnet_2stage_ourlr: lr 8e-4, lr_mult [1.5, 1.2, 1.0]
  t4x2     ref_hnet_2stage_t4x2: targets 4 x 2; 22.32M fwd FLOPs/byte at 4 x 2
schedule: configs/train/ladder.yaml with max_steps 12,000 and no ladder branches:
          warmup 500, stable, linear decay over the last 20% (steps 9,600 to 12,000)
          = the schedule of the 0.79B rung of every ladder run
eval every 2,000 steps: val bpb per domain, ratios, FLOPs, coarse-path probe (8 edu probe batches)
grade: probe/coarse_shuffle_s1 at step 12,000 (the run's final eval)
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `PY=../2026-11-01-reference-hnet-probe/.venv/bin/python TRAIN=ladder scripts/run_arms.sh ignored/runs/reference-hnet-difference-ablation reference-hnet-difference-ablation ref_hnet_2stage_touter:0 ref_hnet_2stage_ourlr:0 ref_hnet_2stage_t4x2:0 ref_hnet_2stage:1 -- train.max_steps=12000 "train.ladder_fracs=[]" train.ratio_integral_lr=0.002`. wandb project `hnet-router`, group `reference-hnet-difference-ablation`.
- Controls: the control arm (seed 1), and the 0.79B rung of the at-our-scale rerun (seed 0). Together they give two seeds of the unchanged reference arm. The anchor 0.0041 comes from our blog split ([compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md), attempt 2, wandb `5rvc25h5`).
- Cost: about 25 min per arm at 105-130 ms/step, about 1.7 h in total. The GPU draws 550-575 W with the reference model (over the power budget), so the CPU stays idle during the runs.
- Why the swaps go in the reference code, not ours: the reference arm is the one with a live main network. Removing one reference feature at a time from a live model gives a clean "this kills it" signal. Adding features to our dead model one at a time needs a port first (Mamba-2 into our Stage), and the port is only worth doing once we know which feature matters.

- Amendment 2026-11-01 17:45: the first launch (wandb `1i6h3rb4`, touter arm) was stopped at the user's request after about 15 minutes. It had logged evals at steps 2,000 to 6,000 to wandb; I did not read them before the rerun (correction 2026-11-02: this line first said it stopped before its first eval). The log dir moved to `ignored/runs/reference-hnet-difference-ablation.stopped-2026-11-01`. The run starts again from step 0 with the launch command above. No ablation data was seen before the rerun.
- Run (2026-11-02 03:49 to 05:22): log dir `ignored/runs/reference-hnet-difference-ablation`, code at commit 48561a1. wandb: touter `5sxt7zxb`, ourlr `4ghq3i9a`, t4x2 `ph134yv7`, control seed 1 `u8hqkojb`.

## Results

All four arms ran to 12,000 steps with no restart. Artifacts: [results/](../results/2026-11-01-reference-hnet-difference-ablation/) (`*-summary.json` per arm, `eval-curves.csv` with every eval of every arm).

### Gates

| gate | touter | ourlr | t4x2 | control (seed 1) | result |
|---|---|---|---|---|---|
| edu ratios at the end (targets ±15%) | 3.02 x 2.94 | 3.03 x 2.98 | 4.11 x 1.95 | 3.03 x 2.94 | pass |
| edu fwd FLOPs/byte at the end (BPE 22.27M ±10%) | 22.80M | 22.49M | 22.20M | 22.61M | pass |
| control `coarse_shuffle_s1` above 0.0041 | | | | 0.132 | pass |
| training bytes, finite loss | 786,432,000, yes | same | same | same | pass |

### Coarse-path probe at the end (12,000 steps, 786.4M bytes)

| arm (one change from the reference) | `shuffle_s1` | `zero_s1` | `shuffle_s0` | `zero_s0` | edu bpb | Go bpb | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| control, seed 1 (no change) | 0.132 | 0.101 | 1.048 | 1.021 | 1.3185 | 0.890 | 104 |
| control, seed 0 (at-our-scale rerun, 0.79B rung) | 0.135 | 0.114 | 1.041 | 0.917 | 1.3178 | 0.884 | |
| t4x2: targets 4 x 2 | 0.152 | 0.093 | 0.889 | 0.996 | 1.3321 | 0.916 | 101 |
| ourlr: our per-stage LR | **0.0078** | 0.170 | 1.011 | 1.856 | 1.3355 | 0.893 | 104 |
| touter: Transformer outer stages | **0.0082** | 0.265 | 0.541 | 3.682 | 1.4684 | 0.931 | 135 |
| our blog split (0.79B rung, [ladder](../refuted/2026-11-01-compute-ladder-vs-bpe.md)) | 0.0008 | 0.156 | 0.256 | 3.513 | 1.5498 | | 52 |

- The two seeds of the unchanged reference agree closely: `shuffle_s1` 0.132 and 0.135, edu bpb 1.3185 and 1.3178.
- Main prediction: the touter arm ends at 0.0082, above 0.0041. **Not supported.**
- Other prediction 1 (touter lowest): does not hold. ourlr is lower (0.0078 vs 0.0082). The two values are close.
- Other prediction 2 (ourlr and t4x2 above 0.0041): holds as written, but the ourlr half holds only on the letter. ourlr's main network is 17x less used than the control's.

### What the arms show

- **Two differences each cut main-network use about 16x. The third does nothing.** At step 12,000, touter and ourlr are at 0.008, the two controls at 0.13, and t4x2 at 0.15. Over the run, touter and ourlr rise slowly (0.0035 → 0.0082 and 0.0034 → 0.0078); the controls and t4x2 rise fast (0.017 → 0.132, 0.029 → 0.152).
- **Both swaps flip the stage-1 signature to the dead pattern.** In the control and t4x2, shuffle is above zero (the decoder reads main-network content). In touter and ourlr, zero is 20-30x larger than shuffle. The decoder uses the main network's output as a near-constant, like our blog split (0.156 vs 0.0008).
- **touter also copies our blog split at stage 0.** Its `zero_s0` 3.68 and `shuffle_s0` 0.54 look like the blog split's 3.51 and 0.26. Every other reference arm has shuffle near or above zero at stage 0. It also costs the most bpb: +0.150 against the control at 786M bytes.
- **ourlr costs little bpb at this budget** (+0.017). At 786M bytes the main network adds little bpb even in the live arms. The main-network effect on bpb shows later (the at-our-scale run: 0.135 at 0.79B, 0.460 at 3.15B, and a tie with BPE).
- **The 0.0041 line was the wrong anchor for one swap at a time.** It is the maximum of a model that has both differences and more. One swap moves a live model most of the way to the dead pattern, but not past the line. The sign of shuffle − zero separates the live and dead arms here with no threshold: it is positive in the control, t4x2, the at-our-scale run, and the released checkpoint, and negative in touter, ourlr, and every blog-split eval.

## Verdict

**Not supported.** The Transformer-outer arm ended at `coarse_shuffle_s1` 0.0082, above the 0.0041 line. No single config difference makes the reference main network as dead as ours. Two of them, the outer-stage block type and the per-stage LR, each cut its use about 16x and flip the stage-1 signature to the dead pattern. The chunk targets do not matter.

The plan's "not supported" branch said: if every arm stays above 0.0041, the cause is in our model code. That branch does not fit the data. Two config swaps nearly kill the main network, so the config differences are causes. They are partial causes, and our model has both of them.

## Updated hypothesis

The main network gets used only when the outer encoder gives it informative chunk inputs early in training. Two things in the reference make the encoder strong early: Mamba-2 blocks (a recurrent state that summarizes the chunk at each boundary) and a high outer-stage LR (the paper's Eq. 11 gives stage 0 3.03x and the main network 0.72x; ours gives 1.5x and 1.0x). Removing either one leaves a weak encoder, the decoder learns to predict from its own byte residual, and the main network becomes a constant. Our blog split has both weaknesses.

Next: test the fix in our own code. A planned experiment gives our blog split the paper's Eq. 11 per-stage LR (a config change), graded by the sign of shuffle − zero at stage 1. After that, Mamba-2 encoders and decoders go into our Stage code.

## Related

- [Reference H-Net at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md)
- [Reference checkpoint main-network use](../successes/2026-11-01-reference-checkpoint-main-use.md)
- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
- [Reference H-Net adapter](../../../.agents/notes/implemented/feature/2026-11-01-reference-hnet-adapter.md)
