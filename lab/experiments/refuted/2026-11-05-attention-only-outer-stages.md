# Experiment: attention-only outer stages

Status: refuted

Date proposed: 2026-11-05
Algorithm: none (a layout config: `lab/spikes/2026-10-30-two-stage-hnet-smoke/configs/experiment/attn_2stage.yaml`)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

With no SSM, does our 2-stage H-Net still beat the matched-FLOPs BPE Transformer on edu val bpb at 12.58B bytes? Every Mamba-2 block of the outer stages becomes a sliding-window attention block, at the same FLOPs per byte.

Context: [reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md) is supported with Mamba-2 outer stages: H-Net 1.0905, 1.0979, 1.0913 against BPE 1.1156, 1.1109. But Mamba-2 kernels take 31 of 71 ms per step for about 12% of the counted FLOPs, so the H-Net trains at 1.56x BPE's wall clock. The user's final design is likely not Mamba-based. This test asks whether the H-Net gain needs the SSM, and measures what the step costs without it.

## Hypothesis

The gain comes from the dynamic chunking and the hierarchy, not from the SSM. Sliding-window attention at stage 0 (704 bytes of context per byte) gives the router and the dechunker what they need. Attention without an MLP has about 8% fewer params than the m blocks (38.55M against 41.80M) at the same FLOPs, and the SSM's inductive bias for byte-level compression is gone. So we expect the attention H-Net to lose some of the lead to the Mamba H-Net, but to stay below BPE.

## Predictions

Arms: `attn_2stage` seed 0 (new). Controls: BPE Transformer seeds 0, 1 (1.1156, 1.1109) and reflayout_2stage seeds 0-2 (1.0905, 1.0979, 1.0913). None of them is rerun.

Main prediction (only this one answers the Question):

- At 12.58B bytes, edu val bpb of attn_2stage seed 0 is lower than both BPE seeds. Below 1.1109: supported. At or above 1.1109: not supported.

Other predictions (at most two):

- At 12.58B bytes, attn_2stage edu val bpb is above every reflayout_2stage seed (the SSM helps at matched FLOPs).
- At 12.58B bytes, attn_2stage Go and SQL bpb are lower than in both BPE seeds.

Measurement (not a prediction): training ms/step and bytes/s, against reflayout_2stage seed 2 (74.0 ms, 886 kB/s) and BPE seed 1 (50.5 ms, 1,384 kB/s), same eval schedule.

If supported, we do: the H-Net gain does not need Mamba-2 at our scale. Phase 2 can build on attention outer stages, and the Mamba-2 kernel work drops in priority. Add a second seed only if the result lands inside the H-Net seed spread (0.007) of BPE seed 1.

If not supported, we do: read the rungs and the probes (sem/word_s0, coarse_shuffle_s0, the crossing against BPE). If the router never comes online (shuffle_s0 near 0), the attention outer stages fail at chunking and the fix is in stage 0 (window, a local conv, or one m block). If it comes online but ends above BPE, Mamba-2 is part of the gain at this scale, and the Mamba-2 kernel work becomes the throughput path.

Gates (the run is broken if one fails; gates are not predictions):

- Suite: passed in the learned-vs-entropy queue (`HNET_TEST_CUDA=1`, 2026-11-05) on the same model code. This run changes only configs.
- Smoke: 200 steps train (done 2026-11-05 14:59, params 38,552,832).
- Edu ratios within 15% of 3 x 3 at every rung; edu fwd FLOPs/byte within 5% of BPE's 22.27M at every rung.
- Training bytes 12,582,912,000 (192,000 x 65,536). Finite loss.

## Method

```
code:    lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
arm:     attn_2stage seed 0 = reflayout_2stage with layouts t4 | T1t4 | T10 | t4T1 | t4
         t = sliding-window attention, no MLP (4 d^2 params); windows 704 (stage 0), 1024 (stage 1); main unchanged
         FLOPs match: t4 at w 704, L 4096 = 0.994x of m4; t4 at w 1024, L 1300 = 0.975x of m4 (hnet/flops.py)
budget:  configs/train/ladder.yaml: 12.58B bytes, rungs at 1/16 and 1/4, the main run decays over its last 20%
grade:   val/edu/bpb at the end of the run against BPE seeds 0, 1
```

- Launch from the spike dir: `TRAIN=ladder scripts/run_arms.sh ignored/runs/attention-only-outer-stages/arms attention-only-outer-stages attn_2stage:0`.
- The run stops before 16:00 on 2026-11-05 (the GPU is needed) and resumes later from `resume.pt` with the same command (train.py checks that the val bpb matches after the resume).
- wandb project `hnet-router`, group `attention-only-outer-stages`.
- Cost: unknown before the run; the smoke ran at 56 ms/step at step 200.

## Results

Run: attn_2stage seed 0, wandb zy00urcv, run dir `ignored/runs/attention-only-outer-stages/arms/attn_2stage_s0/`. Code: the commit that adds this file (configs only). It trained in three parts:

- 2026-11-05 15:00 to 15:49. Stopped by Claude for the user's GPU, after the step-40,000 `resume.pt`.
- 2026-11-06 00:29 to 00:35. An outside SIGKILL at step 44,350. The record does not say what sent it. `resume.pt` stayed at step 40,000.
- 2026-11-06 02:36 to 04:54, to the end.

Both resumes passed the check: val bpb 1.4192 at step 40,000, equal to the value before the first stop.

### Gates

- Suite and smoke as in Gates (params 38,552,832).
- Edu ratios over all 100 evals: stage 0 2.97 to 3.13, stage 1 2.64 to 3.27 (inside 2.55 to 3.45). Edu fwd FLOPs/byte 21.65M to 23.34M against BPE 22.27M (-2.8% to +4.8%).
- Training bytes 12,582,912,000 (192,000 x 65,536). Finite loss.

### Main prediction: edu val bpb at 12.58B bytes

| arm | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| attn_2stage | **1.1764** | | |
| BPE Transformer | 1.1156 | 1.1109 | |
| reflayout_2stage (Mamba-2) | 1.0905 | 1.0979 | 1.0913 |

The attention H-Net is 0.066 above the best BPE seed and 0.079 above the worst Mamba-2 seed.

### Other predictions

- Above every reflayout_2stage seed: **holds** (1.1764 against 1.0905 to 1.0979).
- Go and SQL below both BPE seeds: **does not hold**. Go 0.676 against BPE 0.654, 0.650. SQL 1.384 against 1.337, 1.318.

### Readings

- The router comes online. coarse_shuffle_s0 rise +2.25 bpb and coarse_zero_s0 +1.92. Stage-0 cuts find words better than in the Mamba-2 arm: sem/word_s0 AUROC 0.93 (Mamba-2 seed 2: 0.91), sem/gotok_s0 0.97, alignment F1 to word starts 0.68 against 0.31 for a fixed stride.
- The gap to the Mamba-2 seeds closes with training but does not reach 0. Against the Mamba-2 seed mean: 0.20 at the 0.79B rung (1.5301 against 1.3326), 0.10 at the 3.15B rung (1.2889 against 1.1890), 0.083 at the end.
- Measurement: 53.3 ms/step and 1,230 kB/s, with evals and probes. That is 1.13x BPE seed 1's wall clock per byte (1,384 kB/s), against 1.56x for Mamba-2 seed 2 (886 kB/s). Peak memory 8.2 GB.

## Verdict

**Not supported.** At 12.58B bytes, attn_2stage seed 0 edu val bpb (1.1764) is above both BPE seeds (1.1156, 1.1109).

## Updated hypothesis

The plan's not-supported branch reads: the router comes online but the run ends above BPE, so "Mamba-2 is part of the gain at this scale". That reading is too narrow, because the arm changed more than the sequence mixer.

- An `m` block is a full Mamba-2 mixer: an expand-2 input projection, a causal conv of width 4, the SSM, and a SiLU gate on the output. It has about 7 d^2 params.
- A `t` block is attention with no MLP: 4 d^2 params. Its only nonlinearity is the softmax. Each layer is a weighted sum of linear value projections.

So the arm removed three things at once: the SSM, the short conv, and every gated nonlinear channel map in the outer stages, 16 of 28 blocks. The test shows that MLP-free attention is not a good outer stage. It does not show that the outer stages need an SSM. The router result points the same way. Chunking works as well as or better than with Mamba-2, so the loss is in what the outer stages compute, not in where they cut.

Next test, for a design with no SSM: attention plus a gated MLP (`T` blocks) in the outer stages, at matched FLOPs (fewer blocks or a shorter window). Its main prediction is again edu val bpb below both BPE seeds. If it holds, Phase 2 builds on that layout. If it fails, the short causal conv is the next ingredient to add back.

## Related

- [Reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md)
- [Learned vs entropy boundaries](../successes/2026-11-05-learned-vs-entropy-boundaries.md) (its seed 1 ran after this run)
