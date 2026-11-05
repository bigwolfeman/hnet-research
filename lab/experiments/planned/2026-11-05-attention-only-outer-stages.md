# Experiment: attention-only outer stages

Status: planned

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

## Related

- [Reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md)
- [Learned vs entropy boundaries](2026-11-05-learned-vs-entropy-boundaries.md) (its seed 1 is postponed behind this run)
