# Experiment: our H-Net with Mamba-2

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (a block-type swap through stage layouts, `hnet/blocks.py`; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

With Mamba-2 encoders and decoders and the paper's per-stage LR, does our own 2-stage H-Net get the reference's live main network?

Context: in the reference model, Transformer outer stages and our per-stage LR each nearly killed the main network ([difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md)). In our model, the paper's LR alone gave 10x more main-network use and −0.079 edu bpb, but the decoder still read more of a constant than of content: shuffle − zero = −0.176 at stage 1 ([paper LR in our H-Net](../refuted/2026-11-02-paper-lr-in-our-hnet.md)). This run swaps the other cause: our outer stages get Mamba-2 mixers.

## Hypothesis

Both causes have to go. The paper's LR lets the encoder learn fast relative to the main network. Mamba-2 gives each chunk boundary a state that summarizes the whole chunk, so the main network's input carries document content. With both, the decoder learns to read the main network's content, and its constant part shrinks, as in the reference. With Mamba-2 and our LR, the encoder still learns too slowly, and the main network stays mostly a constant (the reference with our LR: `coarse_shuffle_s1` 0.0078).

## Predictions

Main prediction (only this one answers the Question):

- At the end of the run (12,000 steps, 786.4M bytes), the Mamba-2 + paper-LR arm has the live stage-1 signature: `probe/coarse_shuffle_s1` above `probe/coarse_zero_s1` (anchor: 0 for shuffle − zero). Above: supported. At or below: not supported. Controls: Transformer outer stages with the paper's LR −0.176, with our LR −0.092; the reference arm +0.021 and +0.031 (two seeds).

Other predictions (at most two):

- The Mamba-2 + our-LR arm keeps the dead stage-1 signature (shuffle below zero).
- The Mamba-2 + paper-LR arm has lower edu bpb than the Transformer + paper-LR arm (1.4511) by more than the 2-stage seed spread (0.059).

If supported, we do: our 2-stage design takes Mamba-2 outer stages and the Eq. 11 LR (Agent Note). Then the BPE comparison runs again on the ladder with the new design, at least to 3.15B bytes, against the BPE and 1-stage rungs we already have.

If not supported, we do: the rest of the gap is in our model code, not in block type or LR. The next candidates are the router (our lookahead router against the reference's cosine router with identity q/k init), the dechunk, and the residual path. Swap them one at a time from the reference into our model, with Mamba-2 and the paper's LR in place.

Gates (the run is broken if one fails; gates are not predictions):

- Tests pass: our suite with `HNET_TEST_CUDA=1` (143 passed, 1 skipped), including `tests/test_layouts.py` (Mamba-2 causality, pad safety, finite gradients). The T-only blog split built by the new code is bit-equal to the committed code (parameters, logits, FLOPs; checked 2026-11-02, see the paper-LR experiment's amendment).
- Edu ratios within 15% of the targets (4 x 2) at the end; edu fwd FLOPs/byte within 5% of BPE's 22.27M.
- Training bytes equal across arms (786,432,000). Finite loss.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file; our model (model.kind hnet)
Stack(layout): blocks from a layout string (hnet/blocks.py parse_layout): T = attention + SwiGLU, m = Mamba-2,
               M = Mamba-2 + SwiGLU. Mamba-2 = mamba_ssm Mamba2 (hnet/ssm.py), model.ssm = the reference values
               (d_state 128, d_conv 4, expand 2, headdim 64, chunk_size 256)
arms (seed 0), each = blog_split_2stage with stage 0 enc/dec M2 (was T2), stage 1 enc/dec m1 (was T1):
  mamba         blog_split_2stage_mamba          lr 8e-4 x [1.5, 1.2, 1.0]
  mamba_eq11lr  blog_split_2stage_mamba_eq11lr   lr 6.25e-4 x [3.03, 1.24, 0.76] (Eq. 11)
  fwd FLOPs/byte at 4 x 2: 22.31M (+0.2% vs BPE; blog split 22.44M). Split stage0/stage1/main 42/6/51%. 51.1M params.
  Stage 1 loses its SwiGLU (m1, not M1): with M1 the arm costs +5.8% FLOPs/byte, over the 5% gate.
controls (not rerun): the two arms of the paper-LR experiment (Transformer outer stages, our LR / paper LR, seed 0),
  same schedule, data stream and seed
both: RatioController on (ratio_integral_lr 0.002); configs/train/ladder.yaml with max_steps 12,000, no branches
eval every 2,000 steps: val bpb per domain, ratios, FLOPs, coarse-path probe (8 edu probe batches)
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 12,000
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `TRAIN=ladder scripts/run_arms.sh ignored/runs/our-hnet-with-mamba2 our-hnet-with-mamba2 blog_split_2stage_mamba_eq11lr:0 blog_split_2stage_mamba:0 -- train.max_steps=12000 "train.ladder_fracs=[]"`. wandb project `hnet-router`, group `our-hnet-with-mamba2`.
- Environment: the spike venv now has mamba_ssm @ a6a1dae (MAMBA_SKIP_CUDA_BUILD) and causal_conv1d built for sm_120, the recipe of the [reference probe spike](../../spikes/2026-11-01-reference-hnet-probe/README.md). A stub stands in for the unused Mamba-1 CUDA module (`hnet/ssm.py`).
- Differences from the reference that stay: our router (lookahead), dechunk, residual path, and init; Mamba-2 blocks are not torch.compiled (their MLPs are).
- Cost: about 15 to 25 min per arm. Speed is a measurement: Mamba-2 at d 256 ran launch-bound in the reference arm (113 ms/step).

## Results

Both arms ran to 12,000 steps (commit 8576835). Log dir `ignored/runs/our-hnet-with-mamba2`; wandb mamba_eq11lr `ez2e2whb`, mamba `5vyg9zl2`. Artifacts: [results/](../results/2026-11-02-our-hnet-with-mamba2/) (`*-summary.json`, `eval-curves.csv`).

### Gates

| gate | mamba_eq11lr | mamba | result |
|---|---|---|---|
| tests | 143 passed, 1 skipped (our venv); T-only model bit-equal to the old code | | pass |
| edu ratios at the end (4 x 2 ±15%) | 4.00 x 2.02 | 4.04 x 1.99 | pass |
| edu fwd FLOPs/byte at the end (22.27M ±5%) | 22.18M | 22.25M | pass |
| training bytes, finite loss | 786,432,000, yes | same | pass |

### Coarse-path probe and bpb at the end (12,000 steps), with the Transformer-outer controls

| arm | `shuffle_s1` | `zero_s1` | shuffle − zero | `shuffle_s0` | `zero_s0` | edu bpb | Go bpb | SQL bpb | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mamba-2 + paper LR | 0.0026 | 0.2508 | **−0.248** | 0.336 | 0.297 | **1.4406** | 1.064 | 2.328 | 75 |
| Mamba-2 + our LR | 0.00002 | 0.2197 | −0.220 | 0.313 | 0.259 | 1.4532 | 1.071 | 2.328 | 75 |
| Transformer + paper LR (control) | 0.1115 | 0.2872 | −0.176 | 0.969 | 1.026 | 1.4511 | 0.970 | 1.802 | 55 |
| Transformer + our LR (control) | 0.0107 | 0.1030 | −0.092 | 0.289 | 4.278 | 1.5304 | 1.019 | 1.854 | 55 |

`coarse_shuffle_s1` by eval (steps 2,000 to 12,000): Mamba-2 + paper LR 0.0001, 0.0003, 0.0011, 0.0019, 0.0020, 0.0026; Mamba-2 + our LR at or below 0.0001 at every eval.

- Main prediction: shuffle − zero at stage 1 is −0.248 for the Mamba-2 + paper-LR arm. **Not supported.**
- Other prediction 1 (Mamba-2 + our LR keeps the dead signature): holds (−0.220).
- Other prediction 2 (Mamba-2 + paper LR lower than 1.4511 by more than 0.059): does not hold. It is 0.010 lower, inside the spread.

### What the arms show

- **Mamba-2 outer stages in our model make the main network more dead, not less.** With the paper's LR, `shuffle_s1` falls from 0.112 (Transformer) to 0.0026, a 43x drop. With our LR it is 0.00002. This is the opposite of the reference, where Transformer outer stages killed a live main network.
- **The stage-0 coarse path also carries less.** Zeroing it costs 0.26 to 0.30, against 1.03 to 4.28 for the Transformer arms. The Mamba-2 decoders predict bytes mostly from their own residual.
- **Edu bpb gets better and code gets worse.** With our LR, edu is 0.077 lower than the Transformer control. With the paper's LR it ties (−0.010). Go is worse by 0.05 to 0.09, and SQL by 0.47 to 0.53.
- So the effect of the block type depends on the rest of the design. In the reference layout, Mamba-2 is needed for a live main network. In our layout, it removes the use that the paper's LR had brought back.

## Verdict

**Not supported.** Mamba-2 encoders and decoders with the paper's LR do not give our 2-stage H-Net a live main network. The main network becomes more dead than with Transformer outer stages (`coarse_shuffle_s1` 0.0026 against 0.112).

## Updated hypothesis

Our router is the reference's (cosine of adjacent q/k projections, identity init, start anchors), and so are the residual path and the EMA dechunk. What remains between our model and the reference is mostly configuration: widths (ours 256/384/512, reference 256/256/512), block layout (reference stage 0 `m4` with no MLP, stage 1 `T1m4` / `m4T1`, main `T10`; ours `T2` / `T1` / `T13`), the compute split (ours 39/10/50%, reference 44/22/34%), targets, and LR. The next run builds the reference's exact layout, widths, targets, and LR in our code: [reference layout in our code](2026-11-02-reference-layout-in-our-code.md). If its main network is live, the cause is in the configuration, and we bisect it there. If it is dead, the cause is in an implementation detail, and we audit our code against the reference line by line.

## Related

- [Paper LR in our H-Net](../refuted/2026-11-02-paper-lr-in-our-hnet.md)
- [Reference H-Net difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md)
- [Reference H-Net at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md)
