# Experiment: paper LR in our H-Net

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (a config change; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does the H-Net paper's per-stage LR (Eq. 11) wake the main network of our own 2-stage H-Net?

Context: in the [difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md), giving the live reference model our per-stage LR cut its main-network use about 16x (`coarse_shuffle_s1` 0.132 → 0.0078) and flipped the stage-1 signature to the dead pattern (zero above shuffle). Swapping its Mamba-2 outer stages for Transformer blocks did the same. Our blog split has both differences. The LR is the cheaper one to fix: a config change, no new code.

## Hypothesis

The main network is used only when the outer encoder becomes informative early in training. The paper's Eq. 11 gives the outer stages a high LR and the main network a low one (here stage 0 3.03x, stage 1 1.24x, main 0.76x of 6.25e-4). Our LR gives the outer stages less and the main network more (1.5x, 1.2x, 1.0x of 8e-4). With the paper's LR, our Transformer encoder learns faster relative to the main network, the chunk inputs carry document content sooner, and the stage-1 decoder starts to read the main network's output.

## Predictions

Main prediction (only this one answers the Question):

- At the end of the run (12,000 steps, 786.4M bytes), the paper-LR arm has the live stage-1 signature: `probe/coarse_shuffle_s1` above `probe/coarse_zero_s1` (anchor: 0 for shuffle − zero). Above: supported. At or below: not supported. Every one of the 96 blog-split evals in the [ladder](../refuted/2026-11-01-compute-ladder-vs-bpe.md) had shuffle below zero. Every live reference run and the released checkpoint had shuffle above zero.

Other predictions (at most two):

- The paper-LR arm ends with a larger `coarse_shuffle_s1` than the control arm.
- The paper-LR arm's edu val bpb at the end is not worse than the control's by more than the 2-stage seed spread (0.059).

If supported, we do: the per-stage LR is a fix in our code. Eq. 11 becomes the default LR rule for multi-stage arms (Agent Note), and the next run checks it at 3.15B bytes against BPE. The Mamba-2 port then tests whether it adds more.

If not supported, we do: the LR alone does not fix our model, because our Transformer outer stages still hold it dead (the ablation showed that each difference is enough on its own). Port Mamba-2 encoders and decoders into our Stage code, and test that port with the Eq. 11 LR.

Gates (the run is broken if one fails; gates are not predictions):

- Edu ratios within 15% of the targets (4 x 2) at the end; edu fwd FLOPs/byte within 5% of BPE's 22.27M.
- The control arm reproduces the dead pattern: shuffle below zero at stage 1 at the end. If it does not, the 12,000-step control differs from the ladder's 0.79B rung, and the comparison cannot tell.
- Training bytes equal across arms (786,432,000). Finite loss.

## Method

```
code: lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file; our model (model.kind hnet)
arms (same seed 0, same data stream):
  control   blog_split_2stage                (lr 8e-4 x [1.5, 1.2, 1.0])
  eq11lr    blog_split_2stage_eq11lr         (lr 6.25e-4 x [3.03, 1.24, 0.76]: Eq. 11 with N_GPT 4.6, targets 4, 2,
                                              D 256, 384, 512)
both: RatioController on (ratio_integral_lr 0.002, from blog_split_2stage.yaml)
schedule: configs/train/ladder.yaml, max_steps 12,000, no ladder branches: warmup 500, stable, linear decay
          over the last 20% (the schedule of every ladder run's 0.79B rung)
eval every 2,000 steps: val bpb per domain, ratios, FLOPs, coarse-path probe (8 edu probe batches)
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 12,000
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `TRAIN=ladder scripts/run_arms.sh ignored/runs/paper-lr-in-our-hnet paper-lr-in-our-hnet blog_split_2stage_eq11lr:0 blog_split_2stage:0 -- train.max_steps=12000 "train.ladder_fracs=[]"` (the spike's own venv). wandb project `hnet-router`, group `paper-lr-in-our-hnet`.
- Controls: the control arm, and the 0.79B rung of the ladder's blog split (wandb `5rvc25h5`: shuffle 0.0008, zero 0.156, edu bpb 1.5498).
- Cost: about 15 min per arm (52 ms/step plus compile and evals).
- The stage-0 LR rises to 1.89e-3, 1.6x our current stage-0 LR. Gradient clip 1.0 stays on.

### Amendment 2026-11-02 06:10 (after the run; code version of the control arm)

- The eq11lr arm started at 05:25 on commit f1fb6e4. The control arm started at 05:37, after I had edited `hnet/blocks.py`, `hnet/model.py`, `hnet/flops.py` and `configs/model/base.yaml` (05:29 to 05:30) for the Mamba-2 port. So the control ran on uncommitted code.
- Check: the blog-split model built from f1fb6e4 and from the edited code (seed 0, the same 4 x 4096 input, eager, bf16 autocast) has bit-equal parameters, bit-equal logits, equal chunk counts and equal FLOPs/byte (22,439,904). The only difference was the order of the state_dict keys; the code now registers modules in the old order, and the check gives equal key order too. The control arm is valid. Script: [results/code-equivalence-dump.py](../results/2026-11-02-paper-lr-in-our-hnet/code-equivalence-dump.py) (run once per code version with that version first on `sys.path`, then compare the two dumps).

## Results

Both arms ran to 12,000 steps. Log dir `ignored/runs/paper-lr-in-our-hnet`; wandb eq11lr `gjwtn28f`, control `2orl5dsq`. Artifacts: [results/](../results/2026-11-02-paper-lr-in-our-hnet/) (`*-summary.json`, `eval-curves.csv`).

### Gates

| gate | eq11lr | control | result |
|---|---|---|---|
| edu ratios at the end (4 x 2 ±15%) | 4.01 x 1.96 | 4.07 x 1.99 | pass |
| edu fwd FLOPs/byte at the end (22.27M ±5%) | 22.68M (+1.8%) | 22.26M (0.0%) | pass |
| control has the dead pattern (shuffle < zero at stage 1) | | 0.0107 < 0.103 | pass |
| training bytes, finite loss | 786,432,000, yes | same | pass |

### Coarse-path probe and bpb at the end (12,000 steps)

| arm | `shuffle_s1` | `zero_s1` | shuffle − zero | `shuffle_s0` | `zero_s0` | edu bpb | Go bpb | SQL bpb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eq11lr (6.25e-4 x [3.03, 1.24, 0.76]) | **0.1115** | 0.2872 | **−0.176** | 0.969 | 1.026 | **1.4511** | 0.970 | 1.802 |
| control (8e-4 x [1.5, 1.2, 1.0]) | 0.0107 | 0.1030 | −0.092 | 0.289 | 4.278 | 1.5304 | 1.019 | 1.854 |
| ladder blog split, 0.79B rung (`5rvc25h5`) | 0.0008 | 0.156 | −0.155 | 0.256 | 3.513 | 1.5498 | | |
| reference arm, 0.79B rung (for scale) | 0.135 | 0.114 | +0.021 | 1.041 | 0.917 | 1.3178 | 0.884 | 1.922 |

`coarse_shuffle_s1` by eval (steps 2,000 to 12,000): eq11lr 0.0054, 0.014, 0.029, 0.053, 0.085, 0.112; control 0.0001, 0.0013, 0.0033, 0.0058, 0.0080, 0.0107.

- Main prediction: shuffle − zero at stage 1 is −0.176. **Not supported.**
- Other prediction 1 (eq11lr `shuffle_s1` above the control's): holds, 0.112 against 0.011 (10x).
- Other prediction 2 (edu bpb not worse than the control by more than 0.059): holds. It is 0.079 *better* (1.4511 against 1.5304), more than the 2-stage seed spread.

### What the arms show

- **The paper's LR wakes most of the main network in our code.** Shuffle use grows 10x over the control and reaches 83% of the reference arm's value at the same budget (0.112 against 0.135). It rises at every eval.
- **The decoder still leans on a constant part.** Zeroing the main network costs 0.287, more than shuffling it (0.112). In the reference arm the order is the other way round (0.114 and 0.135). So our decoder reads the main network's content now, and it also reads a large document-independent part. The sign test asked for the reference's balance, and our arm is not there.
- **Stage 0 changes the most.** The control's stage-0 coarse path is a constant (zero 4.28, shuffle 0.29). With the paper's LR the two are equal (1.03 and 0.97), close to the reference arm (0.92 and 1.04).
- **It is the largest bpb gain we have had from one change to the 2-stage model:** −0.079 edu, −0.049 Go, −0.052 SQL at the same FLOPs and bytes. Speed is the same (55 ms/step).
- The control's `shuffle_s1` (0.0107) is 13x the ladder rung's (0.0008) with the same code and seed. The ladder rung branched off a longer run and saw a different data stream. So at 12,000 steps this probe is noisy by an order of magnitude when the main network is dead. The live values (0.11 to 0.15) are far above that noise.

## Verdict

**Not supported.** With the paper's Eq. 11 per-stage LR, our blog split's stage-1 decoder still reads more of a constant than of content from the main network (shuffle − zero = −0.176). The LR is a large partial fix. It gives 10x more main-network use and 0.079 bpb at 786M bytes, but it does not give the reference's signature on its own.

## Updated hypothesis

The difference ablation showed two causes in the reference model: Transformer outer stages and our LR each nearly kill its main network. This run fixes one of the two in our model, and the main network comes most of the way back. The remaining cause is the outer block type. Next: [our H-Net with Mamba-2](../refuted/2026-11-02-our-hnet-with-mamba2.md), with our LR and with the paper's LR. If both fixes together give the reference signature, our 2-stage model gets both, and the BPE comparison runs again at 3.15B bytes.

## Related

- [Reference H-Net difference ablation](../refuted/2026-11-01-reference-hnet-difference-ablation.md)
- [Reference H-Net at our scale](../successes/2026-11-01-reference-hnet-at-our-scale.md)
- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
