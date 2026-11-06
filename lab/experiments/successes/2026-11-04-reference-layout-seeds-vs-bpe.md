# Experiment: reference layout seeds vs BPE

Status: success

Date proposed: 2026-11-04
Algorithm: none (existing arms on the existing compute ladder)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

At the 12.58B-byte top rung, is every seed of our 2-stage H-Net (the reference configuration in our code) below every seed of the BPE Transformer on edu val bpb?

Context: in [reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md) both seeds ended below BPE's one seed (1.0905 and 1.0979 against 1.1156), but the mean lead (0.021) was inside the ±0.026 band. The band was our own seed spread at 3.15B bytes, used because BPE has one seed. On one bytes axis, BPE starts ahead, each H-Net seed crosses it (seed 0 at 2.1B bytes, seed 1 at 6.8B), and the gap then holds at −0.018 to −0.030 to the end. The test was too small in its control. This test adds seeds, so it can grade an ordering with no band.

## Hypothesis

The H-Net lead on edu is real at this scale. BPE's seed spread at 12.58B bytes is smaller than the lead, so a second BPE seed lands above every H-Net seed. Each H-Net seed comes online at its own time, but once past the crossing it stays ahead of BPE.

## Predictions

Control: BPE Transformer seed 0 (1.1156, wandb dnpo8u01) and the new BPE seed 1. H-Net seeds 0 and 1 are fixed (1.0905, 1.0979); the new data is H-Net seed 2 and BPE seed 1.

Main prediction (only this one answers the Question):

- At 12.58B bytes, edu val bpb is lower in every H-Net seed (0, 1, 2) than in every BPE seed (0, 1). All six pairs in that order: supported. Any H-Net seed at or above any BPE seed: not supported.

Other predictions (at most two):

- In H-Net seed 2, the edu gap to BPE seed 0 (at matched bytes, every eval) crosses 0 and stays below 0 from the crossing to the end.
- At 12.58B bytes, Go and SQL bpb are lower in every H-Net seed than in every BPE seed.

If supported, we do: Phase 1's BPE goal holds on edu at about 8e17 training FLOPs, with the 1-stage beaten too (A: by 0.06-0.07). The reference configuration becomes our 2-stage default (Agent Note), and Phase 2 builds on it. Bring the result to the user.

If not supported, we do: read which pair failed. If BPE seed 1 is below an H-Net seed, BPE's spread is as large as the lead, and the edu lead at 42M params is inside seed noise; the clear gain is on code. If H-Net seed 2 ends above BPE seed 0, read its crossing (other prediction 1): a seed that never crossed means the online lottery still decides the result at 12.58B bytes. Either way, bring the windowed and Parcae results to the user with this one, because the next 2-stage design comes from them.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`).
- Params 41,801,344 (H-Net) and 62,465,536 (BPE).
- H-Net edu ratios within 15% of 3 x 3 at every rung; edu fwd FLOPs/byte within 5% of BPE's 22.27M at every rung.
- Training bytes 786,432,000, 3,145,728,000, and 12,582,912,000 at the rungs (BPE: 180,080 steps of 16 x 1,024 tokens, the same bytes within 1%). Finite loss.

## Method

```
code:    lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
arms:    bpe_transformer seed 1   (as compute-ladder-vs-bpe: train.max_steps=180080 data.token.target_train_bytes=12582912000)
         reflayout_2stage seed 2  (as reference-layout-vs-bpe)
budget:  configs/train/ladder.yaml: 12.58B bytes, rungs at 1/16 and 1/4, the main run decays over its last 20%
seed:    sets the init and the train-batch order; the val batches are fixed (the same in every run)
grade:   val/edu/bpb at the end of each run, against H-Net seeds 0-1 and BPE seed 0 (not rerun)
gap:     lab/experiments/results/2026-11-04-reference-layout-vs-bpe/plot_gap.py, extended to the new seeds
```

- Launch, from the spike dir, after Parcae coarse main: `TRAIN=ladder scripts/run_arms.sh ignored/runs/reference-layout-seeds-vs-bpe reference-layout-seeds-vs-bpe bpe_transformer:1 -- train.max_steps=180080 data.token.target_train_bytes=12582912000`, then the same with `reflayout_2stage:2` and no overrides.
- wandb project `hnet-router`, group `reference-layout-seeds-vs-bpe`.
- Cost: about 2.6 h (BPE) + 5.7 h (H-Net), about 8.5 h in total.
- Amendment 2026-11-05: the runs use code at 0252bf6, not the commit that added this file. That commit adds packed inner levels and compiled stack segments (`.agents/notes/implemented/feature/2026-11-05-throughput-compile-and-packing.md`). Same math per valid position: the GPU suite, the golden values and `tests/test_packing.py` pin it; only the fp32 summation order changes. H-Net seeds 0-1 and BPE seed 0 ran on the older code and are not rerun. H-Net cost is now about 4 h (about 71 ms/step against 105).
- Cheaper than the other can't-tell route in A: the train set holds about 14.5B bytes, only 15% past the top rung.

## Results

Runs: BPE seed 1 (wandb 8g0lkndi) and H-Net seed 2 (wandb n8ls6vxe), queue `ignored/runs/reference-layout-seeds-vs-bpe/`, 2026-11-05 02:00 to 08:59, code 0252bf6. Gap curves: [plot_gap.py, gap.png, gap.txt](../results/2026-11-04-reference-layout-seeds-vs-bpe/).

### Gates

- Suite passed (`HNET_TEST_CUDA=1`) and both 200-step smokes trained (chain.log).
- Params 41,801,344 (H-Net) and 62,465,536 (BPE).
- H-Net seed 2 edu ratios over every eval: stage 0 2.99 to 3.18, stage 1 2.66 to 3.15 (inside 2.55 to 3.45). Edu fwd FLOPs/byte 22.04M to 22.99M against BPE 22.27M (-1.0% to +3.2%).
- Training bytes 12,582,912,000 (H-Net, 192,000 x 65,536); BPE 180,080 steps. Finite loss.

### Main prediction: edu val bpb at 12.58B bytes

| arm | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| H-Net reference layout | 1.0905 | 1.0979 | **1.0913** |
| BPE Transformer | 1.1156 | **1.1109** | |

The worst H-Net seed (1.0979) is 0.013 below the best BPE seed (1.1109). All six pairs are in the predicted order. The two BPE seeds differ by 0.005; the three H-Net seeds span 0.007.

### Other predictions

- H-Net seed 2, edu gap to BPE seed 0 crosses 0 and stays below: **does not hold as written**. It crosses at 2.23B bytes, goes back above at 2.36B, and crosses for good at 2.49B. From 2.49B to the end it stays below (final gap -0.0243).
- Go and SQL lower in every H-Net seed than in every BPE seed: **holds**. Go: H-Net 0.581, 0.587, 0.582 against BPE 0.654, 0.650. SQL: H-Net 1.189, 1.207, 1.218 against BPE 1.337, 1.318.

### Readings

- Crossings against BPE seed 1: H-Net seed 0 at 2.23B, seed 1 at 6.82B (one recross at 6.95B, then below from 7.21B), seed 2 at 2.49B. Two of three seeds come online early; seed 1 is the late one.
- Throughput: H-Net seed 2 trained at 886 kB/s (74.0 ms/step with evals and probes) against BPE seed 1 at 1,384 kB/s (50.5 ms/step). At matched FLOPs/byte, the H-Net costs 1.56x the wall clock of BPE.
- H-Net seed 2 final probes: sem/word_s0 AUROC 0.908, sem/hidden_s0 0.794; coarse_shuffle_s0 rise +1.90 bpb, so the main network is in use.

## Verdict

**Supported.** At 12.58B bytes, edu val bpb is lower in every H-Net seed (1.0905, 1.0979, 1.0913) than in every BPE seed (1.1156, 1.1109).

## Updated hypothesis

At about 8e17 training FLOPs and 42M params, the reference configuration in our code beats a matched-FLOPs BPE Transformer on edu by 0.013 to 0.025 bpb, and on Go and SQL by more (0.06 to 0.07 and 0.10 to 0.15). The seed spread on both sides (0.005 to 0.007) is smaller than the lead. Seeds still differ in when they come online (2.1B to 6.8B bytes), but all three were past the crossing well before the decay. Constitution checklist item 1 (compute win) holds at our scale. The wall-clock cost (1.56x BPE at matched FLOPs) is the open item for throughput.

## Related

- [Reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md)
- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
- [Dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md)
