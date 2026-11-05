# Experiment: grown hierarchy keeps main

Status: refuted

Date proposed: 2026-11-03
Algorithm: `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/grow.py` (pseudocode there), `train.grow_from`
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

If the main network is already a working language model when stage 1 is inserted, does it stay in use in the 2-stage H-Net on text?

Context: in the plain reference layout trained from scratch, the stage-1 fate (shuffle cost minus zero cost of the main output at the stage-1 decoder) ends above 0 in one of five seeds at 786M bytes ([main gradient coherence](../refuted/2026-11-03-main-gradient-coherence.md)), and the blog-split main network was dead at every rung up to 12.58B bytes ([compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)). The 1-stage H-Net uses its main network (`coarse_shuffle_s0` up to 1.71). No fix so far changed who leads at step 0: the compute split, the gradient, and the read interface all acted after the outer stages had taken the job ([byte read on text](../refuted/2026-11-03-byte-read-on-text.md)).

## Hypothesis

The dead main network is a race lost at initialization. At step 0 the outer stages learn local statistics faster than a fresh main network behind two chunkings, so gradient descent routes around it and never returns. Start the main network ahead (trained in a 1-stage model, where it is the coarse path), then insert stage 1 with a fresh init. The stage-1 decoder then reads a main output that already predicts well, and keeps reading it.

## Predictions

Main prediction (only this one answers the Question):

- At the end (8,000 steps after the growth, 786M bytes in total), the stage-1 fate (`probe/coarse_shuffle_s1` − `probe/coarse_zero_s1`) is above 0 in every grown seed (0, 1, 2). The plain control from scratch has it above 0 in 1 of 5 seeds. All three above 0: supported. Any at or below 0: not supported.

Other predictions (at most two):

- Before the growth (step 4,000 of the 1-stage phase), the stage-0 fate (`probe/coarse_shuffle_s0` − `probe/coarse_zero_s0`) is above 0 in every seed: the 1-stage main network is in use when stage 1 goes in.
- Edu val bpb at the end is below the best plain seed (1.3133) in every grown seed. This one is not at matched compute: the 1-stage phase spends 36.1M fwd FLOPs/byte, so the grown run spends about 20% more training FLOPs than the control.

If supported, we do: the hierarchy should be grown, not trained from scratch. Next, a matched-FLOPs growth (shorter 1-stage phase or a smaller 1-stage main network) on the compute ladder against BPE, and an Agent Note for the training recipe.

If not supported, we do: the outer path takes the job back even from a main network that starts ahead. Then the cause is not the init race but the stage-1 path itself, and "thin outer stages" (run in this queue) is the test that remains on the 2-stage side.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`).
- The growth log line reports the new stage's keys as the only fresh keys; params 36,175,552 (1-stage) and 41,801,344 (grown).
- Edu ratios within 15% of 3 (1-stage) and 3 x 3 (grown) at the end of each phase; 262,144,000 + 524,288,000 = 786,432,000 bytes; finite loss.
- Edu fwd FLOPs/byte within 5% of 36.12M (1-stage) and 22.58M (grown) at target ratios.

## Method

```
phase 1: reflayout_1stage (stage 0 m4/m4 d256 ratio 3 = reflayout's stage 0; main T10 d512; lr 6.25e-4 x [3.03, 0.72])
         configs/train/ladder.yaml, 4,000 steps, decay_frac 0 (warmup 500, then constant LR), no rungs -> final.pt
growth:  reflayout_2stage with train.grow_from = phase-1 final.pt (hnet/grow.py):
         embed, head, stage 0, main <- phase 1 ; stage 1 (T1m4/m4T1 d256, router, residual) fresh ;
         phase-1 stage-0 pad_dim (256 -> 512) -> stage 1's pad_dim ; AdamW state fresh
phase 2: 8,000 steps of ladder.yaml (warmup 500, WSD decay over the last 20%), no rungs
seeds:   0, 1, 2 (phase 2 seed = phase 1 seed)
control: plain reflayout_2stage from scratch, 12,000 steps, seeds 0-4 = the edu arms of main gradient coherence
         (wandb akqhdkkz, ipcxldpl, zplsyk1o, 5d3o336d, 7dffta9c)
grade:   probe/coarse_*_s1 at the last eval of phase 2 (8 edu val batches, every 2,000 steps)
```

- Queue: `ignored/runs/next-ladder/chain.sh` (B, then C, D, A), launched at 19:30 on 2026-11-03 at the user's request. A 300-step smoke of each new config runs first.
- wandb project `hnet-router`, groups `grown-hierarchy-p1` and `grown-hierarchy`.
- Cost: about 10 min per 1-stage phase and 14 min per 2-stage phase, about 75 min in total.

## Results

Chain `ignored/runs/next-ladder/chain.sh` (copy in [results/](../results/2026-11-03-grown-hierarchy-keeps-main/)), code at c5b2cae, 19:30 to 20:40 on 2026-11-03. Artifacts: `summary.txt` and `val-curves.csv` (`pull.sh`, which calls the spike's `scripts/pull_runs.py`), `chain.sh`. wandb: phase 1 zo9jw4vh, 4bcw63x6, myh03srw; grown hsm8z1b0, 41pqzyjo, eribnpcj.

### Gates (all hold)

- Suite: 198 passed, 1 skipped. Smokes of both phases passed (200 steps each).
- Params 36,175,552 (phase 1) and 41,801,344 (grown). Each growth log line reports 90 fresh keys, all in the new stage (`grow_state_dict` raises on any other key).
- Edu ratios: phase 1 2.99-3.00; grown 3.01-3.05 x 2.86-3.03. Bytes 262,144,000 + 524,288,000. Finite loss.
- Edu fwd FLOPs/byte: phase 1 36.14M-36.22M (36.12M planned); grown 22.29M-22.90M (22.58M planned, -1.3% to +1.4%).

### Main prediction: stage-1 fate at the end of phase 2

`shuf` and `zero` are the bpb costs of shuffling and zeroing the main output at the stage-1 decoder. Fate = shuf - zero.

| seed | phase-1 main use (shuf s0 at step 4,000) | grown zero s1 | grown shuf s1 | grown fate s1 | grown edu bpb |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0002 | 0.0002 | 0.0000 | -0.0002 | 1.4595 |
| 1 | 0.0033 | 0.0182 | 0.0000 | -0.0181 | 1.4513 |
| 2 | 0.7431 | 0.3377 | **0.6218** | **+0.2841** | **1.3112** |
| plain 0-4 | | 0.057-0.154 | 0.0005-0.158 | +0.023 in seed 0 only | 1.3133-1.3661 |

The fate is above 0 in seed 2 only. Seeds 0 and 1 are at or below 0.

### Other predictions

- Phase-1 stage-0 fate above 0 at step 4,000 in every seed: does not hold. Seed 2 is +0.286. Seeds 0 and 1 are -0.086 and -0.066, and their shuffle costs (0.0002, 0.0033) show that the 1-stage main network carries no document content.
- Edu bpb below 1.3133 in every grown seed: does not hold. Seed 2 is 1.3112. Seeds 0 and 1 are 1.45-1.46, worse than every plain seed.

### Observations (not predictions)

- **The growth keeps what phase 1 had.** The seed whose 1-stage main network was in use (seed 2) stays live at both stages after the growth. Its stage-1 shuffle cost (0.62) is 4 times the largest of any plain seed (0.158). The two seeds whose 1-stage main network was dead stay dead. No seed changed state at the growth.
- **The lottery happens in the 1-stage model too.** With the reference layout's stage 0 (Mamba-2, a state over the whole row), the 1-stage main network was in use in 1 of 3 seeds at step 4,000. The 1-stage model in [looped coarse main](./2026-11-03-looped-coarse-main.md) has a windowed attention stage 0 (T2/T2, 512 bytes). Its R = 1 arm uses the main network in 3 of 3 seeds (shuffle cost 0.37-1.03). The two 1-stage configs differ in more than the stage-0 mixer (ratio 3 vs 4, main T10 vs T8, LR multipliers), so this is a lead, not a result.
- **A dead grown model is worse than a dead plain one.** Grown seeds 0 and 1 (1.45-1.46) are 0.09-0.15 behind the worst plain seed. They spent 4,000 of their 12,000 steps in a 1-stage model at constant LR, with a main network that the outer stage routed around.
- **Grown seed 2 is the best 2-stage model at 786M bytes.** Its edu bpb is 1.3112 (Go 0.939, SQL 2.022). That is 0.002 below the best plain seed, inside the plain seed spread, with about 20% more training FLOPs. It is still 0.04 behind BPE (1.271, 1.273).

## Verdict

**Not supported.** At the end, the stage-1 fate is above 0 in one of three grown seeds (seed 2: +0.284). It is at or below 0 in seeds 0 (-0.0002) and 1 (-0.018). The test failed in phase 1, not at the growth: seeds 0 and 1 had a dead main network before stage 1 went in, so they never gave the main network a head start.

## Updated hypothesis

The init-race reading survives in a narrower form. A main network that is in use when stage 1 goes in stays in use (seed 2: stage-1 shuffle cost 0.62, the largest we have measured in our 2-stage models). But the race starts in the 1-stage model. When stage 0 is Mamba-2, the 1-stage main network loses it in 2 of 3 seeds. When stage 0 is windowed attention (512 bytes), the 1-stage main network was in use in every seed we trained. So the lead is the stage-0 mixer's reach. A Mamba-2 stage 0 has a state over the whole 4,096-byte row and can do the main network's job early. Windowed attention cannot see past 512 bytes.

The test of that lead is one large move: grow the 2-stage model from the windowed-attention 1-stage (main live in every seed). A grown model then starts with a live main network every time, and the outcome says whether a live start survives in every seed, not just in one. Also, grade use by the shuffle cost from now on (anchor 0). The fate's sign reads a used main network as dead when zeroing is far out of distribution: see [looped coarse main](./2026-11-03-looped-coarse-main.md), where R = 1 seeds have shuffle costs of 0.37-1.03 and fates of -2.6 to -3.5.

## Related

- [Byte read on text](../refuted/2026-11-03-byte-read-on-text.md)
- [Main gradient coherence](../refuted/2026-11-03-main-gradient-coherence.md)
- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
- [Deep main compute split](../refuted/2026-10-31-deep-main-compute-split.md)
