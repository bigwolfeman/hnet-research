# Experiment: main path recall budget

Status: success

Date proposed: 2026-11-02
Algorithm: none (the sole-path arms at 4x the steps; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

With the main network as the only path across a document, does the route for the document memory open in every seed if training runs 4x longer?

Context: in [main network sole path](../refuted/2026-11-02-main-network-sole-path.md) (6,000 steps), the main path carries content (no-recall shuffle cost 0.32 to 0.42 bpb). For recall, seeds 0 and 1 built a route through the main network (shuffle cost 2.99, 2.13) but passed the no-memory anchor by only 0.015 bpb, during the LR decay. Seed 2 built no route. The Mamba-2 layouts learn the full memory by steps 2,000 to 4,000.

## Hypothesis

Learning through the main path is slow, not blocked. With 24,000 steps, the route opens in every seed and the model takes a large part of the memory through it.

## Predictions

Main prediction (only this one answers the Question):

- At step 24,000, recall `coarse_shuffle_s1` is above the largest value of the 24,000-step no-recall control over its three seeds, in every recall seed (0, 1, 2). All three: supported. Any at or below: not supported.

Other predictions (at most two):

- Recall val bpb at step 24,000 is below the reach-48 oracle (1.8119) in all three seeds: the model holds memory that no outer path can reach.
- In each recall seed, the first eval at which recall `coarse_shuffle_s1` is above the control maximum comes before the LR decay starts (step 19,200).

If supported, we do: budget is the answer. The main network's route opens on its own, later than an outer route would. On text, this says "train longer" before any architecture change: rerun [reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md) at the longest budget we can afford, and grade stage 1 there.

If not supported, we do: the main path is hard to optimize, not only slow. Measure, per step, the gradient norm that reaches the main network from the LM loss against that of the outer stages, in a sole-path seed that opens the route and one that does not. Then test the cheapest lever the measurement points to (the main network's LR multiplier, 0.72, first).

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including `tests/test_attention.py` with flex.
- Ratios within 15% of 3 x 3 at the end. Finite loss. 1,572,864,000 training bytes per arm.
- Oracle lower bound: recall val bpb at or above 1.021, no-recall at or above 2.005.
- No Mamba-2: layouts t4, t4, T1t4, t4T1, windows 4 and 1 in each Hydra config; 38,552,832 params.

## Method

```
data: synth/recall and synth/norecall, unchanged (400M train bytes each: 24,000 steps x 65,536 bytes = 3.9 epochs)
model: as main network sole path (no Mamba-2; stage 0 t4/t4 window 4; stage 1 T1t4/t4T1 window 1; main T10 global)
train: configs/train/ladder.yaml, 24,000 steps (WSD: warmup 500, decay over the last 20% = steps 19,200 to 24,000),
       no rungs, eval every 2,000 steps
arms: data=synth_recall seeds 0, 1, 2 ; data=synth_norecall seeds 0, 1, 2
grade: probe/coarse_shuffle_s1 at step 24,000, recall against no-recall
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, once the suite passes (queue in `ignored/runs/main-path-recall-budget/chain.sh`, before the coarse-JEPA rerun):
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/main-path-recall-budget/recall main-path-recall-budget reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.0.enc_layout=t4 model.stages.0.dec_layout=t4 model.stages.0.window=4 model.stages.1.enc_layout=T1t4 model.stages.1.dec_layout=t4T1 model.stages.1.window=1 train.max_steps=24000 "train.ladder_fracs=[]" train.eval_every=2000`
  - the same with `norecall` in the log dir and `data=synth_norecall`
- wandb project `hnet-router`, group `main-path-recall-budget`.
- The train stream repeats about 3.9 times. The val set is separate (8M bytes, other rng streams), and the recall memory is per document, so repeats cannot memorize val. A falling train loss with flat val is still possible; train and val bpb are both logged.
- Cost: 7 min per 6,000-step arm at the 500 W cap, so about 28 min per arm, about 2.8 h in total.

## Results

All six arms ran to step 24,000 (2026-11-02 21:30 to 23:59; commit f2095fc). Log dir `ignored/runs/main-path-recall-budget/{recall,norecall}`. wandb `hnet-router`, group `main-path-recall-budget`: recall seeds 0, 1, 2 `fe6giopl`, `s8ovj136`, `u6k4r6rf`; no-recall `i77x72h4`, `dnd5vusa`, `kapjptbk`. Artifacts: [results/](../results/2026-11-02-main-path-recall-budget/) (`eval-curves.csv` from `pull.py main-path-recall-budget eval-curves.csv synth`; gate values from `gates.py main-path-recall-budget`).

### Gates

| gate | result |
|---|---|
| suite (`HNET_TEST_CUDA=1`) incl. `tests/test_attention.py` with flex: 164 passed, 1 skipped | pass |
| ratios at the end (3 x 3 ±15%): 2.98 to 3.01 x 3.00 to 3.07 | pass |
| 1,572,864,000 training bytes per arm; finite loss | pass |
| oracle lower bound: recall 1.9671 to 1.9976 ≥ 1.021; no-recall 2.0096 to 2.0097 ≥ 2.005 | pass |
| no Mamba-2: layouts t4, t4, T1t4, t4T1, windows 4 and 1 in the wandb configs; 38,552,832 params | pass |

Train loss at the end is above val bpb in every arm (for example recall seed 0: 2.05 against 1.97), so the 3.9 repeats of the train stream did not memorize.

### Curves (val bpb / `coarse_shuffle_s1`)

| step | recall s0 | recall s1 | recall s2 | no recall s0 | no recall s1 | no recall s2 |
|---:|---|---|---|---|---|---|
| 2,000 | 2.1564 / 2.425 | 2.1756 / 0.525 | 2.1871 / 0.413 | 2.0610 / 0.553 | 2.0554 / 0.617 | 2.0596 / 0.524 |
| 6,000 | 2.0788 / 3.297 | 2.0839 / 0.408 | 2.0807 / 0.307 | 2.0368 / 0.317 | 2.0343 / 0.388 | 2.0364 / 0.335 |
| 12,000 | 2.0352 / 3.960 | 2.0649 / 0.409 | 2.0611 / 0.388 | 2.0298 / 0.418 | 2.0321 / 0.319 | 2.0303 / 0.255 |
| 18,000 | 2.0260 / 4.165 | 2.0551 / 0.379 | 2.0589 / 0.394 | 2.0286 / 0.222 | 2.0302 / 0.177 | 2.0274 / 0.180 |
| 20,000 | 2.0180 / 3.568 | 2.0500 / 0.400 | 2.0493 / 0.390 | 2.0283 / 0.203 | 2.0265 / 0.186 | 2.0257 / 0.200 |
| 22,000 | 1.9904 / 3.707 | 2.0306 / 0.356 | 2.0309 / 0.402 | 2.0191 / 0.194 | 2.0189 / 0.200 | 2.0195 / 0.170 |
| 24,000 | **1.9671 / 1.792** | **1.9948 / 0.382** | **1.9976 / 0.373** | 2.0097 / 0.172 | 2.0096 / 0.214 | 2.0097 / 0.169 |

At step 24,000: `zero_s1` recall 0.701, 0.384, 0.374; no-recall 0.154, 0.233, 0.215. Anchors: no memory 2.006; reach-24 oracle 1.9129; reach-48 1.8119; full oracle 1.021. The LR decays over steps 19,200 to 24,000.

- Main prediction (recall `coarse_shuffle_s1` at 24,000 above the no-recall maximum, 0.214, in every seed): 1.792, 0.382, 0.373. **Supported.**
- Other prediction 1 (recall bpb below the reach-48 oracle 1.8119 in all seeds): does not hold (1.9671, 1.9948, 1.9976).
- Other prediction 2 (first eval above the control maximum at that eval, before step 19,200): holds. Seed 0 at 2,000, seed 1 at 6,000, seed 2 at 14,000.

### What the runs show (observations after the fact)

- **The route opens in every seed, but it carries little memory.** All three recall seeds pass the no-memory anchor (2.006) at the end. The best seed gains 0.039 bpb of the 0.985 bpb that the full memory is worth (4%). Seeds 1 and 2 gain 0.011 and 0.008, and they stay above the reach-24 oracle (1.9129). Their bpb alone does not prove memory beyond the outer reach. Only the shuffle cost shows it.
- **In seeds 1 and 2 the gap comes from the control.** Their shuffle cost stays flat at 0.36 to 0.41 from step 4,000 on. The no-recall control falls from 0.39 to about 0.2 over the same steps. The recall seeds keep a document-specific main content that the control gives up.
- **Seed 0 depends on the main path, without profit, for 16,000 steps.** Its shuffle cost is 3.3 to 4.2 bpb from step 4,000 to 22,000, and its bpb stays at the no-recall control's. Only in the LR decay does it pass the anchor (2.018 to 1.967) while its shuffle cost halves (3.71 to 1.79).
- **The lottery is per run, not per seed.** In [main network sole path](../refuted/2026-11-02-main-network-sole-path.md), seed 1 built a strong route by step 4,000 (shuffle cost 2.64). Here, with the same seed and a longer LR schedule (peak LR held to 19,200), seed 1 built only a weak one. The schedule changes which seed opens the route.
- **Every recall gain happens in the LR decay.** From step 18,000 to 24,000, recall bpb falls 0.059, 0.060, 0.061. The no-recall control falls 0.019, 0.021, 0.018.

## Verdict

**Supported.** At step 24,000, recall `coarse_shuffle_s1` is above the no-recall maximum (0.214) in every seed: 1.792, 0.382, 0.373. With 4x the steps, the route through the main network opens in all three seeds. The main prediction asked only whether the route opens. The hypothesis also said that the model takes a large part of the memory through it, and that did not happen: the best seed holds 4% of the memory.

## Updated hypothesis

Budget opens the route, but budget alone is not enough. Four times the steps took the best gain from 0.015 to 0.039 bpb (of 0.985 available), and most of every gain arrived in the LR decay. At this rate, "train longer" does not bring the main network to the memory on any budget we can afford. Two readings fit, and they lead to different fixes:

1. **The main path's gradient is small or noisy** (LR multiplier 0.72, the STE and EMA at the chunk interface). Seed 0 shows a large dependence on main content with no gain, which looks like a route that carries content but is not tuned. The strong effect of the LR decay also fits optimization noise. Test: measure the gradient norm that the LM loss sends to the main network against the outer stages, per step, in seed 0 (strong route) and seed 2 (weak route). Then raise the main LR multiplier if the measurement points there.
2. **The task is hard for this interface.** The main output for a chunk is causal: it holds nothing of the bytes after the chunk boundary. To recall a word that starts after a boundary, the decoder must match its own few-byte prefix (reach 4 bytes at stage 0) against a bag of document words held in one 512-wide vector. Attention over bytes does this with a content match; the chunk interface needs a learned lookup. Test: measure the recall gain per word by where the word starts relative to the stage-1 boundary. Words that straddle a boundary should gain more than words that start right after one.

The "if supported" action (rerun [reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md) at the longest affordable budget) stays the plan for text. On text, the outer stages have Mamba-2 and the main network has a different job, so the synthetic rate does not transfer directly. The gradient measurement (reading 1) is cheap and runs first.

## Related

- [Main network sole path](../refuted/2026-11-02-main-network-sole-path.md)
- [Main network takes recall](../refuted/2026-11-02-main-network-takes-recall.md)
- [Reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md)
- [Coarse JEPA on dead seeds](../refuted/2026-11-02-coarse-jepa-on-dead-seeds.md)
