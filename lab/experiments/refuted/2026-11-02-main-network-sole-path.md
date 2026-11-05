# Experiment: main network sole path

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (a layout change to an existing arm on existing data; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

When the main network is the only path that can carry information across a document, does our 2-stage H-Net learn the document memory through it?

Context: on the synthetic recall language, the model learns the full memory (val bpb about 1.08, against the no-memory anchor 2.006) with stage 1's attention window at 1,024 and at 8, and in both cases the main network carries none of it (`coarse_shuffle_s1` at the no-recall control's noise level). With window 8, the Mamba-2 stacks of stages 0 and 1 hold the memory ([hierarchy on synthetic recall](2026-11-02-hierarchy-on-synthetic-recall.md), [main network takes recall](2026-11-02-main-network-takes-recall.md)). On text, our main network is read late or not at all.

## Hypothesis

Our main path (chunking, the main network, dechunking, the stage-1 and stage-0 decoders) can carry content, and the main network is ignored so far only because the outer Mamba-2 stacks do the same job. With every outer block a small-window attention block, and windows small enough that the stacked outer layers reach only 24 bytes back, only the main network's global attention spans a document, so the main network takes the memory.

Reach, not window, is what counts: windows stack. Stage 0 has 4 encoder and 4 decoder attention layers. With window w0, they reach 4(w0 − 1) + 4(w0 − 1) bytes back. Stage-1 layers add 10(w1 − 1) positions. A first draft of this plan had w0 = 16 and w1 = 8, a reach of about 370 bytes. The reach-limited oracle (below) gives 1.173 bpb at 370 bytes, so the outer layers alone could have taken most of the memory. This plan uses w0 = 4 and w1 = 1: a reach of 24 bytes in the layers. A byte reads the stage-1 vector of its chunk, computed at the chunk's first byte, so the distance back to the chunk start adds to that (chunks average 3 bytes at ratio 3). The anchor this plan grades against is R = 48, which allows a chunk of up to 24 bytes.

Fixed anchors, the reach-limited oracle on the first 2M bytes of recall val (`scripts/reach_anchors.py`, `oracle_bits(reach=R)`; the memory holds only the words that start in the last R bytes):

| reach R (bytes) | 0 | 24 | 48 | 96 | 192 | 370 | 1,000 | whole document |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle bpb | 2.0061 | 1.9129 | 1.8119 | 1.6419 | 1.4026 | 1.1731 | 1.0263 | 1.0215 |

## Predictions

Main prediction (only this one answers the Question):

- With all outer blocks small-window attention, shuffling the main network's output (`coarse_shuffle_s1`) raises recall val bpb at step 6,000 by more than the largest value of the same-layout no-recall control over its three seeds, in every recall seed (0, 1, 2). All three: supported. Any at or below: not supported.

Other predictions (at most two):

- Recall val bpb at step 6,000 is below the reach-48 oracle (1.8119) in all three seeds: the model holds memory that no outer path can reach.
- Recall val bpb at step 6,000 is above that of the Mamba-2 layout with window 8 (1.0790, 1.0810, 1.0833) in all three seeds: the coarse path learns the memory less well than a Mamba-2 state in the same budget.

If supported, we do: our main path works. On text the main network is redundant at our budget: text has little content that the main network gives and the outer Mamba-2 stacks do not. Then the text question is about budget and data (the paper's regime), not about our code. The B rerun ([coarse JEPA on dead seeds](2026-11-02-coarse-jepa-on-dead-seeds.md)) tells whether a direct loss changes that.

If not supported, we do: if recall val bpb stays near 2.006, nothing learned the memory, and our main path has a defect or does not learn in 6,000 steps. Look at the gradient that reaches the main network from the LM loss, per step, against a 1-stage model with the same main network. If recall bpb is below 1.8119 but `coarse_shuffle_s1` is not above the control, the memory goes through a path this plan missed. The dechunk smoothing is the first candidate: an EMA across chunks (`hnet/stage.py`), a linear recurrence of unlimited range over local features. Find the path before any text work.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including `tests/test_attention.py` with the flex mask (a query sees exactly its last w keys).
- Ratios within 15% of 3 x 3 at the end. Finite loss. 393,216,000 training bytes per arm.
- Oracle lower bound: recall val bpb at or above 1.021, no-recall at or above 2.005.
- No Mamba-2 block in the built model: each arm's Hydra config shows layouts t4, t4, T1t4, t4T1 and windows 4 and 1, and its parameter count is below the Mamba-2 arms' 41,801,344.
- `scripts/reach_anchors.py` reproduces the full oracle of `meta.json` (1.0214751) before it computes the reach anchors (an assert in the script).

## Method

```
data: synth/recall and synth/norecall, unchanged (scripts/make_synth.py, seed 0)
model: reflayout_2stage with each outer Mamba-2 block (m, no MLP) replaced by an attention block (t, no MLP):
    stage 0: enc t4, dec t4, window 4 bytes: 12 bytes of reach in the encoder, 12 in the decoder. A word is at
             most 8 letters + space, so the decoder still sees the whole current word: enough for spelling and for
             the no-recall oracle.
    stage 1: enc T1t4, dec t4T1, window 1 position: no mixing across stage-1 positions
    reach of every path that skips the main network: 24 bytes + the distance back to the current chunk's start
    (the dechunk EMA aside; see If not supported)
    main: unchanged, T10, global attention over about 455 positions = the whole 4,096-byte row
train: as the window-8 arms: configs/train/ladder.yaml, 6,000 steps, no rungs, eval every 1,000 steps
arms: data=synth_recall seeds 0, 1, 2 ; data=synth_norecall seeds 0, 1, 2
grade: probe/coarse_shuffle_s1 at step 6,000, recall against no-recall
reference (not rerun): the window-8 arms of main network takes recall
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, once the suite passes (queue: `ignored/runs/main-network-sole-path/chain.sh`, before the coarse-JEPA rerun):
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/main-network-sole-path/recall main-network-sole-path reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.0.enc_layout=t4 model.stages.0.dec_layout=t4 model.stages.0.window=4 model.stages.1.enc_layout=T1t4 model.stages.1.dec_layout=t4T1 model.stages.1.window=1 train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000`
  - the same with `norecall` in the log dir and `data=synth_norecall`
- wandb project `hnet-router`, group `main-network-sole-path`.
- Parameters change: an attention block without MLP at d 256 has 4d² = 262k weights, a Mamba-2 block about 0.5M. The outer stages get smaller. This is a test of the path, not a matched comparison. Parameter count and fwd FLOPs/byte are logged as measurements.
- Cost: about 12 min per arm at the 500 W cap, about 75 min in total.

## Results

All six arms ran to step 6,000 (2026-11-02 20:26 to 21:09; commit 312ae73, suite 163 passed, 1 skipped, with the flex window test). Log dir `ignored/runs/main-network-sole-path/{recall,norecall}`. wandb: recall seeds 0, 1, 2 `jp3j4sv3`, `nic59rj1`, `8iiuem6n`; no-recall `mvgqbelc`, `6j4skuzx`, `cfitvsw1`. Artifacts: [results/](../results/2026-11-02-main-network-sole-path/) (`eval-curves.csv`, made by `pull.py`).

### Gates

| gate | result |
|---|---|
| suite (`HNET_TEST_CUDA=1`) incl. `tests/test_attention.py` with flex: 163 passed, 1 skipped | pass |
| ratios at the end (3 x 3 ±15%): 2.95 to 3.02 x 2.88 to 3.02 | pass |
| 393,216,000 training bytes per arm; finite loss | pass |
| oracle lower bound: recall 1.9898 to 2.0128 ≥ 1.021; no-recall 2.0109 to 2.0112 ≥ 2.005 | pass |
| no Mamba-2: layouts t4, t4, T1t4, t4T1, windows 4 and 1 in the Hydra configs; 38,552,832 params < 41,801,344 | pass |
| `reach_anchors.py` reproduces the full oracle (assert) | pass |

Fwd FLOPs/byte 14.93 to 15.49M (Mamba-2 layout: 21.72 to 22.66M).

### Curves (val bpb / `coarse_shuffle_s1`)

| step | recall s0 | recall s1 | recall s2 | no recall s0 | no recall s1 | no recall s2 |
|---:|---|---|---|---|---|---|
| 4,000 | 2.1000 / 3.236 | 2.0943 / 2.639 | 2.0938 / 0.376 | 2.0426 / 0.366 | 2.0428 / 0.462 | 2.0399 / 0.411 |
| 5,000 | 2.0685 / 3.282 | 2.0706 / 2.502 | 2.0715 / 0.383 | 2.0338 / 0.340 | 2.0336 / 0.431 | 2.0335 / 0.341 |
| 6,000 | **1.9913 / 2.986** | **1.9898 / 2.128** | **2.0128 / 0.330** | 2.0112 / 0.317 | 2.0109 / 0.415 | 2.0112 / 0.324 |

At step 6,000: `zero_s1` recall 1.532, 0.970, 0.268; no-recall 0.290, 0.349, 0.274. `shuffle_s0` 2.5 to 4.1 in all arms. Anchors: no memory 2.006; reach-24 oracle 1.9129, reach-48 1.8119; full oracle 1.021. The LR decays over steps 4,800 to 6,000.

- Main prediction (recall `coarse_shuffle_s1` above the no-recall maximum, 0.415, in every seed): 2.986, 2.128, 0.330. Seed 2 is below. **Not supported.**
- Other prediction 1 (recall bpb below the reach-48 oracle 1.8119 in all seeds): does not hold (1.9913, 1.9898, 2.0128).
- Other prediction 2 (recall bpb above the Mamba-2 window-8 arms, 1.079 to 1.083): holds.

### What the runs show (observations after the fact)

- **The main path carries content.** In the no-recall arms, shuffling the main output costs 0.32 to 0.42 bpb, and those arms end at 2.011, near the oracle 2.005. With outer reach this small, the decoders take local context from the main network. A defect that blocks the main path is ruled out.
- **The memory through the main network is slow to learn, and it is a lottery.** Recall seeds 0 and 1 depend on document-specific main content from step 4,000 on (shuffle cost 3.2 and 2.6 bpb), but their bpb is worse than the no-recall control's until the LR decay. Only at step 6,000 do they pass the no-memory anchor, by 0.015 bpb. Seed 2 never builds the route: its shuffle cost and bpb match the control. The Mamba-2 layouts learned the full memory (about 0.93 bpb) by steps 2,000 to 4,000.
- This is the same pattern as stage 1 on text: per seed, the route through the main network opens late, or does not open in the budget.

## Verdict

**Not supported.** With the main network as the only path across a document, recall seed 2 does not use it more than the no-recall control (`coarse_shuffle_s1` 0.330 against a control maximum of 0.415). Seeds 0 and 1 do (2.986, 2.128), but they gain only 0.015 bpb of the 0.98 available in 6,000 steps.

## Updated hypothesis

Our main path works, but learning through it is slow and seed-dependent. When a faster route exists (outer Mamba-2 state, stacked outer attention), gradient descent takes that route, and the main network stays unread. That explains the synthetic results and fits the text results without a code defect. Two causes fit the slowness, and they lead to different fixes:

1. **Budget.** The route opens on its own with more steps (seeds 0 and 1 were mid-transition at 6,000). Test: the same layout and arms at 4x the steps.
2. **Optimization of the main path.** Something in the path makes its gradient small or noisy: the main network's LR multiplier (0.72), the chunk interface (STE, the EMA), or the 9-byte misalignment between main positions and words. Test after the budget run, by measuring the main network's gradient norm per step against the outer stages'.

The budget test comes first. It is cheap, and if seed 2 opens its route with more steps, "train longer" is the answer on text too.

## Related

- [Main network takes recall](2026-11-02-main-network-takes-recall.md)
- [Hierarchy on synthetic recall](2026-11-02-hierarchy-on-synthetic-recall.md)
- [Coarse JEPA on dead seeds](../refuted/2026-11-02-coarse-jepa-on-dead-seeds.md)
