# Experiment: main network takes recall

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (one config change to an existing arm on existing data; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

When stage 1's attention cannot reach across a document, does the main network take the document memory?

Context: on the synthetic recall language, all three seeds learned the memory (val bpb 1.07 to 1.09 against the no-memory 2.006), and shuffling stage 0's inner output cost 0.74 to 0.95 bpb. Shuffling the main network's output cost 0.0000: stage 1's own layers held the memory ([hierarchy on synthetic recall](2026-11-02-hierarchy-on-synthetic-recall.md)). Stage 1's T1 attention has a 1,024-position window, about 3,000 bytes, more than a whole document (about 1,000 bytes).

## Hypothesis

The main network is read when it has a job the outer stages cannot do. With stage 1's window cut to 8 positions (about 24 bytes, 3 to 4 words), only the main network's global attention can look across a document, so the main network takes the memory.

## Predictions

Main prediction (only this one answers the Question):

- With stage 1's window at 8, shuffling the main network's output (`coarse_shuffle_s1`) raises recall val bpb at step 6,000 by more than it raises the same-window no-recall control's, in every seed (0, 1, 2), above the control's largest value over its three seeds. All three: supported. Any at or below: not supported.

Other predictions (at most two):

- Recall val bpb at step 6,000 is below the no-memory anchor (2.006) in all three seeds.
- The stage-1 shuffle − zero is above zero in all three recall seeds.

If supported, we do: the layout decides whether the main network is used. On text, our main network reaches only 1.33x further than stage 1 (4,096-byte context, stage-1 window about 3,000 bytes). Next, on text: give the main network a range stage 1 lacks (a smaller stage-1 window, or a longer context) and test it on our dead seeds at 786M bytes, against their controls.

If not supported, we do: read where the memory went. If recall bpb is below 2.006 and the main network's shuffle rise is not, a Mamba-2 stack (stage 1's or stage 0's) holds the memory. Then test the same change with stage 1's Mamba-2 blocks replaced by attention of the small window. If recall bpb is not below 2.006, nothing learned the memory: the main network's path does not learn a recall in 6,000 steps, and we look at its gradients.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`).
- Ratios within 15% of 3 x 3 at the end. Finite loss. 393,216,000 training bytes per arm.
- Oracle lower bound: recall val bpb at or above 1.021, no-recall at or above 2.005.

## Method

```
data: synth/recall and synth/norecall, unchanged (scripts/make_synth.py, seed 0)
model: reflayout_2stage with model.stages.1.window=8 (was 1024): stage 1's T1 attention in the encoder and the
       decoder sees the last 8 stage-1 positions; stage 1's Mamba-2 blocks and the main network are unchanged
       (main: global attention over about 455 positions = the whole 4,096-byte row)
train: as the window-1024 arms: configs/train/ladder.yaml, 6,000 steps, no rungs, eval every 1,000 steps
arms: data=synth_recall seeds 0, 1, 2 ; data=synth_norecall seeds 0, 1, 2
grade: probe/coarse_shuffle_s1 at step 6,000, recall against no-recall
reference (not rerun): the window-1024 arms of hierarchy-on-synthetic-recall
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, two calls in sequence:
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/main-network-takes-recall/recall main-network-takes-recall reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.1.window=8 train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000`
  - the same with `norecall` in the log dir and `data=synth_norecall`
- wandb project `hnet-router`, group `main-network-takes-recall`.
- Cost: about 12 min per arm at the 500 W cap, about 75 min in total. Runs after [coarse JEPA on dead seeds](../refuted/2026-11-02-coarse-jepa-on-dead-seeds.md).
- The smaller window cuts stage 1's attention FLOPs a little. Fwd FLOPs/byte is logged; it is a measurement here.

## Results

All six arms ran to step 6,000 (2026-11-02 18:55 to 20:03; commit d5a41ff). Log dir `ignored/runs/main-network-takes-recall/{recall,norecall}`. wandb: recall seeds 0, 1, 2 `kesyeo0s`, `wj6mgvzg`, `ngrk28y1`; no-recall `clr7nxvs`, `6vq116xu`, `0lmzu7mh`. The Hydra config of each arm shows stage 1 `window: 8`. Artifacts: [results/](../results/2026-11-02-main-network-takes-recall/) (`eval-curves.csv`, made by `pull.py`).

### Gates

| gate | result |
|---|---|
| suite (`HNET_TEST_CUDA=1`): 156 passed, 1 skipped (run after the arms, same model code) | pass |
| ratios at the end (3 x 3 ±15%): 2.95 to 3.08 x 2.98 to 3.04 | pass |
| 393,216,000 training bytes per arm; finite loss | pass |
| oracle lower bound: recall 1.0790 to 1.0833 ≥ 1.021; no-recall 2.0087 to 2.0089 ≥ 2.005 | pass |

Added after the run, because the reading of this result rests on it: `tests/test_attention.py` checks that a sliding-window query at q sees exactly keys q − w + 1 .. q (window 1, 8, 13; dense mask on CPU passes, flex on CUDA under `HNET_TEST_CUDA=1`). Fwd FLOPs/byte fell from 22.52 to 22.66M (window 1024) to 21.72 to 22.33M (window 8).

### At the end (step 6,000)

| data | seed | val bpb | `zero_s0` | `shuffle_s0` | `zero_s1` | `shuffle_s1` | stage-1 shuffle − zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| recall | 0 | 1.0790 | 8.638 | 0.475 | 0.0012 | 2.8e-5 | −0.0012 |
| recall | 1 | 1.0810 | 4.634 | 0.588 | 0.0897 | 3.4e-5 | −0.0897 |
| recall | 2 | 1.0833 | 8.754 | 0.656 | 0.2886 | **−1.2e-6** | −0.2886 |
| no recall | 0 | 2.0089 | 2.446 | 0.000 | 0.0000 | −2.4e-6 | 0.0000 |
| no recall | 1 | 2.0088 | 3.692 | 0.000 | 0.0063 | 2.4e-6 | −0.0063 |
| no recall | 2 | 2.0087 | 2.698 | 0.000 | 0.0000 | 9.3e-6 | 0.0000 |

Anchors: recall oracle 1.021, no-memory 2.006; no-recall oracle 2.005. Window 1024 reference ([hierarchy on synthetic recall](2026-11-02-hierarchy-on-synthetic-recall.md)): recall bpb 1.0731 to 1.0908, `shuffle_s0` 0.74 to 0.95, `shuffle_s1` 0.0000.

- Main prediction (recall `coarse_shuffle_s1` above the no-recall maximum, 9.3e-6, in every seed): 2.8e-5, 3.4e-5, −1.2e-6. Seed 2 is below. **Not supported.** All six values are at the eval's noise level (about 1e-5 bpb); none is a content use.
- Other prediction 1 (recall val bpb below 2.006 in all seeds): holds (1.0790, 1.0810, 1.0833).
- Other prediction 2 (recall stage-1 sign above zero): does not hold (−0.0012, −0.0897, −0.2886).

### What the runs show (observations after the fact)

- **The model learned the full memory without the main network, and without any long-range attention.** Recall bpb is as low as with window 1024. Stage 1's attention now sees about 24 bytes, so the memory is in Mamba-2 state: stage 1's 8 Mamba-2 blocks and stage 0's 8.
- **Stage 0 took part of the memory back.** Shuffling stage 0's inner output cost 0.48 to 0.66 bpb, against 0.74 to 0.95 at window 1024, for the same total value of memory (about 0.93). So stage 0's own Mamba-2 stack holds the rest.
- **A category error in the plan, again.** The plan treated "range" as what the main network has and stage 1 lacks. Mamba-2 has unlimited range. A 32-word set is about 32 x 13 = 416 bits, and one Mamba-2 layer at d 256 (expand 2, d_state 128) carries a state of 65,536 numbers. The main network's true edges are exact retrieval over all positions (no compression into a state) and compute per coarse unit. This task needs neither.
- The stage-1 sign misleads again: `zero_s1` is up to 0.29 in recall seed 2 while `shuffle_s1` is 0.

## Verdict

**Not supported.** With stage 1's attention window at 8, all three recall seeds learn the memory (val bpb 1.079 to 1.083), and the main network carries none of it (`coarse_shuffle_s1` −1.2e-6 to 3.4e-5, the same as the no-recall control). The Mamba-2 stacks of stages 0 and 1 hold it.

## Updated hypothesis

The main network is read only for a job that the outer Mamba-2 stacks cannot do. Range is not such a job, because Mamba-2 has unlimited range. Two jobs are: exact retrieval of more content than a recurrent state holds, and computation over coarse units. A recall language with a small set is neither. The clean test of whether our main path can carry content at all: make every outer block a small-window attention block (no Mamba-2 at stages 0 and 1), so the main network is the only path across a document, on the same recall data. This is the "if not supported" step of this plan, widened from stage 1 to both outer stages because stage 0 took part of the memory here. If the main network takes the memory then, our main path works, and the text result means redundancy: at our budget, text has little content that the main network gives and the Mamba-2 stacks do not. If it does not, our main path has a defect, and we look at its gradients.

## Related

- [Hierarchy on synthetic recall](2026-11-02-hierarchy-on-synthetic-recall.md)
- [Coarse JEPA on dead seeds](../refuted/2026-11-02-coarse-jepa-on-dead-seeds.md)
