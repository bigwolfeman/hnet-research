# Experiment: hierarchy on synthetic recall

Status: refuted

Date proposed: 2026-11-02
Algorithm: the recall language and its exact oracle, `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/synth.py` (pseudocode in its docstring)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

When the data needs a document-level memory, does our 2-stage H-Net's main network come online in a short budget, in every seed?

Context: on text, our code with the reference configuration brings stage 0 online in every seed by 3.15B bytes, but the stage-1 signature (the stage-1 decoder reads the main network's content) is positive in only 1 of 3 seeds there ([dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md)). Two readings fit. (a) On text, the main network's content is worth little until the outer stages have learned the local statistics, so stage 1 wakes late. (b) Something in our stage-1 path stops the decoder from reading the main network even when its content is worth a lot. A cheap language with a large, known payoff for memory separates the two.

## Hypothesis

The main network wakes up as soon as its content pays. In the recall language, a word slot is one of the document's 32 words, so after the first few words of a document a memory of the words seen is worth about 1 bpb (oracle 1.021 against 2.006 with no memory). Attention over word-sized chunks is the cheapest place to hold that memory, so the main network comes online early in every seed.

## Predictions

Main prediction (only this one answers the Question):

- On the recall data, our 2-stage H-Net has the stage-1 shuffle − zero above zero at the end of a 6,000-step run (393M bytes) in all three seeds, 0, 1, and 2. All three: supported. Any at or below zero: not supported.

Other predictions (at most two):

- On the recall data, val bpb is below the no-memory anchor (2.006) in all three seeds: the model uses a document memory somewhere.
- In each seed, the stage-1 shuffle − zero on the recall data is above the same seed's value on the no-recall control data (where a memory is worth nothing).

If supported, we do: our stage-1 path works when the main network's content pays, and the slow stage 1 on text is about when the content pays (reading a). Cheap synthetic data then becomes the test bench for speeding that up, and the JEPA objective (planned next) is graded on text, where the open problem is.

If not supported, we do: read where the memory went. If the stage-0 sign is positive and the stage-1 sign is not, stage 1's own attention layer (T1, a 1,024-position window that covers a whole document) holds the memory and the main network is not needed for it: rerun with the recall span beyond stage 1's window. If neither sign is positive while bpb is below the no-memory anchor, the outer Mamba-2 stack holds the memory. If bpb is not below the anchor, nothing learned the memory in 6,000 steps; rerun longer. Each case points at a different part of our stage-1 path.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including `tests/test_synth.py` (the oracle sums to 1 at every position, for both languages).
- Ratios within 15% of 3 x 3 at the end. Finite loss. 6,000 x 65,536 = 393,216,000 training bytes per arm.
- The oracle is a lower bound: val bpb at or above 1.021 on the recall data and at or above 2.005 on the no-recall data. A run below its oracle has a data or eval bug.

## Method

```
data (scripts/make_synth.py --out $HNET_DATA_DIR/synth, seed 0; 400M train bytes, 8M val bytes per language):
  lexicon   8,192 distinct random words, length uniform 3..8, letters uniform a..z (shared by both languages)
  recall    per document: 32 lexicon words without replacement; each slot uniform over them
  norecall  each slot uniform over the lexicon
  delimiter after every word: SEP (document end) with prob 1/150, else " "  -> documents of about 1,000 bytes
  anchors (first 2M val bytes, full-document context): recall oracle 1.0215, no-memory 2.0061; norecall oracle 2.0049
model: reflayout_2stage (our code, reference configuration; 41.8M params), unchanged
train: configs/train/ladder.yaml, max_steps 6,000, no ladder rungs, eval every 1,000 steps (coarse-path probe on 8
       val batches), WSD decay over the last 20%
arms: data=synth_recall seeds 0, 1, 2 ; data=synth_norecall seeds 0, 1, 2
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 6,000, recall arms
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, two calls in sequence:
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/hierarchy-on-synthetic-recall/recall hierarchy-on-synthetic-recall reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000`
  - the same with `norecall` in the log dir and `data=synth_norecall`
- wandb project `hnet-router`, group `hierarchy-on-synthetic-recall`.
- Cost: about 11 min per arm, about 70 min in total.
- The oracle sees the whole document. A model's 4,096-byte window can start inside a document and miss some of its words, so the model cannot reach the oracle on those positions. The anchors are fixed lower bounds, not targets.

### Amendment 2026-11-02 17:20 (recall seed 2 stopped, rerun)

- The user stopped recall seed 2 at its start (16:32, signal 9, power budget), before any eval. It reruns from step 0 with the same command after the no-recall arms.
- From 16:35 the GPU has a 500 W power cap (power budget). Arms before that ran uncapped. Speed is a measurement, not a gate.
- What I saw before this amendment: the final val bpb of recall seeds 0 and 1 (1.0731, 1.0753) and of the three no-recall seeds (2.0088 each), and recall seed 0's val bpb at steps 2,000 and 3,000 in its log. I saw no probe value. The Predictions do not change.

## Results

All six arms ran to step 6,000 (2026-11-02 16:08 to 17:31; commit d05b63d, recall seed 2 rerun at b3792e5 with the same model code). Log dir `ignored/runs/hierarchy-on-synthetic-recall/{recall,norecall}`. wandb: recall seeds 0, 1, 2 `wwxztp1h`, `1j2fmt9i`, `cn70a6an`; no-recall `xl9t4tqm`, `pgr6nm6w`, `7pmfbi4s` (the stopped recall seed 2 attempt is `8l9krdk5`, state crashed, not used). Artifacts: [results/](../results/2026-11-02-hierarchy-on-synthetic-recall/) (`*-summary.json`, `eval-curves.csv`; wandb names repeat across the two languages, so the curve file prefixes the arm with the data dir).

### Gates

| gate | result |
|---|---|
| suite (`HNET_TEST_CUDA=1`): 151 passed, 1 skipped at launch (155 at the seed 2 rerun, with the JEPA tests) | pass |
| ratios at the end (3 x 3 ±15%): 2.99 to 3.03 x 2.94 to 3.03 | pass |
| 393,216,000 training bytes per arm; finite loss | pass |
| oracle lower bound: recall 1.0731 to 1.0908 ≥ 1.021; no-recall 2.0088 ≥ 2.005 | pass |

### At the end (step 6,000)

| data | seed | val bpb | `zero_s0` | `shuffle_s0` | `zero_s1` | `shuffle_s1` | stage-1 shuffle − zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| recall | 0 | 1.0731 | 8.887 | 0.952 | 0.0020 | 0.0000 | **−0.0020** |
| recall | 1 | 1.0753 | 3.616 | 0.791 | 0.0071 | 0.0000 | **−0.0071** |
| recall | 2 | 1.0908 | 12.080 | 0.740 | 0.2097 | 0.0000 | **−0.2097** |
| no recall | 0 | 2.0088 | 2.076 | 0.000 | 0.0000 | 0.0000 | 0.0000 |
| no recall | 1 | 2.0088 | 4.213 | 0.000 | 0.0000 | 0.0000 | 0.0000 |
| no recall | 2 | 2.0088 | 2.109 | 0.000 | 0.0000 | 0.0000 | 0.0000 |

Anchors: recall oracle 1.021, no-memory 2.006; no-recall oracle 2.005. The recall arms learned the memory between steps 2,000 and 4,000 (val bpb 2.08 → 1.10 to 1.16), in a different step for each seed.

- Main prediction (recall, stage-1 shuffle − zero above zero in seeds 0, 1, 2): −0.0020, −0.0071, −0.2097. **Not supported.**
- Other prediction 1 (recall val bpb below the no-memory anchor 2.006 in all seeds): holds (1.0731, 1.0753, 1.0908).
- Other prediction 2 (recall stage-1 sign above the no-recall sign, per seed): does not hold. The no-recall arms are at 0.0000; the recall arms are below.

### What the runs show (observations after the fact)

- **The memory is inside the hierarchy, and not in the main network.** Shuffling stage 0's inner output (stage 1 plus main) with another document's costs 0.74 to 0.95 bpb on the recall data, about the whole value of memory (2.006 − 1.07 = 0.93). Shuffling the main network's output costs 0.0000 in every seed. The stage-1 decoder ignores the main network's content completely. Stage 1's own layers hold the memory: its T1 attention (1,024 positions, about 3,000 bytes at 3 bytes per position, which covers the whole ~1,000-byte document) and its Mamba-2 blocks.
- **This is a category error in the test, and maybe in our setup.** At our 4,096-byte context, stage 1's attention window covers 3,000 bytes. The main network's global attention reaches only 1.33x further. On this task the main network has no job that stage 1 cannot do. The plan named this case before the run ("stage 1's own attention layer ... holds the memory and the main network is not needed for it").
- **The stage-0 sign misleads here.** Stage-0 shuffle − zero is far below zero (−2.1 to −11.3) while the shuffle rise is 0.74 to 0.95 bpb, a large content use. Zeroing a stage's inner output gives the decoder an input it never saw in training, so `zero` can be huge whether or not content is used. The sign of shuffle − zero is a valid content test only against a control where content does not pay (here the no-recall arms, at 0.000). Earlier experiments graded on the sign. Their stage-0 readings need the shuffle rise next to them.

## Verdict

**Not supported.** With a document memory worth about 0.93 bpb, all three seeds learn it, but the main network carries none of it (`coarse_shuffle_s1` 0.0000; stage-1 sign −0.0020, −0.0071, −0.2097). Stage 1's own layers hold the memory.

## Updated hypothesis

The main network wakes up only when it has a job that stage 1 cannot do. With a 4,096-byte context and stage 1's 1,024-position attention window (about 3,000 bytes), stage 1 can reach almost everything the main network can. Then the main network's only edge is depth and width, and next-byte prediction gives it little reason to be read. That fits the text results too: stage 1 comes online late or not at all in our code. The test: shrink stage 1's attention window so that only the main network can attend across a document, on the same recall data. If the main network then takes the memory, the layout (window against context) decides whether the main network is used, and the text design must give the main network a range that stage 1 lacks: [main network takes recall](../refuted/2026-11-02-main-network-takes-recall.md).

## Related

- [Dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md)
- [Main-network seed rate](../refuted/2026-11-02-main-network-seed-rate.md)
