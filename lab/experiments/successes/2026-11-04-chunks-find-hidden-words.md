# Experiment: chunks find hidden words

Status: success

Date proposed: 2026-11-04
Algorithm: `lab/spikes/2026-10-30-two-stage-hnet-smoke/scripts/chunk_semantics.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

When the spaces between some words are deleted, does the stage-0 router of our trained 2-stage H-Net still cut at the hidden word boundaries?

Context: our 2-stage reference layout now trains to 12.58B bytes, uses its main network at stage 1, and ends below BPE on edu in both seeds ([reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md)). The H-Net authors claim the learned chunks are linguistic units, but they show it with pictures only (arXiv 2507.07955):

- Sec 3.1: "The first stage of H-Net (2-stage) combines spacelike boundaries with first few characters of each word."
- Fig 4: the inner stage "chunks the text into more meaningful units, such as words or numberings", and "it often chunks multiple words that form one semantic group".
- Goomba Lab blog, Part I: "it tends to draw boundaries when surprised".
- Fig 16: with some whitespace removed, "H-Net detects word boundaries even if they are not explicitly separated by whitespaces".

A router that has only learned "cut near a space" also passes a whitespace-alignment probe. The deleted-space test separates the two: with the space gone, only knowledge of words can place the cut.

Data seen before this plan (it is not graded here): at the end of A seed 0, `probe/align_s0` F1 against whitespace word boundaries is 0.53 (fixed-stride chunker at the same cut rate: 0.31), and `probe/align_s1` F1 is 0.38 (stride 0.17, precision 0.72). The per-domain ratios of both seeds. I have not looked at the alignment probes over training, at any rung checkpoint, or at any measure below.

## Hypothesis

The stage-0 router places cuts from what the encoder knows about words, not from the space byte alone. Inside a run of letters with no space, the cut rate at a hidden word start is higher than at the same position inside a real word.

## Predictions

Terms:
- A word is a maximal run of non-whitespace bytes. Its position k counts from 0 at its first byte.
- A cut at index c means a stage-0 chunk starts at byte c (`hnet/probes.py stage_cuts`).
- A join is the first byte of the second word, where a single space between two ASCII letters was deleted. Each such space is deleted with probability 0.5 (fixed seed). In the edited text the join sits at position k = the length of the first word, inside a longer run.
- The matched control: bytes at the same position k inside intact words longer than k, in the same edited text. Each join's k is matched, and the control rate is the mean over joins of the intact-word cut rate at that join's k.

Main prediction (only this one answers the Question):

- At 12.58B bytes, the stage-0 cut rate at joins is higher than the matched control rate, in both seeds (0 and 1). Anchor: equal rates, which is what a router that reacts only to whitespace gives. Both higher: supported. Either equal or lower: not supported.

Other predictions (at most two):

- On intact text at 12.58B bytes, the stage-0 cut rates at word positions 0, 1, and 2 are each higher than the pooled rate at positions 3 and up, in both seeds ("spacelike boundaries with first few characters of each word").
- On intact text at 12.58B bytes, at each word position k = 1 to 4, the model's own byte loss is higher on bytes where a stage-0 chunk starts than on bytes where none starts, in both seeds ("boundaries when surprised"). Only cells with 100 or more bytes on each side count.

If supported, we do: the stage-0 router uses word knowledge, as the authors claim. Then the open semantic claim is stage 1 (multi-word "semantic groups"). Write the stage-1 test with a fixed reference (for example, cuts at phrase boundaries from a parser) and bring the renders to the user.

If not supported, we do: read the per-k rates. If the join rate equals the control, the stage-0 router follows the space byte, and its 0.53 whitespace F1 is a whitespace rule. That bears on Phase 1's "semantic chunks" goal, and it goes to the user with the readings below before any design change.

Gates (the run is broken if one fails; gates are not predictions):

- Checkpoints load and reproduce the logged final edu val bpb within 0.002 on the training eval batches (seed 0 1.0905, seed 1 1.0979).
- The edited text has at least 5,000 joins. Joins whose k has fewer than 100 control bytes are dropped, and they are under 2% of joins.
- The cut rate on intact text is within 10% of the logged `probe/align_s0_cuts_per_byte` for seed 0 (0.315).

## Method

```
checkpoints: reference-layout-vs-bpe seeds 0, 1: ladder_12000.pt (0.79B), ladder_48000.pt (3.15B), final.pt (12.58B)
text:        128 edu val rows of 2,048 bytes that each start at a document start (scripts/sample.py doc_starts, seed 0)
edit:        for each 0x20 byte with ASCII letters on both sides: delete with p = 0.5 (rng seed 0); record joins
run:         model(x) in eval, bf16 autocast; cuts per stage from aux; byte loss from the logits (bits)
measures:    join rate vs matched control (main), per-k cut rates (other 1), per-k loss at cut vs no-cut (other 2)
readings:    the same measures at 0.79B and 3.15B (emergence); stage-1 join rate and matched control;
             words per stage-1 chunk; the 40 most frequent multi-word stage-1 chunks;
             probe/align_s{0,1} model and stride F1 over training from wandb (12 points per seed);
             3 rendered passages (intact and edited) with stage-0 and stage-1 cuts
```

- Inference only, about 2 minutes on the GPU. It runs after Parcae coarse main ends, so it does not share the GPU with a training run.
- Artifacts: `lab/experiments/results/2026-11-04-chunks-find-hidden-words/`.

### Amendment 2026-11-04 13:05 (before any run on A's checkpoints)

- Readings added, not predictions: the semantic chunking score of `hnet/semantics.py` per checkpoint and stage (router-p AUROC at five levels: word, hidden, morph, phrase, gotok; anchor 0.5), and `chunks.html`, the passages drawn as in the paper's Fig 4 (stage-0 boxes, stage-1 outlines, hidden word starts marked). Reason: the user asked for a hard metric for future evals and for pictures to check by eye. Note: [semantic chunking score](../../../.agents/notes/implemented/testing/2026-11-04-semantic-chunking-score.md).
- The Predictions do not change. The hidden-word AUROC measures the same question as the main prediction with the router's `p` instead of its cuts. It is a reading here, and the main prediction stays the cut-rate comparison.

### Amendment 2026-11-04 14:00 (first run broken by a gate; rerun)

- The first run (13:46, code 977a492) failed the dropped-joins gate: 16.5% of joins were dropped, against the 2% gate. Cause: merged chains ("thecatsat") put joins at high positions k, where intact words longer than k are rare, so those joins had fewer than 100 control bytes. Fix: k is capped at 8 for both joins and controls (positions 8 and up pool into one stratum), as `hnet/semantics.py` already does. The rule and the Predictions do not change. The rerun reads the same checkpoints, rows and edits.
- What I saw from the broken run: the main measure in both seeds at every checkpoint, the per-position rates, the surprisal table, the stage-1 units, the semantic scores, and the alignment curves. So the readings in Results come from a run whose numbers I saw once before the fix.
- Reading fix, from the same run: the `word` and `phrase` levels of the semantic score put the candidate on the first letter of a word. Stage 0 mostly starts its chunks at the space before the word (" the"), so the score missed those cuts. Both levels now score each word start over its gap (the whitespace run and the first letter, max p). The other levels do not change.

## Results

Run `scripts/chunk_semantics.py` on the six checkpoints (rerun after the 14:00 amendment, code at the commit that files this record), GPU, about 4 min. Artifacts in [results/](../results/2026-11-04-chunks-find-hidden-words/): one JSON per checkpoint, `run.txt` (the printed summary), `wandb-align.json` (alignment probes over training), and `chunks.html` (the passages, drawn as in the paper's Fig 4).

### Gates

- Edu val bpb on the training eval batches: 1.3184, 1.1694, 1.0905 (seed 0) and 1.3587, 1.2252, 1.0979 (seed 1), within 0.0001 of the logged values.
- 17,700 joins. Dropped joins: 0% (16.5% in the broken first run, see the amendment).
- Seed 0 intact-text stage-0 cut rate 0.313, against the logged 0.315.

### Main prediction: stage-0 cut rate at hidden word starts, 12.58B bytes

| seed | join rate | matched control | per position k = 1, 2, 3, 4, 5, 6, 7, 8+ (join / control) |
|---|---:|---:|---|
| 0 | **0.718** | 0.227 | .61/.22, .81/.44, .82/.22, .71/.16, .68/.14, .64/.15, .66/.18, .71/.24 |
| 1 | **0.633** | 0.271 | .82/.45, .94/.47, .74/.27, .58/.16, .58/.16, .54/.18, .53/.22, .58/.29 |

The join rate is 2.3 to 3.2 times the control in both seeds, and higher at every position.

### Other predictions

- Stage-0 cut rates at word positions 0, 1, and 2 each above the rate at 3 and up: does not hold. Seed 0 holds (0.184, 0.206, 0.413 against 0.140). In seed 1, position 0 (0.175) is below 3 and up (0.182), while 1 and 2 are high (0.435, 0.449). Both seeds start most word chunks at the space before the word: the cut rate on whitespace bytes is 0.873 (seed 0) and 0.516 (seed 1).
- At each position k = 1 to 4, the byte loss is higher where a stage-0 chunk starts: holds in both seeds, by a factor of 1.9 to 6.3 (seed 0: 2.00/1.04, 1.42/0.59, 1.35/0.42, 1.64/0.26 bits; seed 1: 1.69/0.93, 1.27/0.68, 0.97/0.48, 1.44/0.29). Every cell has more than 2,000 bytes on each side.

### Readings: semantic chunking score (router-p AUROC; 0.5 = chance)

| level | seed 0: 0.79B / 3.15B / 12.58B | seed 1: 0.79B / 3.15B / 12.58B |
|---|---|---|
| word, stage 0 | 0.913 / 0.931 / 0.942 | 0.628 / 0.670 / 0.670 |
| hidden, stage 0 | 0.754 / 0.780 / 0.805 | 0.677 / 0.736 / 0.747 |
| morph, stage 0 | 0.488 / 0.511 / 0.573 | 0.431 / 0.460 / 0.530 |
| gotok, stage 0 | 0.869 / 0.966 / 0.943 | 0.842 / 0.890 / 0.849 |
| word, stage 1 | 0.703 / 0.719 / 0.718 | 0.695 / 0.681 / 0.672 |
| phrase, stage 1 | 0.566 / 0.499 / 0.495 | 0.605 / 0.528 / 0.665 |

Stage 1 scores only the positions where stage 0 cut (`seen_frac` 0.23-0.31 on the word levels).

### Other readings

- **Hidden-word cutting grows with training; whitespace alignment does not.** Join rate 0.518, 0.588, 0.718 (seed 0) and 0.363, 0.466, 0.633 (seed 1) at 0.79B, 3.15B, 12.58B. The old probe's model − stride F1 is flat from step 16,000 in both seeds (seed 0 0.21-0.24, seed 1 0.03-0.05).
- **The old probe misreads seed 1.** Its stage-0 F1 gap is 0.04, because seed 1 often cuts after the first letter of a word, not at an exact edge. Its hidden-word lift (0.633 against 0.271) is close to seed 0's.
- **Stage-1 chunks are words and short function-word groups.** 53-58% hold one word and 25-28% hold two. The most frequent multi-word chunks are " it is", " the first", " as well as", ", according to", " in the United States", ". For example". Many end after the first letters of the next word (" able t", " more tha"), as the authors describe for stage 0.
- **Morphemes are near chance.** Stage 0 does not rank gold morpheme boundaries above other in-word positions until 12.58B bytes (0.53-0.57), and then only a little.
- **Stage 1 does not follow constituents.** The phrase AUROC moves between 0.50 and 0.67 with no trend.
- Every stage-1 cut is also a stage-0 cut (fraction 1.0), as the design requires.

## Verdict

**Supported.** With the spaces deleted, stage 0 cuts at hidden word starts at 0.718 and 0.633, against 0.227 and 0.271 at the same positions inside intact words, in both seeds at 12.58B bytes.

## Updated hypothesis

The stage-0 router knows where words start without the space, as the authors claim (their Fig 16), and the knowledge grows with training bytes while whitespace alignment stays flat. Its cuts follow the model's own surprise. The other claims hold less: the "first letters" pattern depends on the seed (seed 1 starts its chunks after the first letter), morpheme boundaries are near chance, and stage 1 groups words into short function-word units, not into constituents. So "semantic" is accurate for words and code tokens and not yet for anything above the word. The semantic chunking score now logs these levels in every run, and the phrase level is the one to watch when the 2-stage model trains longer.

## Related

- [Reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md)
- [Reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md)
