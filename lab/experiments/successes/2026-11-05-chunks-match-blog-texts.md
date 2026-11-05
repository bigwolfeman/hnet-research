# Experiment: chunks match blog texts

Status: success

Date proposed: 2026-11-05
Date run: 2026-11-05
Algorithm: `lab/spikes/2026-10-30-two-stage-hnet-smoke/scripts/blog_texts.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)
Artifacts: [results/2026-11-05-chunks-match-blog-texts](../results/2026-11-05-chunks-match-blog-texts/)

## Question

On the blog's own example texts, do our 2-stage H-Net's stage-1 chunks fall where the blog's H-Net puts its stage-1 chunks, more often than a fixed-stride chunker at our rate?

Context: constitution 1.2.0 checklist items 3, 6 and 7 ask for word-level boundaries, morpheme-like stage-0 cuts, and in-context tokenization. So far we graded them only on our own probe text. The blog's figure data (Figs 10a, 10b, 11; recovered to `ignored/blogs-raw/figdata/`, see its README) gives the exact texts with the blog H-Net's stage-0 and stage-1 chunks, and for Fig 10 the original 2025 H-Net's ("LH-Net") chunks too. Our router is the 2025 lookahead router. No model of ours has been run on these texts before this file.

## Hypothesis

Our stage 1 learned word-level units, as the blog's stage 1 did: its chunks start where the blog's start, up to the space-or-letter convention. Our stage 0 is an LH-Net stage 0: inside words it cuts where LH-Net cuts, not at the new H-Net's morpheme points. In-context tokenization is weak or absent in our model (own probe: 4.70 to 4.65 chunks per new word over 5 repeats).

## Predictions

Terms:
- **Cut**: a chunk starts at byte c. The blog H-Net stores chunk-end bytes; the figdata converts them to starts.
- **Gap rule**: a cut on a whitespace byte moves to the next non-whitespace byte. Our chunks start at the space (" the") and the blog H-Net's start after it ("the "), so without the rule the same word boundary would count as a miss. The rule applies to every chunker alike.
- **F1**: boundary F1 of a chunker's cuts against the blog H-Net's cuts, after the gap rule, pooled over the texts.
- **Texts**: the English examples of Fig 10a (3) and the Go examples of Fig 10b (3). Our model trained on English and Go only.
- **Stride**: cuts every r bytes, with r = our measured bytes per stage-1 chunk on the same texts.

Main prediction (only this one answers the Question):

- In both seeds (A's seeds 0 and 1 at 12.58B bytes), our stage-1 F1 against the blog H-Net's stage-1 cuts is higher than the stride F1. Both seeds: supported. Either seed at or below stride: not supported.

Other predictions (at most two):

- Fig 11 "New words" (borvexila, mipralune, velnorith): for each word, our stage-0 chunks inside its last occurrence are fewer than inside its first. All 3 words in both seeds: holds. The count is 1 + cuts strictly inside the word's bytes (as `probes.in_context`).
- English Fig 10a, in-word cuts only (both neighbors non-whitespace): our stage-0 in-word F1 is higher against LH-Net's in-word cuts than against the new H-Net's, in both seeds.

If supported, we do: item 3 holds on the blog's texts. The stage-1 unit that Phase 2 loops over is word-level in the blog's sense. Grade items 6 and 7 from the other predictions and report them.

If not supported, we do: read the per-language F1. If Go fails and English holds, our code chunks differ from the blog's (their H-Net merges whitespace runs into code chunks). If English fails, our stage-1 chunks are not word-level on clean prose, and item 3 is not met. Bring it to the user before the seeds-vs-BPE run.

Gates (the run is broken if one fails; gates are not predictions):

- Every figdata text re-encodes to its stored `n_bytes`. Blog stage-1 cuts are a subset of blog stage-0 cuts.
- Our checkpoints load strictly, and their edu val bpb on the training eval batches is within 0.001 of the logged value (1.0905, 1.0979).
- The stride chunker's rate is within 2% of our stage-1 rate on each text set.

## Method

```
code:     lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
models:   ignored/runs/reference-layout-vs-bpe/reflayout_2stage_s{0,1}/final.pt (12.58B bytes)
          readings only: ladder_12000.pt (0.79B) and ladder_48000.pt (3.15B) of the same seeds
texts:    figdata fig10a (src English), fig10b (src Go), fig11 (all 4); one text per batch row, no prefix byte
          (as probes.in_context); readings also on the other fig10a/10b languages and fig12 (out of domain for us)
cuts:     ours = probes.stage_cuts per stage; blog = models.hnet_stage{0,1}.chunk_starts, lhnet_stage{0,1}.chunk_starts
gap rule: canon(c) = first non-whitespace byte at or after c, if byte c is whitespace; else c. Drop c = 0 and c = len.
main:     F1(ours s1, blog s1) vs F1(stride at our s1 rate, blog s1), pooled over English + Go, per seed
other 1:  fig11 "New words": chunks inside occurrence k = 1 + cuts in (b0, b1); compare k = first vs k = last
other 2:  in-word cut = canon cut with non-whitespace bytes on both sides; F1(ours s0, LH-Net s0) vs F1(ours s0, H-Net s0)
readings: stage-0 F1 of the same kind; per-language F1; the word chunker (every word start) against the blog;
          fig11 chunk counts per example against the blog's; F1 at the 0.79B and 3.15B rungs; side-by-side
          html (blue / orange / gray) of ours vs the blog H-Net vs LH-Net on each English and Go text
```

- GPU inference only, a few minutes. Artifacts go to `lab/experiments/results/2026-11-05-chunks-match-blog-texts/`.

## Results

Run: `scripts/blog_texts.py` on A's six checkpoints, GPU, under 2 min, at commit fb1dc2a. Artifacts in [results/](../results/2026-11-05-chunks-match-blog-texts/): `summary.json`, one JSON per checkpoint, and a pointer to `blog_texts.html` (ours vs the blog H-Net vs LH-Net on every English, Go and Fig 11 text). The HTML holds the blog's figure texts, so it stays local in `ignored/experiment-artifacts/` and is not redistributed; `scripts/blog_texts.py` rebuilds it from the blog pages.

### Gates

- All 110 figdata texts re-encode to `n_bytes`. Blog stage-1 cuts are a subset of stage-0 cuts in every record.
- Edu val bpb 1.0905 (seed 0) and 1.0979 (seed 1), the logged values.
- Stride chunk count / our chunk count: 1.001 in every run.

### Main prediction: stage-1 F1 against the blog H-Net, English + Go pooled, 12.58B bytes

| seed | ours | stride at our rate | word chunker (reading) | ours bytes/chunk | blog bytes/chunk |
|---|---:|---:|---:|---:|---:|
| 0 | **0.587** | 0.269 | 0.890 | 8.59 | 5.84 |
| 1 | **0.376** | 0.272 | 0.890 | 8.96 | 5.84 |

Both seeds are above stride.

### Other predictions

- Fig 11 "New words", fewer stage-0 chunks in the last occurrence than the first: **does not hold**. Ours is flat in every word and both seeds: seed 0 borvexila 4,4,4,4,4, mipralune 5,5,5,5, velnorith 4,4,4,4; seed 1 5,5,5,5,5 / 3,3,3,3 / 4,4,4,4. The blog H-Net goes 3,3,2,1,1 / 4,1,1,1 / 3,3,2,1. The rungs (0.79B, 3.15B) are flat too.
- English in-word stage-0 F1 higher against LH-Net than against the new H-Net: **holds** in both seeds (seed 0: 0.543 vs 0.223; seed 1: 0.421 vs 0.176). Go shows the same order (0.598 vs 0.410; 0.538 vs 0.357).

### Readings

- **Seed 0 agrees with the blog's H-Net as well as the blog's own LH-Net does.** English stage 1: ours vs blog H-Net 0.651, LH-Net vs blog H-Net 0.642. Ours vs LH-Net 0.723. Seed 1 is weaker: 0.375 vs blog, 0.608 vs LH-Net.
- **The blog's stage 1 is nearly a word chunker on these texts** (word chunker F1 0.927 English, 0.855 Go). Our stage 1 is coarser: on Go, 10.2-11.1 bytes per chunk against the blog's 4.95, with recall 0.28-0.39 at precision 0.58-0.76. Our Go stage-1 cuts are mostly at blog boundaries, but we skip many of them.
- **Repetitive input breaks our stage 0.** Digit blocks (864 bytes): our stage 0 is 1 chunk at 0.79B and 863 chunks (one per byte) at 12.58B in both seeds; stage 1 is 1 chunk. The blog: 233 and 14, near-clean 4-digit blocks. Letter blocks: ours 265-324 stage-0 chunks against the blog's 119. Repeated formats: ours 338-359 / 91-105 against 182 / 50.
- Stage-1 F1 over training (English): seed 0 0.597, 0.638, 0.651; seed 1 0.375, 0.398, 0.375 at 0.79B, 3.15B, 12.58B. Flat after 0.79B.

## Verdict

Supported. Our stage-1 F1 against the blog H-Net's stage-1 cuts is 0.587 (seed 0) and 0.376 (seed 1), against 0.269 and 0.272 for a stride chunker at our rate.

## Updated hypothesis

Our model is an LH-Net, by the blog's own chunks. Its stage-0 in-word cuts match LH-Net's 2.4x better than the new H-Net's, and its best seed matches the new H-Net's stage 1 as well as LH-Net does. The blog's two signature behaviors are absent: in-context tokenization (flat counts on the blog's own texts) and clean chunks on repetitive input (our stage 0 flips between no cuts and a cut at every byte). Both sit with the router, which the blog changed and did not publish. Constitution checklist: item 3 holds on the blog's texts; items 6 and 7 are graded not held, report-only. The seed spread at stage 1 (0.59 vs 0.38) is large; the seeds-vs-BPE run adds a third seed to read it.

## Related

- [Chunks find hidden words](../successes/2026-11-04-chunks-find-hidden-words.md)
- [Reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md)
- [Constitution 1.2.0 Phase 1 exit](../../../.agents/notes/implemented/process/2026-11-05-constitution-phase-1-exit.md)
