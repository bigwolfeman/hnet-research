# Agent Note: semantic chunking score

Status: implemented

## Problem

The H-Net authors say the learned chunks are linguistic units: words, the first letters of words, multi-word "semantic groups" (arXiv 2507.07955, Sec 3.1 and Fig 4). They show it with pictures only. Our probe `probe/align_s*` scores cuts against whitespace word edges with F1, next to a fixed-stride chunker at the same rate. That probe cannot tell word knowledge from a rule like "cut near a space", it moves with the compression ratio, and it has no fixed top anchor. We need a number with fixed anchors that every run logs, so a later eval can say how aware the chunks are, not only that a picture looks right.

## Decision

`hnet/semantics.py` scores each stage's router by AUROC. The score at a cut index is the router's boundary probability `p` there. AUROC is the chance that a true unit boundary gets a higher `p` than another candidate position. 0.5 means `p` has no information about the unit, and 1 means every unit boundary outranks every other candidate. AUROC ranks `p` and does not threshold it, so it does not depend on the compression ratio.

Five levels. Each one removes the cue of the level below it:

| level | text | candidates | unit boundaries | strata |
|---|---|---|---|---|
| word | edu val | word starts after whitespace, and the bytes inside words | the word starts | none |
| hidden | edu val, half the single spaces between letters deleted | inside non-space runs | the deleted-space joins | position in the run |
| morph | edu val | inside gold words | gold morph boundaries (SIGMORPHON 2022 English, the words whose morphs spell the word) | position in the word |
| phrase | UD English-EWT dev and test documents | word starts after whitespace | starts of multi-word dependency subtrees | none |
| gotok | Go val | inside non-space runs, outside strings and comments | Go lexical token starts | position in the run |

- At the word and phrase levels a word start scores the max `p` over its gap (the whitespace run and the first letter). A router that starts its chunks at the space (" the") and one that starts them at the letter both count. The first version scored the letter only and missed the space-start cuts that our stage 0 makes.
- Stratified levels compute AUROC within each position and weight it by positives. A router that only prefers the first letters of a word then scores 0.5, not more.
- `StageAux` carries `p` (detached, fp32) and `in_pos` (the byte position of each input position) in both model codes. Stage 1 scores only the bytes where stage 0 put a chunk. `sem/<level>_s1_seen_frac` reports that share.
- Training logs `sem/<level>_s<stage>_auroc` at every probe (`configs/data/bytes.yaml` `semantic`; null turns it off). `scripts/chunk_semantics.py` reports it per checkpoint.
- The gold data lives in the data dir (`~/data/hnet-bytes/gold`, README there with sources and licenses). It is eval only.

## Alternatives considered

- **Keep boundary F1 against a stride chunker.** Rejected as the main number. It thresholds `p`, so it moves with the ratio. A whitespace rule scores well on it, and it has no top anchor. It stays as `probe/align_s*`.
- **F1 normalized between a random chunker and the best F1 at the same rate.** This fixes the anchors, but it still thresholds `p` and still rewards the whitespace cue. AUROC per level fixes both.
- **A parser (spaCy) for phrase boundaries on our own edu text.** Rejected. A parser adds its own errors and a large dependency. Gold UD trees are human-made, at the cost of a different text domain (web text).
- **Align the canonical SIGMORPHON morphemes to the surface** ("subsidised" = subside + y + ise + ed). Rejected. The alignment is a guess. The surface-concatenative subset (79,732 of 115,126 gold words) has exact boundaries.
- **Pooled AUROC with no strata.** Rejected for the in-word levels. Cut rates depend strongly on the position in the word, so pooled AUROC would credit a position rule.

## Consequences

- The morph level covers only the gold words: about 14% of running edu words, mostly long derived words. It needs more rows than the other levels (256 in the probe).
- The phrase level runs on EWT web text, not on edu, and its constituents come from dependency subtrees. It is a proxy for phrase structure, not a constituency treebank.
- The stage-1 AUROC is conditional on stage 0: it ranks only positions that stage 0 already cut.
- A probe costs a few forward passes over about 500 rows of 2,048 bytes.
- On the reference layout at 12.58B bytes (two seeds): word 0.67-0.94, hidden 0.75-0.81, gotok 0.85-0.94, morph 0.53-0.57 at stage 0; phrase 0.50-0.67 at stage 1. The score separates seeds that the old F1 probe ranks far apart for a convention reason (seed 1 cuts after the first letter).
- First use: [chunks find hidden words](../../../../lab/experiments/successes/2026-11-04-chunks-find-hidden-words.md).
