# Experiment: recall gain by boundary

Status: refuted

Date proposed: 2026-11-03
Algorithm: per-byte coarse-path cost by in-word chunk start, `lab/spikes/2026-10-30-two-stage-hnet-smoke/scripts/boundary_gain.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does the decoder use the main network's memory mostly for bytes whose word has already started inside the current stage-1 chunk?

Context: in [main path recall budget](../successes/2026-11-02-main-path-recall-budget.md), the route through the main network opened in every seed, but the best seed took 4% of the memory in 24,000 steps. One reading is the chunk interface. A byte at position t reads the main output of the stage-1 chunk that holds t, and that output has seen bytes only up to the chunk's first byte b. If b falls inside the current word (b ≥ the word's first letter), the main network has seen a prefix of the word and can find the matching word in its document memory by content. If b falls before the word, the main network knows nothing of the word, and the decoder must match its own few bytes of the word against a bag of the document's words held in one 512-wide vector.

## Hypothesis

The second case is the hard one. Attention in the main network does a content match; a learned lookup in the decoder is slow to train. So the memory that the decoders take through the main network sits mostly in bytes with an in-word chunk start.

## Predictions

Main prediction (only this one answers the Question):

- Let D = (mean per-byte stage-1 shuffle cost over target bytes with an in-word chunk start) − (the same over target bytes without one), both at in-word positions 1 to 3. D is larger in every recall seed (0, 1, 2) than in every no-recall seed (0, 1, 2) of the 24,000-step budget checkpoints. All three recall seeds above the no-recall maximum: supported. Any at or below: not supported.

Other predictions (at most two):

- In every recall seed, D is above zero.
- In the recall seeds, the oracle memory value per byte (reach-24 oracle bits minus full-oracle bits) is about the same with and without an in-word chunk start (inside the spread over the three seeds' byte samples). The memory is worth as much in both cases, so a gap in D comes from the interface, not from where memory pays.

If supported, we do: the interface is a bottleneck. Test a read-side change that lets a byte look up the main network's memory by content, for example the decoder attending over past main outputs (an LSA-style sparse read, arXiv 2610.42147), with the main network still the only path across a document.

If not supported, we do: the decoders read the main memory as well (or as badly) whatever the chunk start, so the one-vector interface is not what holds the route back. Weigh it with [main gradient coherence](2026-11-03-main-gradient-coherence.md), which tests the optimization reading.

Gates (the run is broken if one fails; gates are not predictions):

- The script's mean shuffle cost over all target bytes reproduces the checkpoint's logged `probe/coarse_shuffle_s1` at step 24,000 within 0.05 bpb (different batches; same probe).
- Each (seed, in-word position, chunk-start class) cell holds at least 2,000 target bytes.
- The checkpoints are the six budget `final.pt` files (`ignored/runs/main-path-recall-budget/{recall,norecall}/reflayout_2stage_s{0,1,2}`), 38,552,832 params each.

## Method

```
data: synth/{recall,norecall}/val.bin, 64 rows x 4,096 bytes per checkpoint (fixed offsets, seed 1234)
per checkpoint:
    normal pass:   per-byte CE c_t (bits) for target byte t+1, stage-1 anchors b (byte positions) per row
    shuffle pass:  stage 1 ablate = shuffle (main output rolled across rows), per-byte CE c'_t
    cost_t = c'_t - c_t
    words: maximal runs of letters a-z; for target byte t+1 inside word [s, e): k = t+1 - s (in-word position)
    in-word chunk start at t: the last stage-1 anchor b <= t satisfies b >= s
    cell (k, class) = mean cost_t over target bytes with that in-word position and class, k in 1..3
    D = mean over k = 1..3 of [cell(k, in-word) - cell(k, before-word)]   (equal weight per k)
oracle value (recall data): oracle_bits(reach=24) - oracle_bits(full) per byte, same cells
```

- GPU: forward passes only, a few minutes per checkpoint. Runs at nice 19 while [main gradient coherence](2026-11-03-main-gradient-coherence.md) trains (one training model on the GPU; this adds inference only).
- Artifacts: `results/2026-11-03-recall-gain-by-boundary/` (cells per checkpoint as CSV, the script output).

## Results

Ran 2026-11-03 09:40 to 09:50 on the six budget checkpoints (commit 4b4200b plus the lexicon helper below), 8 fixed val batches x 16 rows x 4,096 bytes each, at nice 19 while main gradient coherence trained. Artifacts: [results/](../results/2026-11-03-recall-gain-by-boundary/) (`cells.csv`, `boundary_gain.out`, from `scripts/boundary_gain.py`).

### Gates

| gate | result |
|---|---|
| mean shuffle cost reproduces `probe/coarse_shuffle_s1` at 24,000 (±0.05) | pass: recall 1.7903 / 0.3820 / 0.3736 against 1.7924 / 0.3816 / 0.3725; no-recall 0.1709 / 0.2137 / 0.1691 against 0.1716 / 0.2136 / 0.1692 |
| at least 2,000 target bytes per cell | **fail**: the in-word class holds 62 to 1,842 bytes per cell (of 67,000 to 81,000) |
| the six budget checkpoints, 38,552,832 params | pass |

The stage-1 router almost never starts a chunk inside a word. At in-word positions 1 and 2, 0.1% to 0.5% of target bytes have an in-word chunk start; at position 3, 0.3% to 2.7%. The in-word class is too small to grade, so the main prediction is not graded.

### Where the main network is used, by in-word position (descriptive)

Shuffle cost in bits per target byte (k = 0 is a word's first letter, −1 a space or separator target), all chunk starts pooled, about 80k bytes per row (13k at k = 7):

| k | oracle memory value | recall s0 | recall s1 | recall s2 | no-recall s0 | s1 | s2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −1 | 0.041 | 1.266 | 0.131 | 0.110 | 0.035 | 0.032 | 0.042 |
| 0 | 0.441 | 0.130 | 0.029 | 0.020 | 0.000 | 0.000 | 0.000 |
| 1 | **2.544** | 0.135 | 0.029 | 0.023 | 0.000 | 0.000 | 0.000 |
| 2 | **2.243** | 0.248 | 0.063 | 0.056 | 0.001 | 0.000 | 0.000 |
| 3 | 0.295 | 0.039 | 0.043 | 0.022 | 0.064 | 0.071 | 0.067 |
| 4 | 0.008 | **4.911** | 1.025 | 1.197 | 0.552 | 0.641 | 0.491 |
| 5 | 0.001 | **6.742** | 1.339 | 1.245 | 0.564 | 0.735 | 0.553 |
| 6 | 0.000 | 6.312 | 1.504 | 1.574 | 0.690 | 0.921 | 0.689 |
| 7 | 0.000 | 5.957 | 1.936 | 1.417 | 0.850 | 1.181 | 1.012 |

Oracle memory value = reach-24 oracle bits − full oracle bits per byte (the same batches in every recall arm).

- Main prediction: not graded (cell-size gate failed).
- Other prediction 1 (D above zero in every recall seed): −0.135, +0.071, +0.217. Computed on the small in-word cells, so it carries no weight.
- Other prediction 2 (oracle value the same in both classes): not testable on cells this small.

### What the runs show (observations after the fact)

- **The memory pays at the start of a word, and the model uses the main network at its end.** The document memory is worth 0.44, 2.54, and 2.24 bits at a word's first three letters and about nothing from the fifth letter on (a 4-letter prefix nearly fixes the word in the lexicon). The decoders depend on the main output the other way round: at letters 0 to 2, shuffling costs 0.02 to 0.25 bits of the 0.44 to 2.54 available; at letters 4 to 7 it costs 1.0 to 6.7 bits in the recall arms and 0.5 to 1.2 in the no-recall arms.
- **The main network is read once it has seen part of the word.** Stage-1 chunks start at or before the space, so at letters 0 to 2 the main output holds nothing of the current word. A 9-byte chunk often ends inside a long word, and the next chunk then starts inside it: from letter 4 on, the main network has seen a prefix. That is where the decoders read it, for a job (complete the word) that the lexicon already does.
- **So the interface reading holds in a stronger form than planned.** Every byte where memory pays is in the hard case: the decoder must pick the next word from a bag in one vector, with no prefix seen by the main network. The decoders barely learned that lookup in 24,000 steps (best seed: 0.25 of 2.24 bits at letter 2).
- The no-recall arms show the same late-word use (0.5 to 1.2 bits at letters 4 to 7). That use is not memory: it is the word completion that any main output with a seen prefix can give.

Code: `hnet/synth.py` now has `read_lexicon` and `write_lexicon`, used by `make_synth.py`, `reach_anchors.py`, and `boundary_gain.py` (one parser, a word-count check, `tests/test_synth.py::test_lexicon_round_trip`).

## Verdict

**Can't tell.** The trained stage-1 routers start a chunk inside a word for less than 3% of the bytes at in-word positions 1 to 3. The class this test compares against is nearly empty, so the main prediction cannot be graded. The descriptive split by in-word position is strong, but it was not the planned prediction.

## Updated hypothesis

The route through the main network is slow because the bytes where the memory pays never have a main output that has seen the current word. To use the memory there, a decoder must turn one vector (a summary of the document so far) and its own 1 to 3 letters into the next word. Attention does this match by content; the chunk interface leaves it to a learned lookup in a small decoder. The main network is read by content only where it has seen a prefix, and there the lexicon makes memory worthless.

The bigger test that would answer the Question, and the next planned work on the read side: give the stage-1 decoder a content read of the main network's past outputs (causal attention from bytes over past main positions; the main network stays the only path across a document). Prediction to write before it runs: with that read, the shuffle cost at letters 1 and 2 rises toward the oracle value in every recall seed, against the plain layout. It waits for [main gradient coherence](2026-11-03-main-gradient-coherence.md), which tests the other reading.

## Related

- [Main path recall budget](../successes/2026-11-02-main-path-recall-budget.md)
- [Main gradient coherence](2026-11-03-main-gradient-coherence.md)
- [Main network sole path](2026-11-02-main-network-sole-path.md)
