# Experiment: decoder reads main by content

Status: refuted

Date proposed: 2026-11-03
Algorithm: MainRead, `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/mainread.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

If the stage-1 decoder can read the main network's past outputs by content, does the model take the document memory through the main network at the letters where it pays?

Context: in [recall gain by boundary](2026-11-03-recall-gain-by-boundary.md), the document memory is worth 0.44, 2.54, and 2.24 bits at a word's first three letters, and about nothing later. The decoders read the main network at letters 4 to 7 (1.0 to 6.7 bits of shuffle cost) and barely at letters 0 to 2 (0.02 to 0.25 bits). Stage-1 chunks start at or before the word, so at letters 0 to 2 the main output a position reads (its own chunk's) holds nothing of the current word. The decoder must pick the next word from a summary in one vector.

## Hypothesis

The bottleneck is the read, not the main network's content. A main output per past chunk already holds that chunk's words (the main network sees every chunk). With an attention read, a stage-1 position whose encoder state has seen the word's first letters can match them against past main outputs and copy the word. That is a content match, which attention learns fast, in place of a lookup in one vector, which the decoders learned slowly.

## Predictions

Main prediction (only this one answers the Question):

- At step 6,000, the mean stage-1 shuffle cost over target bytes at in-word letters 1 and 2 (`scripts/boundary_gain.py`, all chunk starts pooled) is higher in every read seed (0, 1, 2) than in every plain seed (0, 1, 2). All three read seeds above the plain maximum: supported. Any at or below: not supported.

Other predictions (at most two):

- Val bpb at step 6,000 is lower in every read seed than in every plain seed.
- Val bpb at step 6,000 is below the reach-48 oracle (1.8119) in every read seed: the model holds memory that no outer path can reach.

If supported, we do: the chunk interface is a real bottleneck of the design, and a content read fixes it on synthetic recall. Next, put MainRead on the reference layout on text: the dead seeds at 786M bytes against their controls, then the BPE ladder. Write an Agent Note on the design change.

If not supported, we do: a content read does not unlock the memory either, so the read interface is not what holds the route back. Weigh it with [main gradient coherence](2026-11-03-main-gradient-coherence.md) (the optimization reading) and look at what the main network's outputs hold (a linear probe for the current document's word set).

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including `tests/test_mainread.py` (zero at init; a position reads only chunks it has seen; the model with MainRead is causal for all routers; the probe's zero and shuffle act on the read; FLOPs and params counted).
- Read arms: 38,815,488 params (the plain sole-path layout's 38,552,832 plus 262,656 for MainRead at d = 256); layouts t4, t4, T1t4, t4T1, windows 4 and 1, no Mamba-2.
- Ratios within 15% of 3 x 3 at the end; finite loss; 393,216,000 training bytes per arm.
- Oracle lower bound: val bpb at or above 1.021.
- `boundary_gain.py` reproduces each checkpoint's logged `probe/coarse_shuffle_s1` at step 6,000 within 0.05 bpb.

## Method

```
MainRead at stage 1 only (model.stages.1.main_read.head_dim = 64: 4 heads at d = 256):
    q from the stage-1 encoder output at the position; k, v from the main outputs (the width-256 slice the EMA reads)
    a position reads chunks j <= its read index (the chunks its EMA read has seen); zero-init output projection
    decoder input = EMA read + residual + MainRead output
why not at stage 0: stage-1 outputs carry their position's local content through the residual, so a stage-0 read
    over all past stage-1 outputs would be a global path over the document that skips the main network.
    At stage 1 the keys are main outputs only, one per stage-1 chunk.
data: synth/recall (as main network sole path)
model: sole-path layout (stage 0 t4/t4 window 4; stage 1 T1t4/t4T1 window 1; main T10 global) + MainRead at stage 1
train: configs/train/ladder.yaml, 6,000 steps, no rungs, eval every 1,000
arms: read, seeds 0, 1, 2
control: plain, seeds 0, 1, 2 = the synthetic arms of main gradient coherence (same code, same week, same config
         minus MainRead; group main-gradient-coherence, log dir ignored/runs/main-gradient-coherence/synth)
grade: boundary_gain.py on the six final.pt files; cells k = 1, 2, class "all"; mean of the two cells (equal weight)
```

- What the read changes about "the main network is the only path across a document": the read adds one attention layer over main outputs. A main output holds its own chunk's input in the main network's residual stream, so the read could copy a past chunk's local content even if the main network's own layers did nothing. The path across the document still runs only through chunk vectors at the main network's resolution, never through bytes or stage-0 chunks. This experiment asks whether the interface lets the memory through, not whether the main network's layers compute it.
- Launch after main gradient coherence (one training model on the GPU), from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, queue `ignored/runs/decoder-reads-main-by-content/chain.sh`: suite gate, then
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/decoder-reads-main-by-content/read decoder-reads-main-by-content reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.0.enc_layout=t4 model.stages.0.dec_layout=t4 model.stages.0.window=4 model.stages.1.enc_layout=T1t4 model.stages.1.dec_layout=t4T1 model.stages.1.window=1 +model.stages.1.main_read.head_dim=64 train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000`
  - then `scripts/boundary_gain.py` on the read and plain checkpoints.
- Amended 2026-11-03, before any arm ran: the first suite gate failed on `test_mainread.py::test_compiled_read_matches_eager_on_gpu` (max logit diff 6.7). The test was wrong, not the read: it used random weights under bf16, where the routers sit near p = 0.5 and bf16 noise flips boundaries (the plain model fails the same check, max 5.7, with different chunk counts). `tests/test_gpu.py` already isolates this case. The fix runs its two compile-parity tests (default init in bf16, random weights in fp32 with routing equal and logits within 1e-3) with and without MainRead, and `test_mainread.py` keeps a compiled-gradient check. MainRead alone matches eager to 2e-8. The chain then reran from the suite gate.
- wandb project `hnet-router`, group `decoder-reads-main-by-content`.
- Cost: about 7 to 8 min per arm at the 500 W cap, about 25 min in total.

## Results

The three read arms ran to 6,000 steps (2026-11-03 13:00 to 13:19, commit 2cb7d72, after the gate fix in Method). wandb group `decoder-reads-main-by-content`: seeds 0, 1, 2 = `4yv5uikn`, `hepf0tml`, `3ibzji07`. Plain control: the synthetic arms of [main gradient coherence](2026-11-03-main-gradient-coherence.md) (`6779e4vx`, `wrv1ej2g`, `w5rvq0rb`). Artifacts: [results/](../results/2026-11-03-decoder-reads-main-by-content/) (`boundary-gain-cells.csv`, `boundary-gain.out`, `query_prefix.py`, `query-prefix.out`, `chain.sh`).

`boundary_gain.py` named arms by data variant and seed, so the read and plain arms had the same names in the first output. It now names each arm by its run directory; the cells were computed again from the same checkpoints.

### Gates

| gate | result |
|---|---|
| suite (`HNET_TEST_CUDA=1`), with the parity fix | pass: 182 passed, 1 skipped |
| read arms 38,815,488 params | pass |
| ratios within 15% of 3 x 3; finite loss; 393,216,000 bytes | pass: 2.98 to 3.04 and 2.96 to 3.05 |
| val bpb at or above 1.021 | pass: 1.994 to 2.000 |
| `boundary_gain.py` mean shuffle cost reproduces `probe/coarse_shuffle_s1` (±0.05) | pass: read 2.5534 / 0.3597 / 0.3965 against 2.5555 / 0.3595 / 0.3967; plain 2.7664 / 0.2176 / 0.2719 against 2.7683 / 0.2177 / 0.2714 |

### Shuffle cost by in-word letter (all chunk starts pooled, bits per target byte)

| k | oracle memory value | read s0 | read s1 | read s2 | plain s0 | plain s1 | plain s2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −1 | 0.041 | 1.458 | 0.134 | 0.126 | 1.844 | 0.053 | 0.073 |
| 0 | 0.441 | 0.040 | 0.010 | 0.004 | 0.050 | 0.000 | 0.000 |
| 1 | 2.544 | 0.051 | 0.011 | 0.006 | 0.049 | 0.001 | 0.001 |
| 2 | 2.243 | 0.136 | 0.071 | 0.131 | 0.133 | 0.002 | 0.001 |
| 3 | 0.295 | 0.021 | 0.042 | 0.065 | 0.024 | 0.019 | 0.014 |
| 4 | 0.008 | 8.550 | 1.244 | 1.368 | 8.548 | 0.824 | 0.905 |
| 5 | 0.001 | 9.060 | 1.150 | 1.128 | 10.118 | 0.764 | 0.993 |
| 7 | 0.000 | 9.878 | 1.519 | 1.778 | 9.911 | 0.833 | 1.134 |

- Main prediction (mean of k = 1, 2 above the plain maximum in every read seed): read 0.0936, 0.0411, 0.0685 against plain 0.0907, 0.0013, 0.0008. Read seeds 1 and 2 are below plain seed 0. Does not hold.
- Other prediction 1 (val bpb lower than every plain seed): read 1.9950, 2.0003, 1.9940 against plain 1.9936, 2.0118, 2.0129. Does not hold (read seed 0 is above plain seed 0).
- Other prediction 2 (val bpb below reach-48, 1.8119): does not hold in any seed.

### Does the query see the word? (`query_prefix.py`, descriptive)

A byte reads one stage-1 output: the one for its stage-0 chunk. That stage-1 position holds the stage-0 encoder output at the chunk's first byte. So the MainRead query knows the bytes up to the start of the byte's stage-0 chunk, not the byte itself. Fraction of targets whose stage-1 position has seen at least one letter of the current word (4 val batches):

| arm | k = 0 | k = 1 | k = 2 | k = 3 | read rms / decoder-input rms |
|---|---:|---:|---:|---:|---|
| read s0 | 0.000 | 0.001 | 0.012 | 0.222 | 0.287 / 1.916 |
| read s1 | 0.000 | 0.001 | 0.100 | 0.349 | 0.658 / 2.189 |
| read s2 | 0.000 | 0.001 | 0.173 | 0.392 | 0.731 / 2.179 |
| plain s0 | 0.000 | 0.001 | 0.003 | 0.067 | |
| plain s1 | 0.000 | 0.001 | 0.017 | 0.414 | |
| plain s2 | 0.000 | 0.001 | 0.107 | 0.454 | |

### What the runs show (observations after the fact)

- **The query was blind where the memory pays.** Stage-0 chunks start at or before the space, like stage-1 chunks. At letter 1 the stage-1 query has seen a letter of the word for 0.1% of targets, and at letter 2 for 1% to 17%. With no prefix, a content read has nothing to match against: it is the same one-vector lookup as the EMA read, moved one level down. The Hypothesis assumed "a stage-1 position whose encoder state has seen the word's first letters". That assumption was false for this layout.
- **The read learned and is used.** Its output rms is 0.29 to 0.73 against a decoder input of 1.9 to 2.2, and its output weight grew from 0 to rms 0.03 to 0.05. In read seeds 1 and 2 the k = 2 cost rose from about 0.002 (plain seeds 1, 2) to 0.07 and 0.13, where 10% and 17% of k = 2 queries see a letter. The gain follows the prefix the query can see.
- Seed 0 builds the late-word route in both arms (8.5 to 10 bits at k = 4 to 7), as in earlier runs.

## Verdict

**Not supported.** At step 6,000, read seeds 1 and 2 have a mean stage-1 shuffle cost at in-word letters 1 and 2 of 0.041 and 0.069, below plain seed 0 (0.091). The stage-1 content read did not let the memory through at the letters where it pays.

## Updated hypothesis

The test was valid, but the design did not reach its target: the query had no prefix. The read-side reading stands. The memory needs a query that has seen the current word's first letters, and in this H-Net only the byte level has them: both routers start chunks at the space, so every coarse position a word-start byte reads was formed before the word began.

That moves the read to the bytes. The stage-0 decoder input at byte t queries the main network's outputs directly (keys = main outputs, causal over the main chunks byte t's EMA path has seen). The main network stays the only path across a document; stage-1 outputs, which carry local content, are never keys. The plan's step for this case (a linear probe of what the main outputs hold) waits: it asks whether main holds the word set, and this run never put a query in front of it that could use the answer. Next planned: [bytes read main by content](../successes/2026-11-03-bytes-read-main-by-content.md).

## Related

- [Recall gain by boundary](2026-11-03-recall-gain-by-boundary.md)
- [Main path recall budget](../successes/2026-11-02-main-path-recall-budget.md)
- [Main gradient coherence](2026-11-03-main-gradient-coherence.md)
- Latent Sparse Attention (arXiv 2610.42147): content read over clustered keys; the idea behind a content read of coarse outputs.
