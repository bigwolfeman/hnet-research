# Experiment: bytes read main by content

Status: success

Date proposed: 2026-11-03
Algorithm: MainRead at stage 0, `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/mainread.py` and `hnet/stage.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

If every byte can read the main network's past outputs by content, does the model take the document memory at the in-word letters where it pays?

Context: in [decoder reads main by content](../refuted/2026-11-03-decoder-reads-main-by-content.md), a content read at stage 1 did not let the memory through. Its query came from a stage-1 position, and both routers start chunks at or before the space, so at in-word letter 1 that query had seen a letter of the word for 0.1% of targets (letter 2: 1% to 17%). Where the query did see a letter, the read's gain followed. The memory is worth 2.54 and 2.24 bits at letters 1 and 2 ([recall gain by boundary](../refuted/2026-11-03-recall-gain-by-boundary.md)).

## Hypothesis

The memory needs a query that has seen the current word's first letters, and only the bytes have them. A byte at letter k has seen k letters of the word in its own encoder state. Let that byte match them against the main network's past outputs, and attention can copy the word, as an induction head does. The Byte Latent Transformer's local decoder has the same shape: byte queries, cross-attention over the global model's patch outputs.

## Predictions

Main prediction (only this one answers the Question):

- At step 6,000, the mean bits per target byte at in-word letters 1 and 2 (`scripts/boundary_gain.py`, column `bits`, class "all", the mean of the two cells with equal weight) is lower in every read seed (0, 1, 2) than in every plain seed (0, 1, 2). All three read seeds below the plain minimum: supported. Any at or above: not supported.

Other predictions (at most two):

- In every read seed, the read's own shuffle cost (`boundary_gain.py --stage 0 --part read`, class "all") is higher at letters 1 and 2 (mean of the two cells) than at letters 4 to 7 (mean of the four cells). The read carries the memory where the memory pays, not word completion.
- Val bpb at step 6,000 is below the reach-48 oracle (1.8119) in every read seed.

If supported, we do: the chunk interface is the bottleneck, and the fix is a byte-level content read of the main network. Next, put it on the reference layout on text: the dead seeds at 786M bytes against their controls, then the BPE ladder. Write an Agent Note on the design change.

If not supported, we do: with a query that sees the prefix and a content read over every past main output, the memory still does not come through in 6,000 steps. Then the question is what the main outputs hold: a linear probe for the current document's word set on the main outputs, in plain and read checkpoints.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including `tests/test_mainread.py` (stage-0 keys are the main outputs; the composite index; the read-only probe; the model with a read at every stage is causal for all routers) and the compile-parity tests in `tests/test_gpu.py` with the read at every stage.
- Read arms: 38,815,488 params (the plain sole-path layout's 38,552,832 plus 262,656 for MainRead at d = 256); layouts t4, t4, T1t4, t4T1, windows 4 and 1, no Mamba-2.
- Ratios within 15% of 3 x 3 at the end; finite loss; 393,216,000 training bytes per arm.
- Oracle lower bound: val bpb at or above 1.021.
- `boundary_gain.py` reproduces each read checkpoint's logged `probe/read_shuffle_s0` (stage 0, part read) and `probe/coarse_shuffle_s1` (stage 1, part all) at step 6,000 within 0.05 bpb, and each plain checkpoint's `probe/coarse_shuffle_s1`.

## Method

```
MainRead at stage 0 only (model.stages.0.main_read.head_dim = 64: 4 heads at d = 256):
    q from the stage-0 encoder output at byte t (window 4, 4 layers: it has seen the word's letters up to t)
    k, v from the MAIN outputs (the first 256 dims, the slice stage 1's EMA reads); never from stage-1 outputs
    byte t reads main chunk j iff j <= main_idx[t] = stage1.read_idx[stage0.read_idx[t]]
        (the main chunks t's EMA path already depends on; same causality)
    zero-init output projection; decoder input = EMA read + residual + MainRead output
data: synth/recall (as main network sole path)
model: sole-path layout (stage 0 t4/t4 window 4; stage 1 T1t4/t4T1 window 1; main T10 global) + MainRead at stage 0
train: configs/train/ladder.yaml, 6,000 steps, no rungs, eval every 1,000
arms: read, seeds 0, 1, 2
control: plain, seeds 0, 1, 2 = the synthetic arms of main gradient coherence (same code path for the plain model,
         same config minus MainRead; ignored/runs/main-gradient-coherence/synth)
grade: boundary_gain.py on the six final.pt files (8 fixed val batches); cells k = 1, 2, class "all", column bits
```

- The main prediction grades bits, not a probe. A gain at letters 1 and 2 can only come from the past of the document: the plain arms have the same local path. The read-only shuffle (other prediction 1) says whether the gain is document-specific content.
- What the read changes about "the main network is the only path across a document": as at stage 1, a main output holds its own chunk's input in the main network's residual stream. So the read could copy a past chunk's local content even if the main network's own layers did nothing. The path across the document still runs only through main outputs, one per stage-1 chunk. This asks whether the interface lets the memory through, not whether the main network's layers compute it.
- Code change since the stage-1 run: MainRead keys are always main outputs, masked by the composite index (`Stage.forward`, `compose_read_index`); at stage 1 that is the same read as before. `probes.ablated(..., part="read")` ablates the read alone and the trainer logs `probe/read_{zero,shuffle}_s{s}`. `boundary_gain.py` takes `--stage` and `--part` and writes a `bits` column. FLOPs count the keys at main resolution.
- Smoke (500 steps, wandb `hnet-smoke` / `k92glquj`): 38,815,488 params, 56 ms per step, 13.0 GB peak, the read probes logged.
- Launch after the suite, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, queue `ignored/runs/bytes-read-main-by-content/chain.sh`:
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/bytes-read-main-by-content/read bytes-read-main-by-content reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.0.enc_layout=t4 model.stages.0.dec_layout=t4 model.stages.0.window=4 model.stages.1.enc_layout=T1t4 model.stages.1.dec_layout=t4T1 model.stages.1.window=1 +model.stages.0.main_read.head_dim=64 train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000`
  - then `scripts/boundary_gain.py --stage 1 --part all` on the six checkpoints, and `--stage 0 --part read` on the read checkpoints.
- wandb project `hnet-router`, group `bytes-read-main-by-content`.
- Cost: about 6.5 min per arm, about 22 min in total.

## Results

The three read arms ran to 6,000 steps (2026-11-03 13:32 to 13:51, commit 59c240e). wandb group `bytes-read-main-by-content`: seeds 0, 1, 2 = `asjnbjcj`, `5wt4piv1`, `k89lblaa`. Plain control: `6779e4vx`, `wrv1ej2g`, `w5rvq0rb` (the synthetic arms of [main gradient coherence](../refuted/2026-11-03-main-gradient-coherence.md)). Artifacts: [results/](../results/2026-11-03-bytes-read-main-by-content/) (`bg-s1-all.csv` / `.out`: six checkpoints, stage 1 part all, with the `bits` column; `bg-s0-read.csv` / `.out`: read checkpoints, stage 0 part read; `chain.sh`).

### Gates

| gate | result |
|---|---|
| suite (`HNET_TEST_CUDA=1`) | pass: 186 passed, 1 skipped |
| read arms 38,815,488 params | pass |
| ratios within 15% of 3 x 3; finite loss; 393,216,000 bytes | pass: 2.97 to 3.02 and 2.99 to 3.01 |
| val bpb at or above 1.021 | pass: 1.362 to 1.742 |
| `boundary_gain.py` reproduces the logged probes (±0.05) | pass: `read_shuffle_s0` 0.8799 / 0.4328 / 0.9453 against 0.8800 / 0.4328 / 0.9453; `coarse_shuffle_s1` (read) 3.1229 / 0.0634 / 0.0614 against 3.1220 / 0.0634 / 0.0613; plain 2.7664 / 0.2176 / 0.2719 against 2.7683 / 0.2177 / 0.2714 |

Fwd FLOPs/byte: read 15.56M to 15.69M, plain 15.13M to 15.19M (+3%).

### Bits per target byte by in-word letter (class "all", 8 fixed val batches)

| k | oracle memory value | read s0 | read s1 | read s2 | plain s0 | plain s1 | plain s2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −1 | 0.041 | 0.085 | 0.105 | 0.083 | 0.122 | 0.118 | 0.121 |
| 0 | 0.441 | 4.427 | 4.601 | 4.431 | 4.670 | 4.698 | 4.698 |
| 1 | 2.544 | **2.871** | **3.956** | **2.901** | 4.620 | 4.647 | 4.649 |
| 2 | 2.243 | **1.332** | **2.413** | **1.383** | 3.213 | 3.274 | 3.277 |
| 3 | 0.295 | 0.150 | 0.293 | 0.146 | 0.431 | 0.440 | 0.440 |
| 4 | 0.008 | 0.016 | 0.012 | 0.015 | 0.016 | 0.015 | 0.015 |
| 5 | 0.001 | 0.004 | 0.004 | 0.003 | 0.005 | 0.004 | 0.004 |

- Main prediction (mean bits at k = 1, 2 below the plain minimum in every read seed): read 2.102, 3.184, 2.142 against plain 3.916, 3.961, 3.963. Holds.
- Other prediction 1 (read-only shuffle cost higher at k = 1, 2 than at k = 4 to 7, every read seed): 2.227 against 0.056, 1.126 against 0.021, 2.451 against 0.022. Holds. Per k (seed 0): k = 0 0.49, k = 1 1.91, k = 2 2.55, k = 3 0.72, k = 4 to 7 0.03 to 0.09.
- Other prediction 2 (val bpb below reach-48, 1.8119): 1.362, 1.742, 1.374. Holds.

Val bpb through training:

| step | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 6,000 |
|---|---:|---:|---:|---:|---:|---:|
| read s0 | 1.950 | 1.597 | 1.516 | 1.455 | 1.455 | 1.362 |
| read s1 | 2.466 | 1.971 | 1.856 | 1.831 | 1.808 | 1.742 |
| read s2 | 2.082 | 1.625 | 1.514 | 1.484 | 1.468 | 1.374 |

Anchors: no memory 2.006, reach-48 1.8119, full oracle 1.021. Plain arms end at 1.994 to 2.013.

### What the runs show (observations after the fact)

- **The memory comes through fast.** Seeds 0 and 2 take 65% and 64% of the memory (no-memory anchor minus val bpb, over no-memory minus oracle) and seed 1 takes 27%, in 6,000 steps. The plain layout took 4% in 24,000 steps ([main path recall budget](2026-11-02-main-path-recall-budget.md)). Seeds 0 and 2 pass reach-48 by step 2,000.
- **It comes through where the bytes have the prefix.** The read's gain sits at letters 1 to 3 (k = 2: 1.33 against 3.21 bits, of 2.24 available). At letter 0, with no letter of the word seen, the read gives 0.24 of 0.44 bits.
- **The read replaces the stage-1 route.** In read seeds 1 and 2 the stage-1 shuffle cost is 0.06, under the plain seeds' 0.22 and 0.27. Seed 0 keeps a large stage-1 dependence (3.12) next to the read.
- Seed 1 is slower in all metrics from step 1,000 on; the lottery is smaller but still there.

## Verdict

**Supported.** At step 6,000, the mean bits per target byte at in-word letters 1 and 2 is 2.10, 3.18, and 2.14 in the three read seeds, below every plain seed (3.92 to 3.96). With a byte-level content read of the main outputs, the model takes the document memory at the letters where it pays.

## Updated hypothesis

The H-Net's chunk interface is the bottleneck for using what the main network holds. Both routers cut at the space, so every coarse vector a byte reads through the EMA dechunk was formed before the current word began, and a decoder cannot turn one such vector into the next word. A byte-level query over the main outputs does this match in a few thousand steps. The main network was not the limit: its outputs held the document's words well enough for up to 65% of the memory, under the caveat in Method (the read could take local content from main's residual stream).

Next, as planned for this case: the byte read on text. The reference layout in our code, seeds 0 to 4 at 786M bytes, against the five reruns of main gradient coherence and the BPE control (edu 1.273 / 1.271 at matched bytes and FLOPs). Then the BPE ladder.

## Related

- [Decoder reads main by content](../refuted/2026-11-03-decoder-reads-main-by-content.md)
- [Recall gain by boundary](../refuted/2026-11-03-recall-gain-by-boundary.md)
- [Main gradient coherence](../refuted/2026-11-03-main-gradient-coherence.md)
- [Main path recall budget](2026-11-02-main-path-recall-budget.md)
- Byte Latent Transformer (Pagnoni et al., 2024): a local byte decoder that cross-attends to the global model's patch outputs.
