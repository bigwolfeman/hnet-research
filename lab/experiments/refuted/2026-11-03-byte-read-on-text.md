# Experiment: byte read on text

Status: refuted

Date proposed: 2026-11-03
Algorithm: MainRead at stage 0, `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/mainread.py` and `hnet/stage.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

On text, with the reference 2-stage layout in our code at 786M training bytes, does a byte-level content read of the main network's outputs lower edu validation bpb against the same layout without it?

Context: on synthetic recall, the byte read took 27% to 65% of the document memory in 6,000 steps, where the plain layout took 4% in 24,000 ([bytes read main by content](../successes/2026-11-03-bytes-read-main-by-content.md)). On text, the reference layout's stage-1 fate is a lottery: in the reruns of [main gradient coherence](./2026-11-03-main-gradient-coherence.md), one of five seeds ended live, and edu bpb ran from 1.313 to 1.366. The BPE Transformer at matched bytes and FLOPs has 1.273 and 1.271 ([byte and BPE controls](./2026-10-31-byte-and-bpe-controls.md)).

## Hypothesis

On text the H-Net loses to BPE in part because the bytes cannot use what the main network computes: every coarse vector a byte reads was formed before the current word began. A byte read of the main outputs gives each byte a query with its own prefix. If the main network holds useful document context on text, the bytes take it, in every seed, and the stage-1 lottery matters less.

## Predictions

Main prediction (only this one answers the Question):

- At step 12,000, edu val bpb is lower in every read seed (0 to 4) than in every plain seed (the five reruns of main gradient coherence, 1.3133 to 1.3661). All five read seeds below 1.3133: supported. Any at or above: not supported.

Other predictions (at most two):

- Edu val bpb is below both BPE seeds (1.271 and 1.273) in every read seed.
- In every read seed, `probe/read_shuffle_s0` is above `probe/read_zero_s0` at step 12,000: the read carries document-specific content, not a constant input.

If supported, we do: the byte read is a real gain on text at this scale. Run the BPE ladder with it (0.79B, 3.15B, 12.58B bytes, as in [compute ladder vs BPE](./2026-11-01-compute-ladder-vs-bpe.md)), and bring the design change to the user for review before any promotion out of the spike.

If not supported, we do: on text the bytes get no gain from reading the main outputs at this scale. Measure where the read is used (per-byte shuffle cost of the read by word position on edu val) to tell "main holds nothing useful at 786M bytes" from "the read is not learned on text".

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`).
- Read arms: 42,064,000 params (the plain layout's 41,801,344 plus 262,656).
- Edu ratios within 15% of 3 x 3 at the end; 786,432,000 bytes per arm; finite loss.
- Edu fwd FLOPs/byte within 5% of 22.27M (the BPE comparison is at matched FLOPs). The read adds about 0.52M per byte (2.4%).
- `probe/read_*_s0` logged at every eval.

## Method

```
model: reflayout_2stage (stage 0 m4/m4 window 1024; stage 1 T1m4/m4T1 window 1024; main T10 global; 256 | 256 | 512)
       + MainRead at stage 0 (model.stages.0.main_read.head_dim = 64):
         q from the stage-0 encoder output at byte t; k, v from the main outputs (first 256 dims);
         byte t reads main chunk j iff j <= stage1.read_idx[stage0.read_idx[t]]; zero-init output
data: edu 0.9 / Go 0.1 by bytes (configs/data/bytes.yaml), as the seed-rate runs
train: configs/train/ladder.yaml, 12,000 steps, no rungs
arms: read, seeds 0 to 4
control: plain, seeds 0 to 4 = the edu arms of main gradient coherence (ours; same code for the plain path, same
         config minus MainRead; wandb akqhdkkz, ipcxldpl, zplsyk1o, 5d3o336d, 7dffta9c)
grade: edu val bpb at step 12,000 (final eval)
```

- The read on text has an outer path next to it: stage 0 is Mamba-2 with a long state, and stage 1 attends over a 1,024-chunk window. A gain says the bytes use the main outputs better through the read. It does not say the main network is the only route.
- Smoke (500 steps, wandb `hnet-smoke`, group `text-read-smoke`): 42,064,000 params, 98 ms per step, 16.2 GB peak, read probes logged.
- Launch after the suite, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, queue `ignored/runs/byte-read-on-text/chain.sh`:
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/byte-read-on-text/read byte-read-on-text reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 reflayout_2stage:3 reflayout_2stage:4 -- +model.stages.0.main_read.head_dim=64 train.max_steps=12000 "train.ladder_fracs=[]"`
- wandb project `hnet-router`, group `byte-read-on-text`.
- Cost: about 21 min per arm at the 500 W cap, about 1 h 45 min in total.

## Results

Chain `ignored/runs/byte-read-on-text/chain.sh` (copy in [results/](../results/2026-11-03-byte-read-on-text/)), code at 40fea50, 13:58 to 15:45. Artifacts: `summary.txt` and `val-curves.csv` (`pull.py`), `read-curve.txt` (`read_curve.py`), `chain.sh`.

### Gates (all hold)

- Suite: 186 passed, 1 skipped.
- Params 42,064,000 in every read arm. Bytes 786,432,000. Finite loss.
- Edu ratios 3.02-3.08 x 2.85-2.92.
- Edu fwd FLOPs/byte 23.09M to 23.33M: +3.7% to +4.8% against 22.27M, inside 5%. The plain arms spend 22.36M to 22.81M, so the read adds about 0.55M per byte (the plan said 0.52M).
- `probe/read_*_s0` logged at every eval.

### Main prediction: edu val bpb at step 12,000

| seed | plain (control) | read | read stage-1 fate | read use (shuffle − zero, s0) |
|---|---:|---:|---|---:|
| 0 | 1.3133 | 1.3176 | dead (−0.029) | +0.0000 |
| 1 | 1.3661 | **1.3047** | live (+0.007) | −0.0000 |
| 2 | 1.3177 | 1.3365 | live (+0.010) | +0.0015 |
| 3 | 1.3383 | 1.3204 | live (+0.031) | +0.0003 |
| 4 | 1.3561 | **1.3051** | live (+0.070) | +0.0001 |
| mean | 1.3383 | 1.3169 | | |

Two read seeds (1, 4) are below 1.3133. Seeds 0, 2, and 3 are at or above it.

### Other predictions

- Below both BPE seeds (1.271, 1.273) in every read seed: does not hold. The best read seed is 1.3047, 0.032 above BPE.
- `probe/read_shuffle_s0` above `probe/read_zero_s0` in every read seed: holds in sign in 4 of 5 seeds, but at 0.0000 to 0.0015 bits per byte the sign is noise. The read is not used.

### Observations (not predictions)

- **The read is inert for the whole run.** At every eval from step 2,000 to 12,000, zeroing the read costs at most 0.0017 bits per byte and shuffling it at most 0.0039 (seed 2, step 6,000). On synthetic recall, shuffling the read cost 0.43 to 0.95 bits per byte on average and 1.1 to 2.7 at in-word letters 1 and 2.
- **The read arm is a different init draw, not only a different model.** `Stage.init_weights` draws the read's weights before stage 1 and main, so read seed n starts stage 1 and main from different weights than plain seed n. With an inert read, the bpb difference between the arms (mean 1.3169 vs 1.3383) and the fates (4 of 5 live vs 1 of 5) are a second sample of the init lottery. They do not measure the read. The two arms overlap (plain 1.313-1.366, read 1.305-1.337).
- **Go bpb:** read 0.837 to 0.911, plain 0.889 to 0.976.
- The plan's not-supported step (the read's shuffle cost by word position on edu) cannot tell its two cases apart. A per-position split of a mean cost under 0.002 bits per byte shows near zero at every position whatever the cause. I did not run it.

## Verdict

**Not supported.** With a byte-level content read of the main outputs, three of five read seeds have edu val bpb at or above the best plain seed (1.3133). The read itself stays unused through training: its probe cost is under 0.004 bits per byte at every eval. On text at 786M bytes, the bytes take nothing from the main outputs, even through an interface that took 27% to 65% of the synthetic memory in 6,000 steps.

## Updated hypothesis

The read fixed the interface. On synthetic recall that was the bottleneck. On text it is not. If the interface were the cause on text, the read would carry something. It carries nothing, so the main network's outputs hold nothing that the bytes cannot already get from the outer paths. Two facts from this layout make that likely.

1. The outer paths cover most of a row. Rows are 4,096 bytes. Stage 1 attends over 1,024 chunks, about 3,000 bytes, and stage 0 is Mamba-2 with a long state. On synthetic recall the outer windows were 4 bytes and 1 chunk, so the main network was the only long path. On text, the only context that main alone can see is the far end of the row.
2. In the released `hnet_2stage_L`, shuffling the main output costs 1.35 bpb on edu ([reference checkpoint main use](../successes/2026-11-01-reference-checkpoint-main-use.md)). That is far more than any long-range memory is worth. A working main network does the core language modeling for the outer stages: it predicts the next chunk from local context too. Ours does not, at any scale we ran: the blog split's main network was dead at 12.58B bytes ([compute ladder vs BPE](./2026-11-01-compute-ladder-vs-bpe.md)). The question to ask is when the main network becomes the better predictor than stage 1. Document memory is the wrong question on text.

The test that answers Phase 1 is already planned and was stopped before it gave a result: [reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md), the plain reference layout on the 12.58B-byte ladder. In [dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md) the layout came within +0.008 to +0.034 of BPE on edu at 3.15B bytes and beat it on Go and SQL. The read does not go into that run: it is inert on text and costs 2.5% of the FLOPs.

## Related

- [Bytes read main by content](../successes/2026-11-03-bytes-read-main-by-content.md)
- [Main gradient coherence](./2026-11-03-main-gradient-coherence.md)
- [Byte and BPE controls](./2026-10-31-byte-and-bpe-controls.md)
- [Compute ladder vs BPE](./2026-11-01-compute-ladder-vs-bpe.md)
