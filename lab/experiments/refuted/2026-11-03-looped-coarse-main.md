# Experiment: looped coarse main

Status: refuted

Date proposed: 2026-11-03
Algorithm: `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/stage.py` `Main.loops` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

In a 1-stage H-Net whose main network is in use, does a weight-tied looped main network reach lower edu val bpb than an unlooped one with the same FLOPs per byte?

Context: this is the Phase 2 question of the [constitution](../../../docs/constitution.md), asked on the 1-stage H-Net, because no 2-stage model of ours has a main network in use. The constitution starts Phase 2 only after Phase 1; the user authorized this run on 2026-11-03 ("Lets do all 4") as one of four ladders, before the Phase 1 gate. The matched 1-stage uses its main network (`coarse_shuffle_s0` up to 1.71) and is 0.043 bpb behind BPE at 12.58B bytes ([compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)).

## Hypothesis

At this scale our models have more parameters than their data supports: 786M bytes is about 175M BPE tokens, a few tokens per parameter. So at fixed FLOPs per byte, the main network loses little when it trades unique layers for repeated passes, and each pass adds depth over the short coarse sequence. A looped main network then beats an unlooped one of the same FLOPs.

## Predictions

Main prediction (only this one answers the Question):

- At step 12,000, edu val bpb is lower in every R = 2 seed (T4 x 2) than in every R = 1 seed (T8 x 1), seeds 0-2. All three R = 2 seeds below the best R = 1 seed: supported. Otherwise: not supported.

Other predictions (at most two):

- Edu val bpb is lower in every R = 4 seed (T2 x 4) than in every R = 2 seed: the gain grows with R up to 4.
- The stage-0 fate (`probe/coarse_shuffle_s0` − `probe/coarse_zero_s0`) is above 0 in every seed of every arm: the main network is in use, looped or not.

If supported, we do: the looped coarse main network pays at this scale. Run R = 2 and R = 4 against R = 1 and BPE on the compute ladder, and bring the result to the user, because it bears on the order of the two phases.

If not supported, we do: at matched FLOPs, looping does not pay at 786M bytes on the 1-stage. Read R = 4 against R = 1: if R = 4 is worse than R = 1 in every seed, the parameter cut costs more than the depth gives; the next test is at matched parameters (R = 2 with T8) on the ladder, which spends more FLOPs.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including the looped-main compile parity test.
- Params 29,240,832 (R = 1), 16,391,680 (R = 2), 9,967,104 (R = 4).
- Edu ratio within 15% of 4 at the end; 786,432,000 bytes; finite loss.
- Edu fwd FLOPs/byte within 5% of 23.86M at the target ratio, and within 2% of each other across arms at the measured ratios.

## Method

```
model:  matched_1stage (stage 0 T2/T2 window 512 d256, lookahead ratio 4, main d512 global; lr_mult [1.5, 1.0])
arms:   loop1_1stage  main T8, loops 1          8 main-block passes per main position
        loop2_1stage  main T4, loops 2          (weight-tied: for r in 1..R: h = stack(h))
        loop4_1stage  main T2, loops 4
        init: output projections scaled by 1/sqrt(n_res + R x height), so every arm starts with the same scale
train:  configs/train/ladder.yaml, 12,000 steps, no rungs ; data edu 0.9 / Go 0.1
seeds:  0, 1, 2 per arm
grade:  edu val bpb at step 12,000 (final eval)
```

- Queue: `ignored/runs/next-ladder/chain.sh`, after "grown hierarchy keeps main".
- wandb project `hnet-router`, group `looped-coarse-main`.
- Cost: about 13 min per arm at the 500 W cap (matched_1stage ran 53 ms/step), about 2 h for 9 arms.

## Results

Chain `ignored/runs/next-ladder/chain.sh` (copy in [results/](../results/2026-11-03-looped-coarse-main/)), code at c5b2cae, 20:40 to 22:40 on 2026-11-03. Artifacts: `summary.txt` and `val-curves.csv` (`pull.sh`, which calls the spike's `scripts/pull_runs.py`), `loop-passes.txt` (`loop_passes.py`), `chain.sh`.

### Gates

- Suite: 198 passed, 1 skipped, with the looped-main compile parity test. Smokes of R = 2 and R = 4 passed.
- Params 29,240,832 (R = 1), 16,391,680 (R = 2), 9,967,104 (R = 4). Bytes 786,432,000. Finite loss.
- Edu ratios 3.90-4.23 (target 4, inside 15%).
- Edu fwd FLOPs/byte 22.93M-24.30M, inside 5% of 23.86M.
- **"Within 2% of each other across arms at the measured ratios" does not hold as written**: the spread is 6%. The cause is the ratio, not the arm. At the same ratio the three arms have the same FLOPs: 23.857M at 4.0, 24.296M at 3.90, 22.937M at 4.23 for each of R = 1, 2, 4 (`hnet/flops.py` on the run configs). The seed-to-seed ratio spread that the ratio gate allows moves FLOPs by up to 6%. So this gate contradicts the ratio gate, and I wrote it wrong. The R = 1 arms spent 0% to 6% more FLOPs per byte than the R = 2 arms. That is far too small to explain a 0.04-0.17 bpb gap, and the cause of the gap is found below.

### Main prediction: edu val bpb at step 12,000

| seed | R = 1 (T8 x 1) | R = 2 (T4 x 2) | R = 4 (T2 x 4) |
|---|---:|---:|---:|
| 0 | 1.5046 | 1.5754 | 1.5854 |
| 1 | 1.5311 | 1.5718 | 1.5712 |
| 2 | **1.4135** | 1.5816 | 1.5611 |
| range | 1.414-1.531 | 1.572-1.582 | 1.561-1.585 |

Every R = 2 seed is above every R = 1 seed.

### Other predictions

- Every R = 4 seed below every R = 2 seed: does not hold. The two ranges overlap (R = 4 seed 0 is 1.5854, above every R = 2 seed).
- Stage-0 fate above 0 in every seed: does not hold in any seed of any arm (-2.6 to -4.7). For R = 1 the sign is wrong as a use test. Zeroing the main output costs 3.6-3.8 bpb, far out of distribution, and shuffling it costs 0.37-1.03 bpb. So the R = 1 main network carries document content.

### The looped main network outputs a constant

| arm | shuffle cost s0 (bpb) | zero cost s0 (bpb) |
|---|---:|---:|
| R = 1 | 0.369-1.032 | 3.63-3.85 |
| R = 2 | 0.0000-0.0001 | 3.31-4.65 |
| R = 4 | 0.0000 | 3.55-4.38 |

At R = 2 and R = 4, shuffling the main output across rows costs nothing: the output is the same for every document. `loop_passes.py` measures the document share of the main stream after each pass: the across-row variance at a position, over the mean squared norm. Anchor 0: every row holds the same vector.

| run | h0 (main input) | h1 | h2 | h3, h4 |
|---|---:|---:|---:|---:|
| R = 1 seed 0 | 0.808 | 0.221 | | |
| R = 1 seed 2 | 0.450 | 0.459 | | |
| R = 2 seed 0 | 0.736 | 0.0000 | 0.0000 | |
| R = 2 seed 2 | 0.822 | 0.343 | **0.0001** | |
| R = 4 seed 0 | 0.804 | 0.0000 | 0.0000 | 0.0000 |
| R = 4 seed 2 | 0.746 | 0.0000 | 0.0000 | 0.0000 |

The input to the main network varies with the document in every run. In R = 2 seed 2 the first pass keeps it (0.343), and the second pass, with the same weights, erases it (0.0001). In the other looped runs, the stack erases it on the first pass. The looped arms are 1-stage H-Nets whose main network is a learned constant. Their bpb (1.56-1.59) is what stage 0 reaches with a constant in place of the main output.

## Verdict

**Not supported.** At step 12,000, every R = 2 seed (1.572-1.582) is above every R = 1 seed (1.414-1.531). The looped main networks carry no document content: by the last pass, the main output is the same vector for every row.

## Updated hypothesis

This run tested a weight-tied loop that feeds each pass only the previous pass's normalized output. That loop has no way to keep the input in view. Gradient descent found a map that sends every input to one vector, and the stage-0 decoder learned to ignore it. The result does not test the Phase 2 idea (more passes over the coarse sequence). It tests a loop that erases its input. I declined input injection in the [looped main note](../../../.agents/notes/implemented/feature/2026-11-03-looped-main.md) with the claim that "at R up to 4 the residual stream keeps it." The data refutes that claim at R = 2.

Recurrent-depth language models add the input back at every pass (input injection). The next test is the same three arms with input injection, `h_r = stack(h_{r-1} + P h_0)` with a learned projection P, graded on edu val bpb with the shuffle cost as the use check. If R = 2 still loses to R = 1 with a live main network, looping does not pay at 786M bytes on the 1-stage.

## Related

- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
- [Grown hierarchy keeps main](./2026-11-03-grown-hierarchy-keeps-main.md)
