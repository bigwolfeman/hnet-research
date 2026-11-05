# Experiment: windowed growth keeps main

Status: refuted

Date proposed: 2026-11-04
Algorithm: `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/grow.py` (pseudocode there), `train.grow_from`; config `configs/experiment/windowed_2stage.yaml`
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

If a 2-stage H-Net with windowed-attention outer stages grows from a 1-stage model whose main network is in use, does the main network stay in use in every seed?

Context: in [grown hierarchy keeps main](./2026-11-03-grown-hierarchy-keeps-main.md), the growth kept the 1-stage state in every seed: the one seed with a live 1-stage main network ended with a stage-1 shuffle cost of 0.62, and the two dead seeds stayed dead. The dead seeds were dead before the growth, in a 1-stage model with a Mamba-2 stage 0. The 1-stage model with a windowed-attention stage 0 (`loop1_1stage`: T2/T2, 512 bytes) uses its main network in every seed: in [looped coarse main](./2026-11-03-looped-coarse-main.md), its shuffle cost at stage 0 is 0.09-0.60 at step 4,000 and 0.37-1.03 at step 12,000. In the plain reference layout, the largest stage-1 shuffle cost of five seeds is 0.158.

## Hypothesis

Whether the main network is used depends on the reach of the outer mixers. A Mamba-2 stage has a state over the whole 4,096-byte row and can learn the main network's job before the main network does. Windowed attention cannot see past its window, so the context beyond it must come through the main network. With windowed attention at both outer stages, and a main network that is in use at the growth, the main network stays in use in every seed.

## Predictions

Use is the shuffle cost: the bpb rise when the main output at the stage-1 decoder is shuffled across rows (`probe/coarse_shuffle_s1`). Anchor 0: no document content. Plain control: the five reference-layout seeds of [main gradient coherence](./2026-11-03-main-gradient-coherence.md), 0.0005 to 0.158.

Main prediction (only this one answers the Question):

- At the end (8,000 steps after the growth), `probe/coarse_shuffle_s1` is above 0.158 in every grown seed (0, 1, 2). All three above: supported. Any at or below: not supported.

Other predictions (at most two):

- In the same windowed 2-stage layout trained from scratch for 12,000 steps, `probe/coarse_shuffle_s1` is lower than in every grown seed, in every scratch seed: the growth, not the layout alone, keeps the main network.
- Edu val bpb at the end is lower in every grown seed than in every scratch seed. The grown arm spends about 33% more training FLOPs (the 1-stage phase costs 23.86M fwd FLOPs/byte, the 2-stage 17.96M), so this is not at matched compute.

If supported, we do: windowed outer stages plus growth give a 2-stage H-Net whose main network is used in every seed. Next, bring the layout to matched FLOPs against BPE and the reference layout on the compute ladder, and write an Agent Note on the layout and the training recipe. If the scratch arm also passes 0.158 in every seed, the layout alone does it, and the ladder uses the scratch recipe.

If not supported, we do: read the seeds. If a grown seed's main network went dead after the growth, then a live start does not survive a coarser main rate (1/4 to 1/8 of the bytes) with windowed outer stages, and the race is not only about the Mamba-2 reach. Then the 2-stage question goes back to the reference layout at scale, which A ([reference layout vs BPE](./2026-11-02-reference-layout-vs-bpe.md)) measures.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`).
- Params 29,240,832 (1-stage) and 32,651,520 (2-stage). Each growth log line reports the fresh keys as the new stage's only.
- The precondition: phase-1 stage-0 shuffle cost at step 4,000 above 0.0033 in every seed (the larger of the two dead phase-1 seeds of the Mamba-2 growth). On record for this config: 0.09-0.60 at step 4,000. If a seed fails it, that seed does not test the Question; the run is reported, and the seed is replaced by seed 3.
- Edu ratios within 15% of 4 (1-stage) and 4 x 2 (2-stage) at the end; bytes 262,144,000 + 524,288,000 (grown) and 786,432,000 (scratch); finite loss.
- Edu fwd FLOPs/byte within 5% of 23.86M (1-stage) and 17.96M (2-stage) at the target ratios.

## Method

```
phase 1:  loop1_1stage (stage 0 T2/T2 window 512 d256, ratio 4; main T8 d512; lr 8e-4 x [1.5, 1.0])
          configs/train/ladder.yaml, 4,000 steps, decay_frac 0 (warmup 500, then constant LR), no rungs -> final.pt
growth:   windowed_2stage with train.grow_from = phase-1 final.pt (hnet/grow.py):
          embed, head, stage 0, main <- phase 1 ; stage 1 (T2/T2 window 512 chunks, d256, ratio 2) fresh ;
          phase-1 stage-0 pad_dim (256 -> 512) -> stage 1's pad_dim ; AdamW state fresh
phase 2:  8,000 steps of ladder.yaml (warmup 500, WSD decay over the last 20%), lr 8e-4 x [1.5, 1.2, 1.0], no rungs
scratch:  windowed_2stage from its own init, 12,000 steps of ladder.yaml, no rungs
seeds:    0, 1, 2 in each arm (phase 2 seed = phase 1 seed)
grade:    probe/coarse_shuffle_s1 at the last eval (8 edu val batches)
```

- The 2-stage layout spends 17.96M fwd FLOPs/byte at 4 x 2, against 22.58M for the reference layout. Stage 0 and the main network must keep the 1-stage shapes for the growth, so the layout is not matched to BPE here. The Question is about use, not bpb.
- The main network's input rate changes at the growth from 1/4 to 1/8 of the bytes.
- Queue: after A and the thin-outer seed 2 rerun, `ignored/runs/windowed-growth/chain.sh`. A 200-step smoke of each phase runs first.
- wandb project `hnet-router`, groups `windowed-growth-p1`, `windowed-growth`, `windowed-scratch`.
- Cost: about 4 min per 1-stage phase, about 7 min per grown phase, about 10 min per scratch arm, about 65 min in total.

## Results

Chain `ignored/runs/windowed-growth/chain.sh` (copy in [results/](../results/2026-11-04-windowed-growth-keeps-main/)), code at 5e4cba7, 11:15 to 12:24 on 2026-11-04. Artifacts: `summary.txt` and `val-curves.csv` (`pull.sh`). wandb: phase 1 eventxo4, f06x96v5, eegrhaxj; grown pexwcxjc, kdaydhpz, d2r9fr5p; scratch 6hph3e1n, n5r5oyf7, 8vuocj2b.

### Gates

- Suite: 211 passed, 1 skipped. Three smokes passed.
- Params 29,240,832 and 32,651,520. Each growth log line reports "30 fresh keys in the new stage".
- Precondition: phase-1 stage-0 shuffle cost at step 4,000 is 0.125, 0.089, and 0.658 (seeds 0-2), all above 0.0033. No seed is replaced.
- Edu ratios: 1-stage 3.98, 4.29, 3.90; grown 4.03 x 1.99, 4.07 x 1.98, 3.93 x 2.05; scratch 4.07 x 1.98, 4.00 x 1.99, 3.98 x 1.96. All within 15%.
- Bytes 262,144,000 + 524,288,000 (grown) and 786,432,000 (scratch). Finite loss.
- Edu fwd FLOPs/byte: 1-stage 23.94M, 22.73M, 24.31M (−4.7% to +1.9% against 23.86M); 2-stage 17.88M-18.16M (−0.4% to +1.1% against 17.96M).

### Main prediction: `probe/coarse_shuffle_s1` at the end

| seed | phase-1 shuffle s0 (step 4,000) | grown shuffle s1 | scratch shuffle s1 | grown edu | scratch edu |
|---|---:|---:|---:|---:|---:|
| 0 | 0.125 | 0.195 | 0.0001 | 1.4993 | 1.4841 |
| 1 | 0.089 | **0.123** | 0.0000 | 1.5352 | 1.4998 |
| 2 | 0.658 | 0.340 | 0.0002 | 1.4304 | 1.4621 |

Seeds 0 and 2 are above 0.158. Seed 1 is below it.

### Other predictions

- Every scratch seed's shuffle cost is below every grown seed's: holds. Scratch 0.0000-0.0002, grown 0.123-0.340.
- Every grown seed's edu bpb is below every scratch seed's: does not hold. Grown seed 2 (1.4304) is below every scratch seed, but seeds 0 and 1 are not, although the grown arm spent about 33% more training FLOPs.

### Observations (not predictions)

- **The growth carries use, and it does not add any.** The order of the three seeds at the end (1 < 0 < 2) is the order of their phase-1 shuffle costs (0.089 < 0.125 < 0.658). Grown seed 1 did not go dead after the growth. It kept about the level it started with, and that level was below the plain line.
- **The windowed layout alone does not bring the main network into use.** From scratch, the stage-1 decoder ignores the main network's content in every seed (shuffle cost 0.0002 or less), the same as the dead plain seeds. The hypothesis put the cause in the outer mixers' reach, but a 512-chunk stage-1 window still covers about 2,000 bytes, half a row. So this layout does not test short reach at stage 1.
- **The windowed layout is much worse than the reference layout on bpb at 786M bytes:** 1.43-1.54 against 1.31-1.37, at 18.0M against 22.6M fwd FLOPs/byte. That comparison is not matched.
- **Against scale:** at 12.58B bytes, the plain reference layout uses its main network at stage 1 in both seeds (shuffle cost 0.67 and 0.86, [reference layout vs BPE](./2026-11-02-reference-layout-vs-bpe.md)). Budget brings the main network into use there. Neither growth nor the windowed layout does it at 786M bytes.

## Verdict

**Not supported.** At the end, `probe/coarse_shuffle_s1` is 0.195 and 0.340 in grown seeds 0 and 2, but 0.123 in seed 1, at or below the plain control's 0.158.

## Updated hypothesis

Main-network use at stage 1 depends on training budget, not on the outer mixers' reach. Growth transfers the 1-stage model's level of use to the 2-stage model, and it does not raise that level. A live start keeps the main network alive (no grown seed went dead, against 3 of 3 dead from scratch), but the amount of use is the 1-stage lottery again. I stop working on use at 786M bytes. At 12.58B bytes the reference layout already uses its main network in both seeds, so the open Phase 1 question is the BPE margin: [reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md).

## Related

- [Reference layout vs BPE](./2026-11-02-reference-layout-vs-bpe.md)
- [Grown hierarchy keeps main](./2026-11-03-grown-hierarchy-keeps-main.md)
- [Looped coarse main](./2026-11-03-looped-coarse-main.md)
- [Thin outer stages](./2026-11-03-thin-outer-stages.md)
