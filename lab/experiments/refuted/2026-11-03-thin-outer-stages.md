# Experiment: thin outer stages

Status: refuted

Date proposed: 2026-11-03
Algorithm: none (a config: `configs/experiment/thin_outer_2stage.yaml`)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

If the outer stages of the 2-stage H-Net are thin and short-range, with the FLOPs moved into the main network, does the model reach lower edu val bpb than the reference layout at the same FLOPs per byte?

Context: in the reference layout, stage 0 is Mamba-2 (a state over the whole row) and stage 1 attends over 1,024 chunks, so the outer stages can model most of a 4,096-byte row without the main network. They do: the stage-1 fate ends above 0 in one of five seeds ([main gradient coherence](../refuted/2026-11-03-main-gradient-coherence.md)). Moving stage-1 blocks into the main network with the long-range outer paths kept cost +0.108 bpb ([deep main compute split](../refuted/2026-10-31-deep-main-compute-split.md)). This arm removes the long-range outer paths too.

## Hypothesis

The main network is unused because the outer stages can do its job. With outer attention windows of 64 bytes and 32 chunks, only the main network sees more than about 100 bytes back, so it must carry the context, and the FLOPs that the outer stages gave up buy a deeper main network (22 layers instead of 10).

## Predictions

Main prediction (only this one answers the Question):

- At step 12,000, edu val bpb is lower in every thin-outer seed (0, 1, 2) than in every plain seed (the five plain reference-layout seeds, 1.3133 to 1.3661). All three below 1.3133: supported. Otherwise: not supported.

Other predictions (at most two):

- The stage-1 fate (`probe/coarse_shuffle_s1` − `probe/coarse_zero_s1`) is above 0 in every thin-outer seed (plain control: 1 of 5).
- `probe/coarse_shuffle_s1` is above every plain seed's value (largest 0.158) in every thin-outer seed: the stage-1 decoder depends on the main output for content.

If supported, we do: a strong main network behind weak outer stages wins at matched FLOPs. Run it on the compute ladder against BPE, and write an Agent Note on the layout, because it moves away from the reference layout toward the constitution's Transformer-only sketch.

If not supported, we do: read the stage-1 fate. If the main network is in use but bpb is worse, the short outer windows cost more local modeling than the deeper main network gives back; the outer windows are the next knob. If the main network is still unused, the cause is not the outer paths' reach.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`).
- Params 74,606,848.
- Edu ratios within 15% of 3 x 3 at the end; 786,432,000 bytes; finite loss.
- Edu fwd FLOPs/byte within 5% of 22.19M at the target ratios (plain reference layout: 22.58M).

## Method

```
model:   stage 0  enc T1 / dec T1, window 64 bytes, d256, lookahead ratio 3
         stage 1  enc T1 / dec T1, window 32 chunks, d256, lookahead ratio 3
         main     T22 global, d512 (reference layout: T10)
         lr 6.25e-4 x [3.03, 1.75, 0.72], RatioController 0.002, rope_dim 32, mlp 2.75 x 128 (as reflayout_2stage)
train:   configs/train/ladder.yaml, 12,000 steps, no rungs ; data edu 0.9 / Go 0.1
seeds:   0, 1, 2
control: plain reflayout_2stage, seeds 0-4 = the edu arms of main gradient coherence
grade:   edu val bpb at step 12,000 (final eval)
```

- The arm has 74.6M params against 41.8M at about the same FLOPs: a gain may come from parameters, not only from where the context is modeled. The stage-1 fate separates the two readings.
- Queue: `ignored/runs/next-ladder/chain.sh`, after "looped coarse main".
- wandb project `hnet-router`, group `thin-outer-stages`.
- Cost: about 20 min per arm, about 1 h in total.

### Amendment 2026-11-03 23:15 (seed 2 broken, rerun queued)

- Seed 2 stopped with CUDA out of memory between steps 50 and 100: 24.45 GB allocated and 2.54 GB reserved but unallocated, in the stage glue, while the early stage-1 ratio swung (47.8 at step 50). Seeds 0 and 1 peaked at 16.5 and 17.5 GB. The log is kept as `thin_outer_2stage_s2.oom.log`.
- Fix: the same command with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, after the queue (A) ends: `ignored/runs/next-ladder/rerun_thin_s2.sh` (copy in results/). Its result goes into Results below. It cannot change the verdict: seeds 0 and 1 are already above 1.3133.

## Results

Chain `ignored/runs/next-ladder/chain.sh` (copy in [results/](../results/2026-11-03-thin-outer-stages/)), code at c5b2cae, 22:40 to 23:12 on 2026-11-03. Artifacts: `summary.txt` and `val-curves.csv` (`pull.sh`, which calls the spike's `scripts/pull_runs.py`), `chain.sh`, `rerun_thin_s2.sh`. wandb: 8r2q0dl5, jm8sa6lt. Seed 2 rerun (`rerun_thin_s2.sh`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), code at 5e4cba7, 10:58 to 11:13 on 2026-11-04: wandb yviihgz2.

### Gates

- Suite: 198 passed, 1 skipped. Smoke passed.
- Params 74,606,848. Bytes 786,432,000. Finite loss.
- Edu ratios 2.96-3.00 x 2.93-3.03.
- Edu fwd FLOPs/byte 22.38M, 22.77M, and 22.02M (+0.9%, +2.6%, and −0.7% against 22.19M).
- Seed 2: the first attempt was broken (out of memory), see the amendment. The rerun peaked at 25.6 GB and passed every gate.

### Main prediction: edu val bpb at step 12,000

| seed | thin outer | edu bpb | Go bpb | zero s1 | shuf s1 | shuf s0 |
|---|---|---:|---:|---:|---:|---:|
| 0 | T1/T1 w64, T1/T1 w32, main T22 | 1.6253 | 1.6855 | 0.5425 | 0.2058 | 1.4949 |
| 1 | same | 1.6282 | 1.7202 | 2.0007 | 0.1504 | 1.2426 |
| 2 | same | 1.6296 | 1.6786 | 1.1987 | 0.2240 | 1.1517 |
| plain 0-4 | m4/m4, T1m4/m4T1 w1024, main T10 | 1.3133-1.3661 | 0.889-0.976 | 0.057-0.154 | 0.0005-0.158 | 0.43-1.11 |

All three thin seeds are above every plain seed, by 0.26 to 0.32 bpb.

### Other predictions

- Stage-1 fate above 0 in every thin seed: does not hold (-0.337, -1.850, -0.975).
- `probe/coarse_shuffle_s1` above 0.158 in every thin seed: holds in seeds 0 and 2 (0.206, 0.224), not in seed 1 (0.150).

### Observations (not predictions)

- **The main network carries some document content, about as much as the best plain seed.** Shuffle costs at stage 1 are 0.15 to 0.22, at the top of the plain range (up to 0.158). Removing the long-range outer paths did not make the main network the strong path. The large zero costs (0.54 to 2.00) show that the stage-1 decoder leans on the main output's constant part.
- **The loss is local modeling.** The thin layout has one block per side at each stage, against four Mamba-2 blocks per side at stage 0 in the reference layout. Go bpb (1.68-1.72) is 0.7-0.8 worse than plain (0.89-0.98). Code depends on exact local syntax. A 22-layer main network at one ninth of the byte rate does not give that back.
- The run changed two things in the outer stages at once: depth (one block per side) and reach (64-byte and 32-chunk windows). So it cannot say which one cost the most.

## Verdict

**Not supported.** At step 12,000, edu val bpb is 1.6253, 1.6282, and 1.6296 in seeds 0, 1, and 2, above every plain seed (1.3133-1.3661).

## Updated hypothesis

Taking the reach and depth away from the outer stages does not make the main network do their job. The main network's use went up only to about the level of the best plain seed. The model lost 0.3 bpb on edu and 0.75 on Go, so the outer stages' local modeling is worth much more at 786M bytes than a deeper main network. The plan's next knob was the outer windows. I do not take it. The lead from [grown hierarchy keeps main](./2026-11-03-grown-hierarchy-keeps-main.md) is narrower and cheaper. Keep the outer depth, replace only the Mamba-2 stage-0 mixer with windowed attention (as in the 1-stage model whose main network is in use in every seed), and grow the 2-stage model from that 1-stage.

## Related

- [Deep main compute split](../refuted/2026-10-31-deep-main-compute-split.md)
- [Main gradient coherence](../refuted/2026-11-03-main-gradient-coherence.md)
- [Grown hierarchy keeps main](./2026-11-03-grown-hierarchy-keeps-main.md)
