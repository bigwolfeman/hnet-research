# Experiment: end-of-chunk vs lookahead router

Status: refuted

Date proposed: 2026-10-30
Algorithm: [Cartesia blog notes, "Inferred" section](../../../docs/references/cartesia-tokens-to-bytes.md)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does an end-of-chunk boundary rule (boundary decided from bytes up to and including the last byte of a chunk) make a 2-stage byte H-Net better than the 2025 lookahead rule (boundary decided at the first byte of the next chunk)? Cartesia say the lookahead rule hurts next-byte prediction and multi-stage hierarchies, but they do not say what replaced it.

## Hypothesis

In a 2-stage all-Transformer H-Net at matched FLOPs/byte and matched target ratios, the `eoc` router reaches lower validation bpb than the `lookahead` router, and the gap is larger for 2 stages than for 1 stage.

## Predictions

Main prediction (only this one answers the Question):

- At the end of training, 2-stage `eoc` has lower validation bpb than 2-stage `lookahead`, by more than the seed spread of the 2-stage `lookahead` runs.

Other predictions (at most two):

- The bpb gap (`lookahead` minus `eoc`) is larger at 2 stages than at 1 stage.
- Stage-0 `eoc` boundaries have higher whitespace-boundary F1 than stage-0 `lookahead` boundaries.

If supported, we do: make `eoc` the default router, write an Agent Note, and use it for the 1-stage vs 2-stage vs BPE comparison.

If not supported, we do: keep `lookahead` as default. Next, test a cosine-similarity form of `eoc` (compare h_t with a predicted h_{t+1}). If that also fails, the new router differs from our reading, and we revise the hypothesis in the reference notes.

Gates (the run is broken if one fails; gates are not predictions):

- Causality unit test passes for both routers (no future-byte leak).
- Measured ratio per stage stays within 15% of its target after warmup and never collapses to 1x or one chunk.
- Finite loss for the whole run. Equal FLOPs/byte across arms within 3%, using measured ratios.

## Method

```
for stages in [1, 2]:
  for router in [lookahead, eoc]:
    for seed in seeds:
      train H-Net(stages, router, targets=[4] or [4, 2]) on FineWeb-Edu bytes
      log val bpb, measured ratios, flops/byte, boundary F1, in-context probe
```

- Code: the smoke spike `lab/spikes/2026-10-30-two-stage-hnet-smoke/` (see that README for the router pseudocode).
- Size and step count: set after the smoke runs from measured throughput, so that the 2-stage arms fit in one night on the 5090. Recorded here as an amendment before the run.
- Seeds: 2 per arm at least (gives the spread for "the same").
- Controls: same data order, same optimizer and schedule, same encoder/decoder/main shapes. Only `router` changes within a stage count.

### Amendment 2026-10-30 (before any experiment run)

- `eoc` means `read_own_end=true, force_last=false`: byte t reads the latest chunk ending at or before t. The smoke spike first shipped a one-chunk-lag read (`read_own_end=false`); that variant runs as an extra arm `eoc_lag` and does not answer the Question.
- `force_last=true` with `read_own_end=true` leaks "sequence ends here" into the last position, so the code rejects that combination.
- Data: FineWeb-Edu bytes, `ignored/data/fineweb-edu/` (12 GB train, 64 MB val). The 2000-step smoke runs used plain FineWeb and are not results for this experiment.

### Amendment 2026-10-31 (size and protocol, before any experiment run)

- Model: d_model [256, 384, 512] (stage 0, stage 1, main), main 8 layers, enc/dec 2 layers per stage, windows 512 / 256, head dim 64. 36.6M params at 2 stages. Ratio-loss weight 0.03 per stage, targets 4 then 2.
- Train: `train=router`. seq 4096 bytes, batch 16, 12,000 steps (786M bytes per arm), AdamW lr 8e-4 with stage multipliers [1.5, 1.2, 1.0], WSD with 500 warmup and 20% linear decay, bf16, compiled stacks. Data mix edu 0.9 / go 0.1. Val: edu (primary), go, sql.
- Arms in this order: 2-stage eoc s0, lookahead s0, eoc s1, lookahead s1; 1-stage eoc s0, lookahead s0, eoc s1, lookahead s1; eoc_lag 2-stage s0. Launcher: `lab/spikes/2026-10-30-two-stage-hnet-smoke/scripts/run_router_experiment.sh`. wandb project `hnet-router`, group `end-of-chunk-router`.
- Data is read from an NVMe copy (`~/data/hnet-bytes`, same bytes as `ignored/data`). The first launch read from the HDD, slowed to 122 kB/s by step 550, and was stopped and restarted. Its partial arm is not a result.
- Hardware: local RTX 5090. Measured before launch: 360 kB/s for 60 steps compiled, but the GPU ran kernels only 48% of the wall time. Throughput work runs in parallel on other hardware and does not change this experiment's code.

### Amendment 2026-10-31 (seed 2 for the main arms, decided before any arm-vs-arm data)

- On NVMe data the 2-stage arm runs at about 950 kB/s (68 ms/step), 3.6x the pre-launch estimate, so the planned arms leave most of the night free. Seed 2 of 2-stage `eoc` and 2-stage `lookahead` runs after the planned arms. At decision time only arm 1 (eoc 2-stage s0) was partly trained and no other arm had started. Reason: three seeds give a better spread for the main prediction's "more than the seed spread" test. The main prediction uses all seeds that finish.
- Launch: `scripts/run_arms.sh` (generic sequential runner); `scripts/run_router_experiment.sh` lists the full arm set.

## Results

**Not supported.** 2-stage `eoc` has *higher* edu val bpb than 2-stage `lookahead` in every seed. Paired by seed (same data order), `eoc` minus `lookahead` = +0.096, +0.038, +0.095 bpb, mean +0.076. The worst `lookahead` seed (1.446) beats the best `eoc` seed (1.467).

All runs: 12,000 steps, 786M bytes, local RTX 5090. Final eval at step 12,000. Per-run numbers and wandb links: [summaries.json](../results/2026-10-30-end-of-chunk-router/summaries.json). Samples: [seed 0](../results/2026-10-30-end-of-chunk-router/samples-s0.md), [other seeds and arms](../results/2026-10-30-end-of-chunk-router/samples-rest.md).

| Arm | edu bpb (s0 / s1 / s2) | Ratios s0 x s1 (seed 0) | Fwd FLOPs/byte | Stage-0 word F1 (model / stride) | kB/s |
|---|---|---|---|---|---|
| eoc 2-stage | 1.510 / 1.467 / 1.542 | 3.51 x 2.00 | 21.3M | 0.13, 0.63, 0.20 / 0.29 | 850-970 |
| lookahead 2-stage | 1.414 / 1.429 / 1.446 | 3.13 x 1.98 | 23.5M | 0.76, 0.74, 0.82 / 0.32 | 835-900 |
| eoc 1-stage | 1.558 / 1.552 | 3.88 | 24.1M | 0.64, 0.66 / 0.29 | 1070-1110 |
| lookahead 1-stage | 1.407 / 1.414 | 3.10 | 29.0M | 0.46, 0.49 / 0.32 | 980-1020 |
| eoc_lag 2-stage (extra arm) | 1.558 | 4.06 x 1.97 | 19.6M | 0.18 / 0.27 | 1030 |

Seed spread (max minus min) of edu bpb: 2-stage `lookahead` 0.032, 2-stage `eoc` 0.075.

Go and SQL val bpb follow the same order (2-stage Go: `eoc` 0.98-1.02, `lookahead` 0.96-0.99).

### Gate deviation

The `lookahead` arms failed two gates. Their stage-0 ratio settled at 3.1x, 22% below the 4x target (gate: 15%). Thus they spend 10% more FLOPs/byte than the `eoc` arms (gate: 3%). By protocol that makes the comparison a broken run. I still file a verdict, for this reason: the deviation gives the control (`lookahead`) more compute, so it works against `lookahead`'s loss, not for it. Inside this experiment, 23% more FLOPs (lookahead 1-stage vs 2-stage) buys at most 0.02 bpb. The 10% gap would buy about 0.01. A compute power law with exponent 0.05 gives a similar 0.007. Both are under 1/7 of the measured +0.076. A ratio-matched rerun cannot plausibly flip the sign. The rerun is still owed (see Updated hypothesis).

### Other predictions

- Larger gap at 2 stages: the `lookahead`-minus-`eoc` gap is -0.076 at 2 stages and -0.145 at 1 stage. It is less negative at 2 stages, but `eoc` never leads, so the mechanism the prediction stood for (`eoc` helps more in deeper hierarchies) has no support. The 1-stage pair is also FLOPs-confounded (29.0M vs 24.1M).
- Higher stage-0 word F1 for `eoc`: not supported. `lookahead` 0.74-0.82, `eoc` 0.13-0.63.

### What the samples show

- 2-stage `eoc` has two outcomes. Seed 1 found word-aligned cuts (F1 0.63, best `eoc` bpb). Seeds 0 and 2 have F1 *below* the fixed-stride chunker. Their cuts sit one byte after each word start (seed 0: `M|ak|ing u|se o#f b|right`). The chunk then holds the first letter of the next word. The coarse stages see that letter before they predict the rest of the word. This is the same "peek" failure that Cartesia show for their lazy/lookahead variant (`fro|m the fact`, `r|hino`).
- `lookahead` cuts land on word starts and spaces in all three seeds (`Making| |use# |of#`).
- 1-stage `eoc` is word-aligned in both seeds and ends chunks on the space (`use |of |bright |lights`). It still loses to 1-stage `lookahead` by 0.145 bpb. The peek optimum shows up only with a second stage on top, so it does not explain the whole gap. A `lookahead` chunk starts with the first byte of a word, so its coarse vector always sees that byte. An `eoc` chunk that ends on the space sees none of the next word.
- Measured hierarchy behavior, not part of the predictions: final-stage hnet_over_bpe is higher on Go (2.18) and SQL (1.85-1.91) than on edu (1.31-1.33) in every `lookahead` 2-stage seed, the Figure 6 direction (Constitution criterion 4). In-context drop (chunks per occurrence, occurrence 1 to 5) is 0.0-0.45 with no consistent sign: no in-context tokenization at 786M bytes.

## Verdict

**Not supported.** 2-stage `eoc` is worse than 2-stage `lookahead` by +0.076 bpb (paired mean of 3 seeds), outside both arms' seed spread. The `lookahead` arms broke the ratio and FLOPs gates in the direction that cannot explain the gap.

As planned: `lookahead` stays the default router.

## Updated hypothesis

Our `eoc` reading (a sigmoid on h_t, byte t reads its own chunk) is not the new Cartesia router. The data says the coarse network gains a lot from seeing the first byte of the unit it is about to predict. `lookahead` gives it that byte by construction. Clean `eoc` (ends on the space) withholds it and loses 0.145 bpb at 1 stage. Under 2 stages, `eoc` often learns to cut one byte into the next word to buy that byte back, and the cuts stop being words. A better causal router must keep word-aligned units *and* give the coarse stage that first byte. How Cartesia do that is still unknown to us. Next:

1. A ratio-matched `lookahead` rerun (raise its ratio-loss weight until stage 0 holds 4x). This closes the gate deviation and gives the controls experiment a gate-clean H-Net arm.
2. The planned cosine form of `eoc` (compare h_t with a predicted h_{t+1}). It runs only after the rerun.

## Related

- [Constitution](../../../docs/constitution.md), Success criteria 1-3.
