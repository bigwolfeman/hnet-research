# Experiment: Parcae coarse main

Status: refuted

Date proposed: 2026-11-04
Algorithm: `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/parcae.py` (pseudocode there; a port of Parcae, arXiv 2604.12946, MIT code)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

In the 1-stage H-Net, does a Parcae looped main network (prelude, core loop with stable input injection, coda) reach lower edu val bpb than the unlooped main network with the same FLOPs per byte?

Context: the naive weight-tied loop in [looped coarse main](../refuted/2026-11-03-looped-coarse-main.md) erased its input. At R = 2 and R = 4 the main output was the same vector for every row (shuffle cost 0.0000-0.0001), and every looped seed lost to every R = 1 seed (1.4135-1.5311). Parcae re-injects the prelude output at every step through a diagonal map whose decay is in (0, 1), samples the depth per row, and backpropagates through the last steps only. Parcae beats parameter-matched Transformers at 140M-1.3B parameters. The user asked for this port on 2026-11-04 ("we need to borrow the parcae implementation. Prelude, core loop, coda"), as Phase 2 work ahead of the Phase 1 gate, as with the first looped run.

## Hypothesis

The naive loop failed because nothing kept the input in view. With injection, every step sees the coarse input, and the contraction keeps the state bounded. At this scale our models have more parameters than the data supports (786M bytes, about 175M BPE tokens, for 23M-29M parameters). So a main network that trades unique layers for repeated core steps at the same FLOPs loses little capacity and gains depth over the short coarse sequence.

## Predictions

Control: the three R = 1 seeds of looped coarse main (`loop1_1stage`, main T8, seeds 0-2, edu 1.5046, 1.5311, 1.4135; wandb zhpmgzlf, mzq49grx, zssjv0k6). The code path of that arm is unchanged by the Parcae port (`Main` with loops 1).

Main prediction (only this one answers the Question):

- At step 12,000, edu val bpb is lower in every Parcae R = 2 seed (prelude T2, core T2 x mean 2, coda T2) than in every R = 1 seed. All three below 1.4135: supported. Otherwise: not supported.

Other predictions (at most two):

- The main network carries document content in every Parcae seed of both arms: `probe/coarse_shuffle_s0` above 0.0001 (the largest naive-loop value) at step 12,000.
- Edu val bpb is lower in every Parcae R = 6 seed (prelude T1, core T1 x mean 6, coda T1) than in every R = 1 seed.

If supported, we do: a stable looped main pays at matched FLOPs on the 1-stage H-Net. Run it on the compute ladder against BPE and the 1-stage, and put the Parcae main into the 2-stage model that comes out of [windowed growth keeps main](../refuted/2026-11-04-windowed-growth-keeps-main.md). Bring it to the user, because it bears on the order of the two phases.

If not supported, we do: read the use prediction first. If the Parcae main network carries content but loses on bpb, then at 786M bytes, unique layers are worth more than repeated steps at the same FLOPs; the next test is at matched parameters (Parcae's own comparison: prelude T2, core T4, coda T2, more FLOPs). If it carries no content, the injection did not fix the collapse in our H-Net, and the cause is in the chunk interface, not the loop.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), with `tests/test_parcae.py` and the Parcae compile parity tests.
- Params 23,342,080 (R = 2) and 13,705,216 (R = 6).
- Edu ratio within 15% of 4 at the end; 786,432,000 bytes; finite loss.
- Edu fwd FLOPs/byte within 5% of 24.12M at ratio 4 (the R = 1 arm: 23.86M; the Parcae main adds the injection and C, 2 d^2 each per main position). Compare arms at the same ratio, not at each run's measured ratio (the lesson of looped coarse main).
- `train/parcae/decay_max` below 1 at every log (it holds by construction; a value of 1 means a broken parameterization).

## Method

```
model:  loop1_1stage's stage 0 (T2/T2 window 512 bytes, d256, lookahead ratio 4; lr 8e-4 x [1.5, 1.0])
        main (d512) = ParcaeMain:
          e = prelude(h)                               # Stack ending in RMSNorm
          decay = exp(-softplus(dt_bias) exp(A_log))   # init: A_log 0, decay sqrt(1/5); B = I
          x ~ TruncNormal(0, 0.02)                     # fixed generator at eval
          train: T_i ~ max(1, Poisson(R)) per row; last min(T_i, s) steps with grad, the rest under no_grad
          eval:  R steps;   x <- core(decay x + dt B e)
          y = coda(C x)                                # Stack ending in RMSNorm
arms:   parcae_r2_1stage  prelude T2, core T2, coda T2, R 2, s 1    2 + 2x2 + 2 = 8 block passes
        parcae_r6_1stage  prelude T1, core T1, coda T1, R 6, s 3    1 + 6x1 + 1 = 8 block passes
        (s = ceil(R / 2), as the paper's backprop depth)
train:  configs/train/ladder.yaml, 12,000 steps, no rungs ; data edu 0.9 / Go 0.1 ; AdamW (not Parcae's Muon);
        injection params (A_log, dt_bias, B) without weight decay, as Parcae
seeds:  0, 1, 2 per arm
grade:  edu val bpb at step 12,000 (final eval)
```

- Fwd FLOPs/byte match at the mean depth. Training FLOPs do not: the no-grad steps cost a forward only, so the Parcae arms spend less training compute than R = 1. Wall time is higher than the FLOPs say, because each batch runs to the deepest row (Poisson tail).
- Differences from Parcae: our blocks and init rule (not scaled-zero), AdamW, no value embeddings, the eval state from a fixed generator, and the H-Net around the main network. The port is faithful in the injection, the state init, the depth sampler (`poisson-truncated-full`, per row), and the no-grad prefix.
- Queue: after windowed growth keeps main, `ignored/runs/parcae-coarse-main/chain.sh`. A 200-step smoke of each arm runs first.
- wandb project `hnet-router`, group `parcae-coarse-main`.
- Cost: about 15-20 min per arm, about 1.5-2 h in total.

## Results

Chain `ignored/runs/parcae-coarse-main/chain.sh` (copy in [results/](../results/2026-11-04-parcae-coarse-main/)), code at 5e4cba7 (committed before the run; aeeca90 changed only a link in this file), 12:24 to 13:41 on 2026-11-04. Artifacts: `summary.txt` and `val-curves.csv` (`pull.sh`, with the R = 1 control ids). wandb: R = 2 fj02tf2j, 5kvuseai, unwkowfr; R = 6 3v2exqwj, u25meyq8, hrz1echo.

### Gates

- Suite: 211 passed, 1 skipped. Both smokes passed.
- Params 23,342,080 (R = 2) and 13,705,216 (R = 6).
- Edu ratios 3.94-4.12. Bytes 786,432,000. Finite loss.
- Edu fwd FLOPs/byte 23.63M-24.38M at the measured ratios, within 2.1% of 24.12M.
- `train/parcae/decay_max` peaks at 0.59-0.69 over 241 logs per run, below 1. The mean training depth is 2.12-2.15 (R = 2) and 5.97-6.03 (R = 6), as the truncated Poisson gives (R + e^-R for R = 2).

### Main prediction: edu val bpb at step 12,000

| seed | R = 1 control (T8) | Parcae R = 2 (T2 + T2 x 2 + T2) | Parcae R = 6 (T1 + T1 x 6 + T1) |
|---|---:|---:|---:|
| 0 | 1.5046 | 1.5286 | 1.4878 |
| 1 | 1.5311 | 1.5568 | 1.5644 |
| 2 | **1.4135** | 1.5730 | 1.5598 |
| shuffle s0 (0 / 1 / 2) | 0.443 / 0.369 / 1.032 | 0.897 / 0.095 / 0.055 | 0.868 / 0.098 / 0.151 |

Every R = 2 seed is above 1.4135.

### Other predictions

- The main network carries document content in every Parcae seed (shuffle cost above 0.0001): holds. 0.055-0.897 (R = 2) and 0.098-0.868 (R = 6), against 0.0000-0.0001 for the naive loop at R = 2 and 4.
- Every R = 6 seed below every R = 1 seed: does not hold. R = 6 seed 0 (1.4878) is below R = 1 seeds 0 and 1, but no R = 6 seed is below 1.4135.

### Observations (not predictions)

- **The injection fixes the collapse.** The naive loop's main output was one vector for every row. With Parcae's injection, the main output carries document content in every seed, up to the level of the unlooped main (seed 0: 0.87-0.90 against 0.44).
- **At matched forward FLOPs, the loop costs bpb at this scale.** The R = 2 mean is 1.553 and the R = 6 mean is 1.537, against 1.483 for R = 1. The spread of R = 1 alone is 0.118, so the arm means are inside about one control spread.
- **R = 6 matches R = 2 with 41% fewer parameters** (13.7M against 23.3M): the means differ by 0.016, inside the seed spread.
- **Seed 0 has the most main-network use in every arm**, and seed 2 has the most use in the R = 1 arm only. Seed sets both init and data order, so this does not separate the two.
- Training FLOPs are lower than R = 1 in both Parcae arms (the no-grad prefix has no backward), so the Parcae arms also trained with less compute.

## Verdict

**Not supported.** At step 12,000, edu val bpb is 1.5286, 1.5568, and 1.5730 in the three Parcae R = 2 seeds, all above the best R = 1 seed (1.4135).

## Updated hypothesis

Parcae's stable injection makes a looped main network work in our H-Net: the main output carries document content, and the depth sampler and decay behave as in the paper. It does not pay at matched forward FLOPs on 786M bytes with a 1-stage model. Unique layers are worth more than repeated steps at this budget, or the loop needs the paper's other parts (Muon, its init) to pay. The plan's next step was a matched-parameter test (prelude T2, core T4, coda T2), which spends more FLOPs. I do not queue it now. The Phase 1 question (the 2-stage model against BPE) is open and has a cheaper next test, and the Parcae main can move into the 2-stage model once that is settled. The R = 6 result (the same bpb at 41% fewer parameters) is the lead to keep for Phase 2.

## Related

- [Looped coarse main](../refuted/2026-11-03-looped-coarse-main.md)
- [Windowed growth keeps main](../refuted/2026-11-04-windowed-growth-keeps-main.md)
- [Looped main note](../../../.agents/notes/implemented/feature/2026-11-03-looped-main.md)
