# Experiment: main gradient coherence

Status: refuted

Date proposed: 2026-11-03
Algorithm: per-level gradient statistics, `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/gradstats.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

In seeds where stage 1 ends dead, is the learning signal that reaches the main network less consistent than in seeds where it ends live, early in training?

Context: at 786M bytes (12,000 steps) the stage-1 fate is a lottery in both codes ([main-network seed rate](2026-11-02-main-network-seed-rate.md)): our code is live in seeds 0 and 2 and dead in 1, 3, 4; the reference code is live in 0, 1, 3 and dead in 2, 4. The stage-1 sign at step 2,000 does not show the fate. On synthetic recall, the route through the main network opens slowly and per run ([main path recall budget](../successes/2026-11-02-main-path-recall-budget.md)), and a JEPA loss on the main output made the decoders close it ([coarse JEPA on dead seeds](2026-11-02-coarse-jepa-on-dead-seeds.md)). Two readings remain: the main network gets a weak or noisy signal (optimization), or it gets a usable signal and the decoders do not read what it learns (the read side).

## Hypothesis

AdamW hides the gradient scale but not its consistency. If the main network's gradient is mostly noise in a seed, its updates random-walk, its output carries nothing the decoders can use, and the decoders learn to ignore it. That seed ends dead. In a live seed, the main network gets a consistent signal early, before the stage-1 probe can see it.

## Predictions

Main prediction (only this one answers the Question):

- In our code, the main network's AdamW update rms (`train/grad/update_rms_l2`), averaged over steps 500 to 4,000, is lower in every seed that ends dead than in every seed that ends live. Fate is read from each rerun's own stage-1 shuffle − zero at step 12,000 (above zero: live). Every dead seed below every live seed: supported. Any dead seed at or above any live seed: not supported. If the reruns end with all seeds live or all dead, can't tell.

Other predictions (at most two):

- The same ordering holds in the reference code (seeds 1, 2, 3, 4).
- On synthetic recall, in the sole-path layout, the seed that builds no route through the main network (seed 2 in [main network sole path](2026-11-02-main-network-sole-path.md)) has a lower main update rms over steps 500 to 3,000 than the seeds that build one (0 and 1).

If supported, we do: the main network's signal is the bottleneck. Find what makes it noisy in a dead seed. Then test the cheapest lever on the main group's optimizer, the LR multiplier (0.72) first, against the same seeds.

If not supported, we do: the signal into the main network is as consistent in dead seeds, so the decoders fail to read a main network that learns. Run the read-side test next: the per-word recall gain by where a word starts relative to the stage-1 boundary, on the synthetic checkpoints. Then try a change on the read side.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including `tests/test_gradstats.py` (update rms reads 1 for a constant gradient and the noise floor for pure noise). Reference venv `test_refmodel.py` passes.
- Edu ratios within 15% of 3 x 3 at the end; edu fwd FLOPs/byte within 5% of 22.27M; 786,432,000 bytes per edu arm; finite loss.
- `train/grad/*` is logged at every `log_every` step, for levels 0, 1, 2, in every arm.
- The statistics are read-only: `tests/test_gradstats.py::test_stats_do_not_change_training` trains a small model on CPU for 20 AdamW steps with and without the statistics and asserts bit-equal parameters.

## Method

```
stats (hnet/gradstats.py), every log_every = 50 steps, read-only:
    levels from the optimizer groups: 0 = stage 0 + embeddings + head, 1 = stage 1, 2 = main network
    grad/norm_l{L}       = ||grad||_2 over level L, before clipping
    grad/update_rms_l{L} = rms of m_hat / (sqrt(v_hat) + eps) over level L, after the AdamW step
        (near 1: the gradient keeps its sign; low: it is zero-mean noise, or its scale is falling)
edu arms (12,000 steps, as the seed-rate runs):
    ours       reflayout_2stage  seeds 0, 1, 2, 3, 4   (spike venv)
    reference  ref_hnet_2stage   seeds 1, 2, 3, 4      (reference venv, train.ratio_integral_lr=0.002)
synthetic arms (6,000 steps, as main network sole path, recall data only):
    reflayout_2stage, layouts t4 / t4 / T1t4 / t4T1, windows 4 and 1, seeds 0, 1, 2
grade: mean of grad/update_rms_l2 over logged steps 500..4,000, per edu arm of our code, ordered against the arm's fate at 12,000
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, after the suite (queue in `ignored/runs/main-gradient-coherence/chain.sh`):
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/main-gradient-coherence/ours main-gradient-coherence reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 reflayout_2stage:3 reflayout_2stage:4 -- train.max_steps=12000 "train.ladder_fracs=[]"`
  - `PY=../2026-11-01-reference-hnet-probe/.venv/bin/python TRAIN=ladder scripts/run_arms.sh ignored/runs/main-gradient-coherence/ref main-gradient-coherence ref_hnet_2stage:1 ref_hnet_2stage:2 ref_hnet_2stage:3 ref_hnet_2stage:4 -- train.max_steps=12000 "train.ladder_fracs=[]" train.ratio_integral_lr=0.002`
  - `TRAIN=ladder scripts/run_arms.sh ignored/runs/main-gradient-coherence/synth main-gradient-coherence reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.0.enc_layout=t4 model.stages.0.dec_layout=t4 model.stages.0.window=4 model.stages.1.enc_layout=T1t4 model.stages.1.dec_layout=t4T1 model.stages.1.window=1 train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000`
- wandb project `hnet-router`, group `main-gradient-coherence`.
- Fate comes from the reruns, not from the earlier runs. The trainer is not bit-repeatable on the GPU: two 200-step smokes with the same seed and config (`train.grad_stats=false`, wandb `hnet-smoke` / `gradstats-smoke`) end at val bpb 2.1775 and 2.1799, and one with `true` at 2.1750 (`fttu62mf`). So a seed does not fix the stage-1 fate, and a rerun may change it. The earlier fates are the expectation, not the label.
- The noise floor sqrt((1 − b1)/(1 + b1)) = 0.229 holds only for a gradient whose scale is constant. Early in training the scale falls, and v remembers larger past gradients, so values below 0.229 are expected (the smoke reads 0.07 to 0.14 for the main network over steps 50 to 200). The prediction is an ordering across seeds, not a comparison with the floor.
- Cost at the 500 W cap: about 25 min per edu arm (ours), about 20 min (reference), 7 min per synthetic arm. About 3 h 50 min in total.

## Results

All 12 arms ran to their step counts (2026-11-03 09:20 to 12:52, commit 717f57e). Log dirs `ignored/runs/main-gradient-coherence/{ours,ref,synth}`. wandb group `main-gradient-coherence`: ours `akqhdkkz`, `ipcxldpl`, `zplsyk1o`, `5d3o336d`, `7dffta9c` (seeds 0 to 4); reference `w96mr9va`, `26llvly2`, `lo4o7up8`, `zhoalme6` (seeds 1 to 4); synthetic `6779e4vx`, `wrv1ej2g`, `w5rvq0rb` (seeds 0 to 2). Artifacts: [results/](../results/2026-11-03-main-gradient-coherence/) (`pull.py`, `summary.txt`, `grad-curves.csv` with every logged step of the six grad statistics).

### Gates

| gate | result |
|---|---|
| suite (`HNET_TEST_CUDA=1`) and reference `test_refmodel.py` | pass: 168 passed, 1 skipped; reference 7 passed |
| edu ratios within 15% of 3 x 3 | pass: 3.01 to 3.05 and 2.87 to 3.01 |
| edu fwd FLOPs/byte within 5% of 22.27M | pass: 22.36M to 22.81M (at most +2.4%) |
| 786,432,000 bytes per edu arm (393,216,000 per synthetic arm) | pass |
| `train/grad/*` at every `log_every` step, levels 0 to 2 | pass: 241 rows per edu arm, 121 per synthetic arm |
| finite loss | pass |
| statistics read-only (`test_stats_do_not_change_training`) | pass |

### Main update rms against fate (edu, 12,000 steps)

Update rms = mean of `train/grad/update_rms_l2` over logged steps 500 to 4,000 (71 rows). Fate = `coarse_shuffle_s1 − coarse_zero_s1` at 12,000.

| code | seed | fate | state | main update rms | `coarse_shuffle_s1` | main grad norm (same window) |
|---|---:|---:|---|---:|---:|---:|
| ours | 0 | +0.023 | live | 0.1874 | 0.158 | 0.0087 |
| ours | 1 | −0.056 | dead | **0.1031** | 0.0005 | 0.0041 |
| ours | 2 | −0.023 | dead | **0.1934** | 0.131 | 0.0077 |
| ours | 3 | −0.035 | dead | 0.1698 | 0.079 | 0.0070 |
| ours | 4 | −0.017 | dead | 0.1865 | 0.049 | 0.0060 |
| reference | 1 | −0.052 | dead | 0.1681 | 0.055 | 0.0065 |
| reference | 2 | −0.011 | dead | 0.1923 | 0.134 | 0.0080 |
| reference | 3 | +0.027 | live | 0.2022 | 0.194 | 0.0106 |
| reference | 4 | −0.023 | dead | 0.1896 | 0.095 | 0.0075 |

- Main prediction (ours: every dead seed below every live seed): seed 2 (dead, 0.1934) is above seed 0 (live, 0.1874). Does not hold.
- Other prediction 1 (the same ordering in the reference code): holds. The one live seed (3, 0.2022) is above all three dead seeds.
- Other prediction 2 (synthetic: the seed with no route has the lower main update rms over 500 to 3,000): does not hold. In these reruns only seed 0 builds the route (`coarse_shuffle_s1` 2.768; seeds 1 and 2 at 0.218 and 0.271, under the no-recall control's 0.415 from [main network sole path](2026-11-02-main-network-sole-path.md)). The update rms is 0.2163 (seed 0), 0.2169, and 0.2188: the seeds without the route are not lower. Val bpb at 6,000: 1.9936, 2.0118, 2.0129 (no-memory anchor 2.006).

Fates changed in two of nine reruns, as the Method allowed: our seed 2 went from +0.004 in [main-network seed rate](2026-11-02-main-network-seed-rate.md) to −0.023, and reference seed 1 from +0.032 ([difference ablation](2026-11-01-reference-hnet-difference-ablation.md)) to −0.052. The other seven kept their sign. In each code, one seed of the rerun is live.

### What the runs show (observations after the fact)

- **The main gradient follows the read; it does not lead it.** Over the 9 edu arms, the main update rms ranks with `coarse_shuffle_s1` (Spearman 0.83; 0.76 without our seed 1) and with fate (0.78). That is post hoc and does not grade anything. But the order is not there early: at step 500 the update rms is 0.140 to 0.191 in our code, with live seed 0 second lowest (0.146). A decoder that reads the main output sends the main network a steady gradient; one that ignores it sends noise. The statistic measures the read, from the main side.
- **Our seed 1 is a full shut-off.** Its decoders stopped reading the main output (`coarse_shuffle_s1` 0.0005; stage-0 shuffle 0.43, against 0.70 to 1.11 in the others). From step 2,000 its main update rms falls to 0.06 to 0.08 and its main grad norm to about 0.002, and it stays there to 12,000. At step 1,000 it is not yet apart (0.155, against 0.143 to 0.188). This is the same learned shut-off as the JEPA dead seeds in [coarse JEPA on dead seeds](2026-11-02-coarse-jepa-on-dead-seeds.md), without a JEPA loss.
- **The other dead seeds look like the live ones.** Update rms 0.17 to 0.19 against 0.187 and 0.202; the main network gets a signal of the same consistency. In those seeds the decoders use the main output (zero cost 0.07 to 0.15 bits) but mostly as a document-independent input: shuffling costs less than zeroing.
- Level 1 (stage 1) and level 0 do not separate the seeds either: stage-1 update rms 0.184 to 0.210, level 0 0.210 to 0.213.

## Verdict

**Not supported.** In our code, dead seed 2 has a higher main update rms over steps 500 to 4,000 (0.1934) than live seed 0 (0.1874). The main network's learning signal is not less consistent early in seeds that end dead.

## Updated hypothesis

The optimization reading is out as a cause. The main network gets a signal as consistent in most dead seeds as in live ones, and the update rms tracks how much the decoders read the main output, after the fact. What decides the fate is on the read side: whether the decoders come to depend on document-specific main content. With the plain chunk interface, the stage-1 decoder sees the main output only as one vector for its own chunk, and on synthetic recall it reads it where memory is worthless ([recall gain by boundary](2026-11-03-recall-gain-by-boundary.md)).

Next, as the plan says for this case: a change on the read side. [Decoder reads main by content](2026-11-03-decoder-reads-main-by-content.md) gives the stage-1 decoder an attention read over past main outputs, on synthetic recall, against the three synthetic arms of this run as the plain control. The LR-multiplier lever for the main group is dropped: its premise (a noisy main signal) did not hold.

## Related

- [Main-network seed rate](2026-11-02-main-network-seed-rate.md)
- [Main path recall budget](../successes/2026-11-02-main-path-recall-budget.md)
- [Coarse JEPA on dead seeds](2026-11-02-coarse-jepa-on-dead-seeds.md)
- [Main network sole path](2026-11-02-main-network-sole-path.md)
