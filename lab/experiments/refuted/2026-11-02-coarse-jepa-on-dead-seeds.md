# Experiment: coarse JEPA on dead seeds

Status: refuted

Date proposed: 2026-11-02
Algorithm: coarse JEPA, `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/jepa.py`; view pairs, `scripts/goviews/main.go` (pseudocode in both)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does an auxiliary JEPA loss on good views at the main network's output bring stage 1 online at 786M bytes, in seeds where it stayed dead without the loss?

Context: at 786M bytes (12,000 steps), our code with the reference configuration ends with the stage-1 signature below zero in seeds 1, 3, and 4 (−0.127, −0.117, −0.034), and the stage-0 signature too (−0.304, −0.329, −0.198) ([main-network seed rate](2026-11-02-main-network-seed-rate.md), [faithful init and RoPE](2026-11-02-faithful-init-and-rope.md)). Stage 0 comes online later with more bytes, stage 1 mostly not by 3.15B ([dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md)). Balestriero and LeCun (arXiv 2610.39302) argue that next-token prediction is a JEPA with bad views (two parts of one document), and that good views (two observations of one thing) reach the same representation alignment with 42x fewer samples.

## Hypothesis

The main network's output is ignored early because next-byte prediction gives it a weak, slow reason to carry content. A loss that asks the pooled main output of a Go function to predict that of its doc comment (and the reverse) gives it that reason directly, from the first step. With content worth reading, the stage-1 decoder reads it within 786M bytes.

## Predictions

Main prediction (only this one answers the Question):

- With the JEPA loss (weight 1.0), the stage-1 shuffle − zero is above zero at step 12,000 in all three seeds, 1, 3, and 4. Each seed's control (the same seed, init, and LM data stream, no JEPA loss) ended below zero. All three above zero: supported. Any at or below zero: not supported.

Other predictions (at most two):

- The stage-0 shuffle − zero is above zero at step 12,000 in all three seeds (controls −0.304, −0.329, −0.198).
- Edu val bpb at step 12,000 is at or below each seed's control (1.3536, 1.3522, 1.3573).

If supported, we do: good views bring the main network online with less data. Next, test it on fresh seeds against fresh controls, at matched total FLOPs (the view rows add up to 7.8%), and with views for edu text. Then put it on the BPE ladder.

If not supported, we do: content richness at the main output is not what keeps stage 1 dead at this budget, or this loss does not create it. Read the JEPA logs (prediction cosine, same-pair against other-pair cosine) to tell which, and weigh the result with [hierarchy on synthetic recall](2026-11-02-hierarchy-on-synthetic-recall.md), which tests whether stage 1 wakes when content pays a lot.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), including `tests/test_jepa.py` (pooled embedding ignores padding, the loss reaches the main network, stage 0, and the predictor, the pair stream resumes exactly).
- Edu ratios within 15% of 3 x 3 at the end; edu fwd FLOPs/byte of the LM rows within 5% of 22.27M. LM training bytes 786,432,000. Finite loss.
- No collapse: at the end, `jepa/z_cos_same_pair` is above `jepa/z_cos_other_pair` (averaged over the last 1,000 steps). If not, the JEPA loss reached a constant embedding and the arm did not test the hypothesis.

## Method

```
pairs (scripts/goviews/main.go on code/go_train.bin, the Go train split only; GPU idle while it runs):
    for each Go file that parses, each function with a doc comment:
        view a = source from `func` to the end of the body (no doc comment), 200..2,048 bytes
        view b = doc comment text, leading function name removed, whitespace collapsed, 40..512 bytes
    -> $HNET_DATA_DIR/code/goviews/{code,doc}.bin
JEPA (hnet/jepa.py, SimSiam form): z = mean of the main network's output over valid main positions (forward hook)
    loss = 1/2 (1 - cos(pred(z_a), sg(z_b))) + 1/2 (1 - cos(pred(z_b), sg(z_a)))
    pred = Linear(512, 2048) -> LayerNorm -> GELU -> Linear(2048, 512), in the main network's LR group (x0.72)
    each step: the usual LM batch (16 x 4,096 bytes, unchanged) + 2 random pairs, cut to 2,048 (code) and 512 (doc)
    bytes, one extra forward of 4 rows x 2,048 bytes that stops after Main (no decoders, no LM head)
    total loss = LM loss + ratio terms + 1.0 * JEPA loss ; the view rows add no LM or ratio loss
    PairLoader has its own rng (seed + 1,000,003): the LM batches and the init are those of the control runs
arms: reflayout_2stage, train.jepa.weight=1.0, seeds 1, 3, 4, 12,000 steps (as the controls)
controls (not rerun; same model code, same LM stream):
    seed 1  wandb yh1nptqe   (faithful init and RoPE)
    seed 3  wandb hd9kqj1n   (main-network seed rate)
    seed 4  wandb 21qy39rj   (main-network seed rate)
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 12,000
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`: `TRAIN=ladder scripts/run_arms.sh ignored/runs/coarse-jepa-on-dead-seeds coarse-jepa-on-dead-seeds reflayout_2stage:1 reflayout_2stage:3 reflayout_2stage:4 -- train.max_steps=12000 "train.ladder_fracs=[]" train.jepa.weight=1.0 train.jepa.pairs_dir=code/goviews`
- wandb project `hnet-router`, group `coarse-jepa-on-dead-seeds`.
- Cost: smoke runs at the 500 W GPU cap: 134 ms/step against 110 for the same arm without the JEPA loss (+22%; host syncs of the extra forward, not FLOPs). About 29 min per arm, about 90 min in total.
- GPU power cap 500 W (set by the user at 16:35, power budget). The controls ran uncapped. Speed is a measurement, not a gate.
- The JEPA weight (1.0) is not tuned. A loss in [0, 2] at weight 1 is the same order as the LM loss (about 0.9 nats), the "strong" end. If the main prediction fails with healthy JEPA logs, a weight sweep does not rescue the hypothesis; if it fails with a collapsed or flat JEPA loss, the gate already marks the run broken.
- The JEPA arms see up to 7.8% more bytes than the controls (the view rows, before padding). A bpb gain can come partly from that; the stage-1 sign is the graded quantity.

### Amendment 2026-11-02 19:05: first run broken (collapse gate failed), loss changed to the paper's LeJEPA form

The first run (SimSiam form above, 2 pairs per step) finished 12,000 steps in all three seeds (wandb `chl6kcwn`, `wd8rtp1u`, `vq3a9gey`). The no-collapse gate failed in all three: over the last 1,000 steps `jepa/z_cos_same_pair` equaled `jepa/z_cos_other_pair` to 4 decimals (seed 1: −0.607 and −0.607; seed 3: −0.480 and −0.480; seed 4: 1.000 and 1.000). The JEPA loss reached a constant embedding (seed 3 loss 0.0001, seed 4 0.0000), so the run did not test the hypothesis, and nothing is graded from it. Artifacts: `results/2026-11-02-coarse-jepa-on-dead-seeds/simsiam-run-broken-*` (probe curves, JEPA log tails; `pull.py`, `jepa_tail.py` made them). Observations from the broken run, kept because they inform the rerun:

- The collapse spread into the LM path. Shuffling stage 0's inner output cost 0.0004 / 0.0176 / 0.0000 bpb at 12,000 (the stage-0 decoder ignored its whole inner path), and shuffling the main output cost 0.0000. Edu val bpb 1.4444 / 1.4329 / 1.4462, against the controls' 1.3536 / 1.3522 / 1.3573.
- The design assumption "the LM loss holds the representation away from collapse" is false in exactly these seeds: their decoders already give the inner path little LM gradient.
- At step 1 the pooled embeddings of different pairs were already nearly as alike as those of the same pair (seed 1: 0.669 and 0.661), so the predictor could reach a low loss with a near-constant output.

Changed method (predictions, arms, controls, budget, and the collapse gate unchanged):

```
JEPA (hnet/jepa.py, LeJEPA form as in arXiv 2610.39302: no predictor, no stop-gradient):
    z = proj(mean of the main output over valid main positions), proj = Linear(512, 2048) -> LayerNorm -> GELU
        -> Linear(2048, 128), in the main network's LR group (x0.72)
    inv    = mean over pairs and dims of (z_view - pair center)^2
    sigreg = Epps-Pulley statistic against N(0, 1) on 256 random unit directions (drawn from the step), per view
    loss   = 0.02 * sigreg + 0.98 * inv        (lambda 0.02 as in the paper)
    each step: 32 random pairs, both views cut to 512 bytes, one extra forward of 64 rows x 512 bytes that stops
    after Main
```

- Why 32 pairs: SIGReg is a batch statistic, and the paper trains with batches of 256. Two pairs cannot show a batch to be degenerate. 32 is the most the step-time cost allows (measured in the rerun smoke; recorded below the launch).
- Why both views at 512 bytes: 32 code rows at 2,048 bytes would be as many bytes as the LM batch. 512 bytes keeps the signature, the receiver, and the first statements of the body. View bytes per step: 32,768, 50% of the LM bytes (up from 7.8%), so a bpb gain can come partly from the extra encoder updates. The graded quantity stays the stage-1 sign.
- New logs: `jepa/inv`, `jepa/sigreg`, `jepa/top1_doc_given_code` (retrieval of the right doc among 32 in the batch; chance 1/32). The gate stays same-pair cosine above other-pair cosine, now averaged over the 31 other pairs per row.
- Launch: the same command, in a new log dir `ignored/runs/coarse-jepa-on-dead-seeds-lejepa`, wandb group `coarse-jepa-on-dead-seeds-lejepa`.

### Amendment 2026-11-02 20:35: LeJEPA rerun stopped at step 2,100; lambda scaled for the batch; retrieval gate added

The LeJEPA rerun (seed 1, wandb `liw70jg2`, group `coarse-jepa-on-dead-seeds-lejepa`) was stopped by me at about step 2,100 of 12,000. Its SIGReg term stayed at 11 to 13 for 2,000 steps. A batch with no spread along a direction scores about 0.409 x N = 13.1 at N = 32, so the embedding had collapsed. Retrieval of the right doc among 32 stayed at chance (0.00 to 0.06; chance 0.031). Cause: the Epps-Pulley statistic is multiplied by the batch size, so the paper's lambda 0.02 at batch 256 is 8x weaker at 32 pairs. The gate would have failed at the end, so the arm would not have tested the hypothesis.

Calibration, before any new arm (wandb project `hnet-smoke`, group `lejepa-lambda-smoke`, seed 1, 800 steps; not graded, not arms):

| lambda | run | SIGReg, steps 0-199 → 800 | invariance | top-1 retrieval, 0-199 → 800 |
|---|---|---:|---:|---:|
| 0.16 | `hnvrnr5j` | 6.3 → 2.2 | 0.17 → 0.32 | 0.109 → 0.062 |
| 0.5 | `jhq6zzxv` | 6.2 → 2.2 | 0.20 → 0.43 | 0.117 → 0.031 |

- lambda 0.16 (= 0.02 x 256 / 32, the paper's strength per item) keeps the embedding spread. Over 800 steps the two views drift apart (invariance up), and retrieval falls toward chance. Whether alignment comes later is unknown.
- Changed method: `sigreg_lambda` 0.16 (the default in `configs/train/*.yaml`). Everything else as in the amendment above.
- Added gate (a check that the views were aligned at all, set before the rerun's data): top-1 retrieval averaged over the last 1,000 steps is above chance (1/32). If not, the JEPA loss acted only as a spreading regularizer, the arm did not test the hypothesis, and the verdict is can't tell.
- Order: this rerun runs after [main network sole path](2026-11-02-main-network-sole-path.md). That test asks whether our main path can carry content at all, which this hypothesis assumes. Log dir `ignored/runs/coarse-jepa-on-dead-seeds-lejepa16`, wandb group `coarse-jepa-on-dead-seeds-lejepa16`.

### Amendment 2026-11-02 21:25: lambda-0.16 rerun, seed 1 out of GPU memory; JEPA backward moved before the LM forward

Seed 1 (wandb `b1jzwt01`) ran out of GPU memory at about step 1,000, in the JEPA forward (27.8 GB in use; a launcher held 1.1 GB). I stopped seed 3 (`1tohdxec`, a few minutes in, old code). Change, no effect on the gradient: train.py now backprops the JEPA term before the LM forward, so the view activations are freed first (`tests/test_jepa.py::test_separate_backward_equals_summed_loss`). Peak memory in a 300-step smoke: 15.7 GB, and 140 ms/step. All three seeds restart from step 0 in the same log dir and wandb group.

### Amendment 2026-11-02 21:40: rerun stopped again, router ratios unstable; the JEPA gradient now stops at Main's input

The restarted seed 1 (wandb `wqol9jx2`) was stopped by me at about step 2,300. Its training chunk ratios swung far from 3 x 3: stage 1 up to 1,606 (13 evals above 6 in the first 2,300 steps), stage 0 up to 14.1. The control (`yh1nptqe`, no JEPA) stays at or below 4.9 over the same steps. Every JEPA run so far spiked (SimSiam `chl6kcwn` stage 1 up to 2,080; LeJEPA lambda 0.02 `liw70jg2` up to 17.7). Cause: the JEPA gradient reshapes the outer encoders, the routers read the encoder output, and the view rows carry no ratio loss.

Changed method: the JEPA pass detaches Main's input (a forward pre-hook in `hnet/jepa.py`), so the JEPA loss trains the main network and the projector only. The encoders and routers learn from the LM loss alone. This is closer to the hypothesis, which is about the content at the main output. `tests/test_jepa.py` checks that no parameter outside Main and the projector gets a JEPA gradient.

Smoke (wandb `hnet-smoke` / `lejepa-detach-smoke`, `j1vk28xt`, seed 1, 1,000 steps): training ratios at most 4.58 (stage 0) and 6.49 (stage 1, at step 50 only); peak memory 17.7 GB; 134 to 156 ms/step. Top-1 retrieval 0.008 to 0.10 (chance 0.031), so the retrieval gate may fail; the gate decides, not this smoke.

Order: [main path recall budget](../successes/2026-11-02-main-path-recall-budget.md) runs first, then this rerun. Log dir `ignored/runs/coarse-jepa-on-dead-seeds-detach`, wandb group `coarse-jepa-on-dead-seeds-detach`.

## Results

The detached rerun ran all three seeds to step 12,000 (2026-11-02 23:59 to 2026-11-03 01:16; commit f2095fc, suite 164 passed, 1 skipped). Log dir `ignored/runs/coarse-jepa-on-dead-seeds-detach`. wandb `hnet-router`, group `coarse-jepa-on-dead-seeds-detach`: seeds 1, 3, 4 `i581xzb1`, `xdktm0nr`, `191clzls`. Artifacts: [results/](../results/2026-11-02-coarse-jepa-on-dead-seeds/), files `detach-run-*` (probe curves from `pull.py`, JEPA logs from `jepa_tail.py` and `jepa_curve.py`), `stage1-probe-raw.txt` (`s1raw.py`: stage-1 probes of every JEPA run and the controls), `stage-interface-diag.txt` (`s1diag.py`, see below).

### Gates

| gate | seed 1 | seed 3 | seed 4 |
|---|---|---|---|
| ratios at the end (3 x 3 ±15%) | 3.01 x 2.83 pass | 3.05 x 2.63 pass | 2.96 x 3.10 pass |
| LM fwd FLOPs/byte within 5% of 22.27M | 23.00M (+3.3%) pass | 23.48M (+5.4%) **fail** | 22.49M (+1.0%) pass |
| 786,432,000 LM bytes; finite loss | pass | pass | pass |
| no collapse: same-pair cosine above other-pair, last 1,000 steps | 0.197 > 0.193 pass | 0.042 < 0.060 **fail** | 0.015 > 0.001 pass |
| retrieval: top-1 above 1/32 (0.031), last 1,000 steps | 0.034 pass | 0.030 **fail** | 0.044 pass |

Seed 3 is a broken arm, and nothing is graded from it. Seeds 1 and 4 pass every gate. Their retrieval passes by a small margin: each value averages 20 logged steps of a 32-way top-1, so its noise is about 0.007, and seed 1 is 0.5 of that above chance. The views barely aligned in any seed.

Mid-run, the stage-1 val ratio swung in seed 1 (1,375 at step 8,000, 1.00 at step 10,000) although the JEPA gradient no longer reaches the routers.

### Probes at step 12,000

| | seed 1 | seed 3 (broken) | seed 4 | controls (seeds 1, 3, 4) |
|---|---|---|---|---|
| stage-1 `zero` / `shuffle` | 1.0e-5 / 2.4e-5 | -1.9e-5 / 6.9e-6 | -4.1e-5 / 8.1e-6 | 0.135 / 0.008, 0.161 / 0.043, 0.080 / 0.046 |
| stage-1 shuffle − zero | 0.0000 | 0.0000 | 0.0000 | −0.127, −0.118, −0.034 |
| stage-0 `zero` / `shuffle` | 4.029 / 0.000 | 3.919 / 0.000 | 2.504 / 0.000 | 0.846 / 0.542, 1.083 / 0.754, 0.815 / 0.617 |
| edu val bpb | 1.4836 | 1.5157 | 1.4792 | 1.3536, 1.3522, 1.3573 |

- Main prediction (stage-1 shuffle − zero above zero in all three seeds): 0.0000 in seeds 1 and 4, the two valid arms. **Not supported.** Seed 3 cannot change this: the rule needs all three above zero.
- Other prediction 1 (stage-0 shuffle − zero above zero): does not hold (−4.03, −2.50 in the valid arms).
- Other prediction 2 (edu bpb at or below the control): does not hold (+0.130, +0.122).

### What the runs show (observations after the fact)

- **The JEPA loss cut the hierarchy out of the LM.** From the first eval (step 2,000) on, ablating stage 1's inner path changes bpb by less than 1e-4 in every seed, against 0.008 to 0.16 in the controls. Shuffling stage 0's whole inner path costs 0.000 bpb, against 0.54 to 0.75 in the controls. Zeroing it costs 2.5 to 4.0 bpb: the stage-0 decoder reads only a constant from the hierarchy below it. The model is a byte model with one outer stage, and it is 0.12 to 0.16 bpb worse than the controls.
- **The same happened in every JEPA variant.** SimSiam (`chl6kcwn`, no detach), LeJEPA at lambda 0.02 (`liw70jg2`) and 0.16 (`wqol9jx2`), and the detached form all show stage-1 probes at 1e-5 from step 2,000 (`stage1-probe-raw.txt`). The broken SimSiam run already showed the stage-0 cut (see the 19:05 amendment).
- **The inner signal is present, and the decoder ignores it.** In the seed-1 final checkpoint (`s1diag.py`, one val batch), the stage-1 inner output adds a vector of rms 0.71 to a residual of rms 4.94 at the stage-1 decoder input. In the control the two are 1.19 and 2.01. The residual grew 2.5x, and the decoder learned weights that give the inner part no effect. This is a learned shut-off, not a missing signal.
- **A likely mechanism.** In these seeds the LM loss sends little gradient through the main network (that is why they are dead seeds). At weight 1.0 the JEPA gradient dominates the main network's updates, and its output changes in ways that are noise for the next-byte loss. The decoders gain by shutting the noise out, and once it is shut out the LM sends no gradient back through it. Nothing pulls the path back open. Detaching Main's input removed the router damage, but not this one.
- **The views never aligned.** Top-1 retrieval started at 0.08 (step 0 to 200, from the random init) and fell to chance by step 6,000 in every seed. A 40M byte model at this budget does not map a Go function and its doc comment to nearby points. The loss acted as a spreading regularizer on the main output.

## Verdict

**Not supported.** With the coarse JEPA loss at weight 1.0, the stage-1 shuffle − zero is 0.0000 at step 12,000 in both valid seeds (1 and 4). Seed 3 failed three gates and is not graded. The loss did not bring stage 1 online. In every seed it cut the whole hierarchy below stage 0 out of the LM, and bpb was 0.12 to 0.16 worse than the controls.

## Updated hypothesis

The main output in a dead seed is not empty of content waiting for a reason to fill. It is a path the decoders read little. A second loss that writes into it, at a strength comparable to the LM loss, makes it worse to read, and the decoders close it. This matches [main path recall budget](../successes/2026-11-02-main-path-recall-budget.md): there, content worth 0.985 bpb sat behind the main network for 24,000 steps, and the decoders took 4% of it. Both results put the bottleneck on the read side (the decoders and the chunk interface), not on what the main network holds.

A dual-view loss is still a fair test of the data-efficiency idea in arXiv 2610.39302, but this test did not reach it. The views never aligned, and the loss competed with the LM for one output. A test that could answer it needs all three:

1. Views that the model aligns from early on: retrieval well above chance through the run (for example, two crops of one document, or a known paraphrase pair), checked in a smoke before any arm.
2. A JEPA strength that keeps the path read: the stage-0 and stage-1 shuffle costs at or above the control's at every eval, as a gate.
3. The read side first. The next planned work measures the read path in the synthetic checkpoints: the LM gradient that reaches the main network against the outer stages, and the per-word recall gain by where a word starts relative to the stage-1 boundary. A JEPA rerun waits for that result.

## Related

- [Main-network seed rate](2026-11-02-main-network-seed-rate.md)
- [Dead seeds, longer budget](../successes/2026-11-02-dead-seeds-longer-budget.md)
- [Hierarchy on synthetic recall](2026-11-02-hierarchy-on-synthetic-recall.md)
- [Main path recall budget](../successes/2026-11-02-main-path-recall-budget.md)
