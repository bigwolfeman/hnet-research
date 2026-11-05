# Agent Note: Frozen entropy boundaries (router "given")

Status: implemented

## Problem

Constitution 1.2.0 checklist item 5 asks whether learned boundaries beat boundaries from an entropy-threshold proxy at the same architecture and ratio (the blog's Part II, Table 2; the BLT heuristic). Our model had no way to train with boundaries from outside: every router computed p from the encoder states.

## Decision

- `hnet/boundaries.py`: `EntropyBoundaries` wraps a frozen byte LM (the proxy). For each batch it returns one (B, L) bool mask per stage: byte t starts a chunk at stage s when the proxy's next-byte entropy before t exceeds theta_s. Thresholds rise stage by stage, so the masks nest. `calibrate` picks thresholds from val entropies for target ratios; `scripts/calibrate_entropy.py` runs it on val rows drawn with the train-mix weights.
- `hnet/routers.py`: router `given` (no parameters, start anchors, `read_idx = cumsum - 1` as lookahead). `Router.forward(h, lengths, given)` fails loud when `given` and `needs_given` disagree.
- `HNetLM` builds the source from `model.boundaries` and keeps it as a plain attribute, not a submodule. The proxy is outside `parameters()`, the optimizer and the state_dict. `_apply` moves it with the model, and `compile_stacks` compiles it. `HNetLM.forward` computes the masks and passes `given` down the levels. Each `Stage` reads its own mask at the byte positions of its inputs and passes the rest inward. `Main` and `ParcaeMain` assert that no mask is left over. The config fails loud unless every stage router is `given` exactly when `model.boundaries` is set.
- Everything else is the end-to-end model's: layout, widths, EMA dechunk, STE (conf is 1), LR. `configs/experiment/entropy_2stage.yaml` inherits `reflayout_2stage`, with ratio-loss weight 0 and no RatioController. The ratio is set by the thresholds.
- FLOPs: `given` adds 0 router FLOPs. The proxy is not counted, as in the blog.

## Alternatives considered

- **Precompute boundaries for the training set.** The loader draws random offsets over about 14.5B bytes, and an offline pass costs the same proxy forward, plus storage and a second data path. Online is simpler, and the cost is one proxy forward per step.
- **Pass masks through the trainer, eval and probe calls.** Every caller of `model(x, lengths)` would change: evaluation, probes, semantics, JEPA, scripts. With the source attached to the model, all of them work unchanged, and checkpoints rebuild it from their config.
- **The proxy as a registered submodule with frozen weights.** That puts its weights in every checkpoint and breaks the per-stage optimizer groups, which own every parameter by level.
- **Soft p from the entropy (for example a sigmoid around the threshold).** The blog's control uses hard boundaries. A soft p would make the EMA blend chunks in a way the heuristic does not define.

## Consequences

- Building a model whose config has `model.boundaries` loads the proxy checkpoint. The path must exist wherever such a checkpoint is evaluated.
- A training step costs one extra proxy forward (estimate 15-20% at our sizes; the smoke measures it).
- Tests: `tests/test_boundaries.py` covers calibration rates, the router contract, chunking exactly at the masks (stage 1 inside stage 0), causality, the proxy staying out of the model and its gradients, and loud misconfiguration.
- First use: [learned vs entropy boundaries](../../../../lab/experiments/planned/2026-11-05-learned-vs-entropy-boundaries.md).
