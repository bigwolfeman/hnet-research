# Agent Note: Reference layout is the 2-stage default

Status: implemented

## Problem

`configs/config.yaml` still defaulted to `eoc_2stage`, the first smoke design (end-of-chunk router, 2-layer stages, ratio 4). Every result since 2026-11-02 uses `reflayout_2stage`, and Phase 2 needs one 2-stage base to build on. [Reference layout seeds vs BPE](../../../../lab/experiments/successes/2026-11-04-reference-layout-seeds-vs-bpe.md) is supported: three H-Net seeds (1.0905, 1.0979, 1.0913) are below both BPE seeds (1.1109, 1.1156) on edu at 12.58B bytes, and below them on Go and SQL too. That plan said to make the reference configuration the default if it held.

## Decision

- The spike's `configs/config.yaml` defaults to `experiment: reflayout_2stage` (Mamba-2 outer stages, SWA at stage 1, a 10-layer Transformer main, widths 256 | 256 | 512, lookahead routers at 3 x 3, the paper LR multipliers). `python train.py` with no experiment override trains it.
- The golden tests keep pinning `eoc_2stage`. `tests/golden_util.py:smoke_cfg` names the experiment, so the golden values do not follow the default.
- New experiments and Phase 2 arms derive from `reflayout_2stage` (as `entropy_2stage` already does through Hydra defaults).

## Alternatives considered

- **Leave the default and pass `experiment=reflayout_2stage` everywhere.** Every queue already does this, so nothing breaks. But a bare run trains a design we no longer use, and a new session reads the default as the current design.
- **Regenerate the golden values on reflayout_2stage.** The goldens pin the math of the shared modules, not a design choice. Regenerating them now would drop a check that has held across the packing and compile changes, for no gain.

## Consequences

- A run with no experiment override builds the 42M-param reference model, not the small eoc smoke model.
- `eoc_2stage` and the other experiment configs stay, for reruns of the experiments that used them.
