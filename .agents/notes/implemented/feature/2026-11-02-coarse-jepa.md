# Agent Note: Coarse JEPA auxiliary loss

Status: implemented

## Problem

The stage-1 decoder of our 2-stage H-Net ignores the main network's output in most seeds, at 786M bytes and at 3.15B. Next-byte prediction rewards content at the main output only once that content beats what the outer stages compute locally. Balestriero and LeCun (arXiv 2610.39302) argue that next-token prediction is a JEPA with bad views (parts of one document), and that good views (two observations of one thing) reach the same representation alignment with 42x fewer samples. We want a direct, early reason for the main network to carry content, without changing the LM data stream.

## Decision

An auxiliary LeJEPA-form loss (the paper's own recipe) on the pooled main-network output (`lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/jepa.py`, `hnet/data.py` PairLoader, `train.py`, config `train.jepa`):

```
z(x) = proj(mean of Main's output over valid main positions)   (forward hook on the Main leaf; no model code change)
inv  = mean (z_view - pair center)^2 ; sigreg = Epps-Pulley vs N(0, 1) on random 1-D directions (batch statistic)
loss = lambda * sigreg + (1 - lambda) * inv     (lambda 0.02, 32 pairs per step, both views 512 bytes)
total = LM loss + ratio terms + jepa.weight * loss ; view rows add no LM or ratio loss
Main's input is detached in the JEPA pass: the loss trains the main network and the projector only
views: a Go function's code and its doc comment (scripts/goviews/main.go, leading function name removed)
```

- `jepa.weight: 0` (the default in every train config) runs no extra forward and draws no random numbers, so old configs train exactly as before.
- The PairLoader has its own rng (seed + `seed_offset`), so a JEPA arm's LM batches and init equal those of the same seed without JEPA. Existing runs serve as controls.
- The projector joins the main network's LR group (`stage<N>_extra_*`), is clipped with the model, snapshotted in ladder branches, and saved in `resume.pt` with the pair rng.
- SIGReg's random directions come from a generator seeded by the step, so they draw nothing from the global rng.
- `full_cfg.jepa_resolved` logs the pair count, projector size, and the view bytes per step against the LM bytes.
- train.py backprops the JEPA term before the LM forward, so peak memory is the larger pass, not the sum (same gradient; tested).
- Tests: `tests/test_jepa.py` (SIGReg is small for a standard Gaussian batch and large for a collapsed or scaled one, embedding ignores padding, gradient reaches the main network and the projector and nothing outside them, split backward equals the summed loss, the loss depends on the step only, the hook is idle outside the loss, the pair stream resumes exactly).

## Alternatives considered

- **Views by identifier renaming and reformatting** (what I first proposed to the user). Rejected: invariance to renaming asks the main network to drop identifier identity, while copying identifiers is a main use of long-range memory in code. The loss would fight the LM objective. Code and its doc comment share purpose, the closest match to the paper's image and caption.
- **Next-chunk latent prediction** (predict the main output of a later chunk). This is the bad-view case the paper argues against: parts of one document.
- **SimSiam form** (predictor, stop-gradient targets, 2 pairs per step). Built and run first. Rejected on evidence: in all three seeds of [coarse JEPA on dead seeds](../../../../lab/experiments/refuted/2026-11-02-coarse-jepa-on-dead-seeds.md) (amendment 2026-11-02) the embedding collapsed to a constant (same-pair cosine equal to other-pair cosine), and the collapse spread into the LM path (stage 0 stopped reading its inner output; edu bpb 0.09 worse than the controls). I had assumed that the LM loss holds the representation away from collapse. In the dead seeds the decoders give the inner path little LM gradient, so nothing held it.
- **A JEPA gradient into the outer encoders** (the first three runs). Rejected on evidence: the routers read the encoder output and the view rows carry no ratio loss, so the chunk ratios swung (stage 1 up to 1,606 against a target of 3; the control stays at or below 4.9). A ratio loss on the view rows would bound the ratio but still let the JEPA reshape the boundaries the LM depends on.
- **InfoNCE with in-batch negatives.** It also needs a batch, and it is a contrastive special case in the paper's framing. SIGReg is what the paper ran, so its sample-efficiency claim applies to SIGReg.
- **An EMA teacher** (BYOL / I-JEPA). Costs a second copy of the model and its forward, and does not by itself stop collapse at 2 pairs per step.
- **Per-position (dense) targets.** Code and comment have no byte alignment, so dense targets need a matching step. The pooled target needs none.

## Consequences

- The view rows (32 pairs per step, 512 + 512 bytes) add 50% bytes per step at 16 x 4,096 LM bytes (encoder and main only, no decoders), and one extra compiled stage-0 shape (one recompile at step 1).
- Edu text has no natural good views; this loss only sees Go. Views for text are open.
- First use: [coarse JEPA on dead seeds](../../../../lab/experiments/refuted/2026-11-02-coarse-jepa-on-dead-seeds.md), not supported. At weight 1.0 with Go code/doc views, every form of this loss (SimSiam, LeJEPA, detached) cuts the hierarchy out of the LM in dead seeds: stage-1 probes at 1e-5 and the stage-0 shuffle cost at 0.000 from step 2,000, edu bpb 0.12 to 0.16 worse. The views never align at this scale (retrieval at chance). The default stays `weight: 0.0`. Any new use must first show view alignment in a smoke and gate on the shuffle costs staying at or above the control's.
