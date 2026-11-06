# Agent Note: Surprise router (lookahead plus the encoder's own next-byte entropy)

Status: implemented

## Problem

Constitution checklist item 7 (in-context tokenization) is not seen with our learned router. A made-up word keeps the same number of stage-0 chunks each time it repeats (drops 0.05, 0.05, 0.00 over three seeds). Frozen entropy-proxy boundaries do show it, 6.85 to 1.80 chunks ([learned vs entropy boundaries](../../../../lab/experiments/successes/2026-11-05-learned-vs-entropy-boundaries.md)), but they lose 0.04 bpb to the learned router. We want both: the context signal of the entropy rule and end-to-end training.

## Decision

- `hnet/routers.py`: router `surprise`, a subclass of `lookahead`, for stage 0 only.
  - An aux head `W` (d x 256, no bias) predicts the next byte from each encoder state. Its CE (weight `lm_weight`) is the router's `aux_loss`. It trains `W` and the encoder.
  - The entropy of the head's prediction for byte t, computed at t-1, is detached and standardized by running mean and variance. The running stats are BatchNorm-style buffers, updated in training only, so the term at byte t never reads other positions of its batch.
  - z_t = logit(clamp(p^lookahead_t)) + beta (Hz_t - tau), p_t = sigmoid(z_t), p_0 = 1. beta and tau are learned.
- `Router.forward` takes `nxt` (the next byte per position, -100 where there is none) and fails loud when `needs_next` and `nxt` disagree. `HNetLM` builds `nxt` from the input bytes, passes it to stage 0 only, and refuses a `needs_next` router at a deeper stage.
- `StageAux.router_loss` carries the weighted aux loss. `total_loss` adds it. train.py logs it with beta and tau.
- FLOPs: the head adds 2 d 256 per stage-0 position (`hnet/flops.py`).
- Config: `configs/experiment/surprise_2stage.yaml`. Tests: `tests/test_surprise_router.py`, and the surprise cases of `tests/test_causality.py`.
- First use: [surprise router, in-context tokenization](../../../../lab/experiments/planned/2026-11-06-surprise-router-in-context-tokenization.md).

## Alternatives considered

- **A raw entropy threshold in nats (tau_init = the proxy's theta_0).** Built first. In the smoke, the stage-0 ratio swung from 11.7 to 1.78 within 50 steps. The head's entropy falls from ln 256 to about 2 nats early in training, so one fixed threshold first cuts everything, then nothing. The standardized term removes that drift.
- **Batch statistics instead of running buffers.** The mean over a batch includes later positions of the same row, a small leak that the prefix test would catch.
- **A pure entropy router (no cosine term), threshold set by the ratio loss.** That is the frozen entropy arm with a learned proxy. The cosine term is what beat entropy boundaries by 0.04 bpb, so we keep it and add the entropy on top.
- **Use the model's own final next-byte prediction.** That prediction depends on the main network's output for the current chunk, which depends on the boundaries. Using it for routing would need a sequential pass over the bytes. The encoder-level head is parallel.
- **An OR of the two signals (cut if either says so).** An OR can only add cuts. In-context tokenization needs the router to remove cuts inside a repeated word.

## Consequences

- The stage-0 router has 65,538 more parameters (W, beta, tau) and two buffers, and the step costs one d x 256 head plus a 256-way softmax per byte.
- `beta` can go to 0 if the LM loss does not reward the entropy term. The run logs it, so a null result shows which part failed.
- Lookahead, eoc and given routers are unchanged. `Router.forward` has a new optional argument.
