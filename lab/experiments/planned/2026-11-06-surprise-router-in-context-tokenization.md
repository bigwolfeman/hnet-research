# Experiment: surprise router, in-context tokenization

Status: planned

Date proposed: 2026-11-06
Algorithm: router `surprise` in `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/routers.py` (pseudocode in the module docstring)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

If the stage-0 router also reads the encoder's own next-byte entropy, all trained end to end, does our H-Net show in-context tokenization? That means fewer stage-0 chunks in a new word each time the word repeats (constitution checklist item 7, the blog's made-up-word figure).

Context: in [learned vs entropy boundaries](../successes/2026-11-05-learned-vs-entropy-boundaries.md), frozen entropy-proxy boundaries cut a repeated made-up word into 6.85 chunks at its first occurrence and 1.80 at its fifth. Our learned lookahead router drops by 0.05, 0.05 and 0.00 (three seeds), and the attention-only arm by 0.00. The learned router finds words (word AUROC 0.91 to 0.93), but its cut rule does not use context.

## Hypothesis

In-context tokenization needs two things.

1. A signal at byte t that says "this byte was predictable from context".
2. A router that uses that signal.

The proxy's next-byte entropy is that signal: once a word is in context, the proxy's induction makes its later bytes predictable. The lookahead rule p_t = (1 - cos(q h_{t-1}, k h_t)) / 2 measures how much the encoder state changes between two bytes. Nothing in training ties that change to predictability, and the gradient that reaches the router (through the STE and the ratio loss) is weak. So we give the router the signal directly.

- An aux head on the stage-0 encoder predicts the next byte, with its own CE.
- Its entropy over byte t (computed before byte t is read, and detached) is standardized by running stats and shifts the lookahead logit: z_t = logit(p^lookahead_t) + beta (Hz_t - tau).
- beta and tau are learned.

The aux CE also trains the encoder to predict bytes from context, so the encoder learns the copying that a repeated word needs. We expect the drop to appear, with edu bpb no worse than the lookahead seeds: the router keeps the cosine term that made learned boundaries beat entropy boundaries.

## Predictions

Arms: `surprise_2stage` seed 0 (new). Controls: `reflayout_2stage` (lookahead) seeds 0, 1, 2. ICL drops 0.05, 0.05, 0.00; edu val bpb 1.0905, 1.0979, 1.0913. Reference: the entropy arm (drop 5.05). None of them is rerun.

Main prediction (only this one answers the Question):

- At 12.58B bytes, the stage-0 in-context drop (`probe/icl_s0_drop_1_to_5`, chunks in a made-up word at occurrence 1 minus occurrence 5) of surprise_2stage seed 0 is larger than in every lookahead seed (above 0.05): supported. At or below 0.05: not supported.

Other predictions (at most two):

- At 12.58B bytes, edu val bpb of surprise_2stage is inside or below the lookahead seed range (at or below 1.0979).
- At 12.58B bytes, the first occurrence of a made-up word gets more stage-0 chunks (`probe/icl_s0_occ1`) than in every lookahead seed (3.95, 4.7, 3.8). A new word is surprising.

If supported, we do: our router reproduces item 7 end to end. Then we test which part carries it: a run with the aux head on a detached encoder (does the encoder already know?) and one with beta fixed at 0 (is it the entropy or the CE on the encoder?). Phase 2 builds on the router that shows the behavior if its bpb holds.

If not supported, we do: read the aux head's own entropy on the probe words.
- If the entropy falls on repeats but the cuts do not, the router learned to ignore it (check beta). The objective does not reward the behavior, and the fix is in the loss.
- If the entropy does not fall, the stage-0 encoder (4 Mamba-2 blocks) cannot copy the word from context. The fix is a stage-0 attention block.

Gates (the run is broken if one fails; gates are not predictions):

- Suite passes (`HNET_TEST_CUDA=1`), with `tests/test_surprise_router.py` and the surprise cases of `tests/test_causality.py`.
- Smoke: 200 steps train (done 2026-11-06 13:45, params 41,866,882, stage-0 ratio 3.43 at step 200).
- Edu ratios within 15% of 3 x 3 at every rung. Edu fwd FLOPs/byte within 5% of the lookahead arm's 22.75M at every rung (the aux head adds 2 x 256 x 256 per byte and is counted).
- Training bytes 12,582,912,000 (192,000 x 65,536). Finite loss.

## Method

```
code:    lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
arm:     surprise_2stage seed 0 = reflayout_2stage with the stage-0 router "surprise"; stage 1 keeps lookahead
router:  logits_t = W h_t (W: 256 x 256, no bias)          # predicts byte t+1 from the encoder state at t
         aux     = lm_weight * CE(logits_t, x_{t+1})       # lm_weight 1.0; trains W and the encoder
         H_t     = entropy(softmax(logits_{t-1})), detached  # nats: how unsure the encoder was about byte t
         Hz_t    = (H_t - mu) / sigma                      # running mean/var buffers, momentum 0.1, training only
         z_t     = logit(clamp(p^lookahead_t, 1e-3, 1 - 1e-3)) + beta * (Hz_t - tau)
         p_t     = sigmoid(z_t), p_0 = 1                   # beta_init 1.0, tau_init 0 (std units)
         start anchors, read_t = cumsum(b)_t - 1, ratio loss and RatioController as in reflayout_2stage
budget:  configs/train/ladder.yaml: 12.58B bytes, rungs at 1/16 and 1/4, the main run decays over its last 20%
grade:   probe/icl_s0_drop_1_to_5 at the end of the run (data/probe_paragraphs.json: 20 made-up words,
         5+ occurrences each, 33 to 205 bytes apart) against the lookahead seeds
readings: probe/icl_s0_occ1..5, beta and tau over training (train/router_beta_s0, train/router_tau_s0),
         train/router_loss_s0, sem/word_s0 AUROC, coarse_shuffle_s0, Go and SQL bpb, ICL drop at the rungs
```

- Launch from the spike dir: `TRAIN=ladder scripts/run_arms.sh <repo>/ignored/runs/surprise-router-in-context-tokenization/arms surprise-router-in-context-tokenization surprise_2stage:0`.
- wandb project `hnet-router`, group `surprise-router-in-context-tokenization`.
- Cost: about 4 h (the lookahead step, about 73 ms, plus the aux head).

- Agent Note: [surprise router](../../../.agents/notes/implemented/feature/2026-11-06-surprise-router.md).

## Related

- [Learned vs entropy boundaries](../successes/2026-11-05-learned-vs-entropy-boundaries.md)
- [Chunks match blog texts](../successes/2026-11-05-chunks-match-blog-texts.md) (item 7 not seen)
- [End-of-chunk router](../refuted/2026-10-30-end-of-chunk-router.md)
- [Cartesia blog notes](../../../docs/references/cartesia-tokens-to-bytes.md)
