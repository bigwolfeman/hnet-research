# Experiment: learned vs entropy boundaries

Status: planned

Date proposed: 2026-11-05
Algorithm: `lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/boundaries.py` (pseudocode there)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

With the same architecture and the same ratios, does our 2-stage H-Net reach lower edu val bpb with its learned (end-to-end) boundaries than with frozen boundaries from a proxy byte LM's next-byte entropy?

Context: constitution 1.2.0 checklist item 5, the blog's Part II Table 2. There, at 1e19 to 1e20 FLOPs, end-to-end H-Net beat entropy-proxy boundaries by 0.011 to 0.016 bpb (0.8001 vs 0.8160; 0.7488 vs 0.7623; 0.7027 vs 0.7138). The entropy rule is the defining heuristic of BLT. Our seeds 0 and 1 of the end-to-end arm exist ([reference layout vs BPE](../refuted/2026-11-02-reference-layout-vs-bpe.md): 1.0905, 1.0979 at 12.58B), and seed 2 ran in [reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md).

## Hypothesis

End-to-end training is what makes the boundaries good, at our scale too. Entropy boundaries cut at surprising bytes but cannot learn what helps the main network, so at the end of training they lose. Early in training the order may flip: entropy boundaries are useful from step 0, while each learned-router seed must first come online (our seeds crossed BPE at 2.1B and 6.8B bytes).

## Predictions

Arms: end-to-end = `reflayout_2stage` seeds 0, 1, 2. Entropy = `entropy_2stage` seeds 0, 1: the same model with router "given" at both stages, boundaries where the proxy's next-byte entropy exceeds theta_0 (stage 0) and theta_1 (stage 1), calibrated to 3 and 9 bytes per chunk on val rows of the train mix.

Main prediction (only this one answers the Question):

- At 12.58B bytes, edu val bpb is lower in every end-to-end seed (0, 1, 2) than in every entropy seed (0, 1). All six pairs in that order: supported. Any end-to-end seed at or above any entropy seed: not supported.

Other predictions (at most two):

- At the 0.79B rung, the entropy seed mean is below the end-to-end seed mean (entropy boundaries ahead early).
- At 12.58B bytes, Go val bpb is lower in every end-to-end seed than in every entropy seed.

If supported, we do: item 5 holds. Learned chunking beats the BLT-like heuristic at our scale, so the stage-1 units that Phase 2 loops over earn their place. Record it in the Phase 1 report.

If not supported, we do: read the rungs. If the entropy arm leads at every rung, our learned router adds nothing over a heuristic at 42M params and 12.58B bytes, and item 5 fails: bring it to the user before Phase 2, with the router (the blog's unpublished change) as the open lead. If the end-to-end arm leads but one pair inverts, the arms tie inside seed noise: write the larger test (more seeds) as the next planned file.

Gates (the run is broken if one fails; gates are not predictions):

- Our suite passes (`HNET_TEST_CUDA=1`), with `tests/test_boundaries.py`.
- Proxy: the 200-step smoke trains; the full proxy run ends with finite loss.
- Calibration: on its rows, stage-0 and stage-1 bytes per chunk within 2% of 3 and 9 (pooled over the mix).
- Entropy arm: edu ratios within 15% of 3 x 3 at every rung; edu fwd FLOPs/byte within 5% of the end-to-end arm's (22.75M) at every rung, proxy not counted; training bytes as in the seeds-vs-BPE gates; finite loss.

## Method

```
code:      lab/spikes/2026-10-30-two-stage-hnet-smoke at the commit that adds this file
proxy:     byte_transformer (6 x d256, global attention, 22.35M fwd FLOPs/byte; the BLT proxy is a byte Transformer)
           seed 0, train=ladder, max_steps 60720 = 3.98B bytes = 12.58B / 10^0.5 (the blog's proxy budget rule),
           ladder_fracs [], WSD decay over the last 20%. Its FLOPs are not counted, in training or inference.
calibrate: scripts/calibrate_entropy.py <proxy final.pt> --ratios 3 3 --rows 256  (val rows, train-mix weights)
           -> model.boundaries.thresholds [theta_0, theta_1]
arms:      entropy_2stage seeds 0, 1; train=ladder (12.58B bytes, rungs at 0.79B and 3.15B), as reflayout_2stage
           model.boundaries.proxy=<proxy final.pt> model.boundaries.thresholds=[theta_0,theta_1]
entropy:   H_t = entropy of proxy(x_<t) over byte t, fp32; byte t starts a stage-s chunk if H_t > theta_s; byte 0 always
given:     p_t = boundary in {0, 1}; no router parameters, ratio-loss weight 0, no RatioController; EMA and STE as usual
grade:     val/edu/bpb at the end of each run against reflayout_2stage seeds 0, 1 (A) and 2 (seeds vs BPE)
readings:  rung bpb, Go and SQL bpb, sem/* AUROC of the entropy boundaries vs the learned ones, coarse-path shuffle
```

- Queue: after the seeds-vs-BPE queue. wandb project `hnet-router`, group `learned-vs-entropy-boundaries`.
- Cost: proxy about 1 h; each entropy arm about 4.5 to 5 h (the reflayout step plus one proxy forward). About 11 h in total.

## Related

- [Chunks match blog texts](../successes/2026-11-05-chunks-match-blog-texts.md)
- [Reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md)
- Agent Note: [frozen entropy boundaries](../../../.agents/notes/implemented/feature/2026-11-05-frozen-entropy-boundaries.md)
