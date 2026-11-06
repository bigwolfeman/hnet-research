# Experiment: learned vs entropy boundaries

Status: success

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

## Results

Runs: proxy byte_transformer seed 0, entropy_2stage seed 0 (wandb z8xxxblu, 2026-11-05 09:46 to 14:42) and seed 1 (wandb r5tq8a9q, 2026-11-06 04:54 to 09:45). Queue `ignored/runs/learned-vs-entropy-boundaries/` (chain.sh, then `ignored/runs/resume-chain-2026-11-06.sh` for seed 1). Code: the `entropy-boundaries` worktree at the commit that adds this file. Seed 1 first started on 2026-11-05 and was stopped at 12 minutes, before any `resume.pt`, to free the GPU; it then ran again from step 0.

### Gates

- Suite: 237 passed, 1 skipped (`HNET_TEST_CUDA=1`, suite.log), with `tests/test_boundaries.py`.
- Proxy: the smoke trained; the full run (60,720 steps, 4.95M params) ended with finite loss.
- Calibration (calibration.json, 256 val rows, train-mix weights): thresholds [1.0787, 2.4348] give ratios 3.0 and 3.0 pooled over the mix. Per domain: edu 2.82 and 8.34 bytes per chunk, Go 4.81 and 16.54.
- Both seeds: edu ratios 2.84 and 2.96 at every eval (inside 2.55 to 3.45; frozen boundaries give the same value at every eval). Edu fwd FLOPs/byte 23.07M against 22.75M (+1.4%), proxy not counted.
- Training bytes 12,582,912,000 (192,000 x 65,536). Finite loss.

### Main prediction: edu val bpb at 12.58B bytes

| arm | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| end-to-end (reflayout_2stage) | 1.0905 | 1.0979 | 1.0913 |
| entropy (entropy_2stage) | **1.1378** | **1.1400** | |

The worst end-to-end seed (1.0979) is 0.040 below the best entropy seed (1.1378). All six pairs are in the predicted order. The entropy seeds differ by 0.002.

### Other predictions

- At the 0.79B rung, the entropy mean is below the end-to-end mean: **does not hold**. Entropy 1.3419, 1.3588 (mean 1.3504) against end-to-end 1.3185, 1.3587, 1.3206 (mean 1.3326). Only the late end-to-end seed 1 is level with the entropy arm at this rung.
- Go lower in every end-to-end seed than in every entropy seed: **holds**. End-to-end 0.581, 0.587, 0.582 against entropy 0.712, 0.715.

### Readings

- Both entropy seeds are also above both BPE seeds (1.1156, 1.1109). With frozen entropy boundaries, the H-Net loses its lead over BPE at our scale.
- SQL: entropy 1.479, 1.471 against end-to-end 1.189 to 1.218 and BPE 1.318 to 1.337.
- The entropy boundaries cut code much less at stage 0: Go 5.22 bytes per chunk, against 2.58 in the end-to-end arm. The proxy predicts Go well, so few Go bytes cross theta_0.
- Word alignment: sem/word_s0 AUROC 0.85 and sem/word_s1 0.82, against end-to-end seed 2 at 0.91 and 0.71. Both entropy seeds give the same boundary probes, because the same frozen proxy sets the boundaries.
- In-context tokenization probe: stage-0 chunks for a repeated new word drop from 6.85 at the first occurrence to 1.80 at the fifth. The end-to-end seeds drop by 0.05, 0.05 and 0.00 (constitution item 7). The entropy rule gets this behavior for free: the proxy's entropy on a word falls once the word is in context.
- coarse_shuffle_s0 rise: +1.16 bpb (seed 1), so the main network is in use.
- Step time 84 ms (with the proxy forward) against about 73 ms for the end-to-end arm.

## Verdict

**Supported.** At 12.58B bytes, edu val bpb is lower in every end-to-end seed (1.0905, 1.0979, 1.0913) than in every entropy seed (1.1378, 1.1400).

## Updated hypothesis

At 42M params and 12.58B bytes, learned boundaries beat frozen entropy-proxy boundaries by 0.040 to 0.050 bpb on edu, about 3x the blog's 0.011 to 0.016 at 1e19 to 1e20 FLOPs. The entropy arm also loses to BPE, so in our setup the H-Net's lead over BPE needs the learned router. The entropy arm was not ahead early either: learned boundaries are at least level by 0.79B bytes. Constitution checklist item 5 holds at our scale. The entropy rule does show in-context tokenization, which our learned router does not; a router that reads the context the way the proxy does may be what the blog's new router adds.

## Related

- [Chunks match blog texts](../successes/2026-11-05-chunks-match-blog-texts.md)
- [Reference layout seeds vs BPE](../successes/2026-11-04-reference-layout-seeds-vs-bpe.md)
- Agent Note: [frozen entropy boundaries](../../../.agents/notes/implemented/feature/2026-11-05-frozen-entropy-boundaries.md)
