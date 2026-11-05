# Experiment: reference checkpoint main-network use

Status: success

Date proposed: 2026-11-01
Algorithm: none (a probe of released weights; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Does a working 2-stage H-Net use its main network in a way our coarse-path probe can see?

Context: in every 2-stage model we trained, the main network carries no document information. Shuffling its output across documents costs at most 0.0008 bpb, even at 12.58B training bytes ([compute ladder](../refuted/2026-11-01-compute-ladder-vs-bpe.md)). We have never run the probe on a 2-stage H-Net that is known to work. Cartesia released one: `hnet_2stage_L` (goombalab/hnet, trained on 100B tokens of FineWeb-Edu). Experiment B (train the reference code at our scale) reads its result with this probe. So the probe must first be shown to see main-network use when it exists.

## Hypothesis

In a trained 2-stage H-Net the main network does most of the document-level modeling. The decoder at stage 1 reads the main network's output for content, not as a constant. So our shuffle probe, which swaps that output between documents, raises bpb by a clear margin in the reference model. Our 2-stage models are the exception, not the probe.

## Predictions

Main prediction (only this one answers the Question):

- On the same edu val batches, swapping the main network's output across documents raises bpb more in `hnet_2stage_L` than in our blog-split 2-stage final model. Our model's value over its last 20 evals ranged up to 0.0003 bpb. Reference rise above that range: supported. Inside or below it: not supported.

Other predictions (at most one):

- At the stage-1 decoder input, the main network's share of the signal norm, `||up * conf|| / ||resid||` averaged over positions, is larger in `hnet_2stage_L` than in our blog-split model.

If supported, we do: the probe works, and our dead main network is a real defect of our model or our training regime. Run experiment B (reference code at our scale) and read it with this probe.

If not supported, we do: either the probe cannot see main-network use, or even a working 2-stage H-Net barely uses its main network for document content. Then B cannot be read with this probe. Redesign the probe first (for example, a long-range copy task that only the main network can solve), then run B.

Gates (the run is broken if one fails; gates are not predictions):

- Harness reproduces our model: on the 8 probe batches, our blog-split `final.pt` gives the `probe/coarse_base_bpb` and `coarse_shuffle_s1` that training logged at step 192,000 (1.2960 and 0.00029), within 1e-3 bpb.
- The `flash_attn` shim matches a direct float32 attention, RMSNorm, rotary, and SwiGLU reference (unit tests, bf16 tolerance).
- Reference model sanity: its edu val bpb is lower than every model we trained (it is about 30x larger and saw about 30x more data). If not, the shim or the data mapping is wrong.
- Finite outputs. Stage ratios of the reference model on our val text are reported (no target to hit).

## Method

```
data: our edu val memmap, ByteLoader(seq 4096, batch 16).fixed(16, seed 1234)[:8]     # the exact probe batches of training
      reference inputs: byte 0 (our doc separator) -> 254 (reference BOS); targets mapped the same way
models:
  ours:  blog_split_2stage_s0/final.pt (ladder attempt 2), eager, bf16 autocast
  ref:   cartesia-ai/hnet_2stage_L (sha256 5711de80...), goombalab/hnet @ 3673fe1, bf16, mask mode (no packing)
         flash_attn replaced by a torch shim (SDPA, torch RMSNorm/rotary/SwiGLU); Mamba-2 via mamba_ssm Triton kernels
for model in (ours, ref):
    base = bpb(batches)
    for stage s, mode in {zero, shuffle}:
        coarse path at stage s = dechunked output of the network inside stage s
        zero:    path = 0 ;  shuffle: path = path rolled by one row (another document)
        rise[s, mode] = bpb(ablated) - base
    share[s] = mean over positions of ||path * conf|| / ||resid||   at stage s's decoder input
also: hnet_1stage_L (same probe, stage 0 only) as a 1-stage calibration point
```

- Code: `lab/spikes/2026-11-01-reference-hnet-probe/` (harness, shim, tests). Reference clone: `ignored/ref/hnet` (not committed). Checkpoints: `ignored/ref/ckpt/`.
- Our model's stage s and the reference's stage s ablate the same thing: the inner network's dechunked output before the residual add.
- Risk: FineWeb-Edu val documents may be in the reference training set. This lowers the reference bpb but does not change the meaning of a shuffle rise.

## Results

Run 2026-11-01 on the local RTX 5090. Harness: `lab/spikes/2026-11-01-reference-hnet-probe/` (`probe_ours.py`, `probe_ref.py`). Raw JSON per model: [results/](../results/2026-11-01-reference-checkpoint-main-use/).

### Gates (all hold)

- Harness reproduces our model: `base_bpb` 1.29605 (logged 1.29605), `coarse_shuffle_s1` 0.000303 (logged 0.000289).
- The shim tests pass (10 of 10: windowed, packed, and varlen attention; fused-add RMSNorm; rotary, interleaved, with offsets, and the inverse; SwiGLU).
- Reference sanity: `hnet_2stage_L` has 0.7694 edu bpb on the probe batches. Our best model has 1.1582, and BPE has 1.1156 on the full eval set.
- All outputs finite.

### Coarse-path probe (8 edu val batches of 16 x 4096 bytes; bpb rise when the path is ablated)

| model | base bpb | ratios on edu val | zero s0 | shuffle s0 | zero s1 | **shuffle s1** | share s0 | share s1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| reference `hnet_2stage_L` | 0.7694 | 2.79 x 2.48 | 1.430 | 2.448 | 0.683 | **1.346** | 0.152 | 0.347 |
| our blog split 2-stage | 1.2960 | 4.03 x 2.00 | 1.050 | 1.173 | 0.017 | **0.0003** | 0.480 | 0.262 |
| reference `hnet_1stage_L` | 0.7834 | 4.71 | 1.159 | 2.026 | | | 0.128 | |
| our matched 1-stage | 1.1582 | 4.12 | 2.432 | 1.705 | | | 1.260 | |

Share = mean over positions of `||coarse path|| / ||residual||` at the stage's decoder input.

- Main prediction: the reference shuffle rise at stage 1 is 1.346 bpb, about 4,500x our model's 0.0003. Holds.
- Other prediction: share at stage 1 is 0.347 in the reference and 0.262 in ours. Holds, but the gap is small: our main network sends a signal of normal size that carries no document information.

### Observations (not predictions)

- **In the reference models, a wrong coarse path hurts more than no coarse path.** Shuffle rises exceed zero rises at every stage: 1.7x and 2.0x in the 2-stage model, 1.75x in the 1-stage model. The decoder reads document content from the coarse path, so another document's content misleads it. In our models it is the other way round at two of three points: stage 1 of the 2-stage (0.017 zero vs 0.0003 shuffle) and the 1-stage (2.43 vs 1.71). Only our 2-stage stage 0 has the reference order (1.05 vs 1.17). Our decoders depend on the path's typical value more than on its content. Even our working 1-stage model has this signature.
- **The reference coarse path is small next to its residual.** Share s0 is 0.13-0.15 in both reference models and 0.48-1.26 in ours. The reference decoders run mainly on the byte-level residual and add a small, content-bearing coarse signal.
- **The reference routers run below our targets on this text.** `hnet_2stage_L` gives 2.79 x 2.48 (6.9 bytes per main position), and `hnet_1stage_L` gives 4.71. We hold 4 x 2 (8.0 bytes).

## Verdict

**Supported.** Shuffling the main network's output across documents raises edu val bpb by 1.346 in the released `hnet_2stage_L`, against 0.0003 in our blog-split 2-stage model on the same batches. The probe sees main-network use when it exists. Our dead main network is a real defect of our model or our training regime, not a probe artifact.

As planned: run experiment B, the reference code at our scale, and read it with this probe.

## Updated hypothesis

A working 2-stage H-Net puts heavy document-level modeling in its main network. Shuffling it costs more than half the stage-0 shuffle cost (1.35 vs 2.45). Ours puts none there. The probe now separates the two cases by more than three orders of magnitude, so experiment B has a clear readout. If the reference code trained at our scale shows a rising `shuffle s1`, the cause is in our implementation or recipe. If it stays near zero, the cause is scale.

The shuffle-vs-zero signature is a second readout with no extra cost. A healthy coarse path hurts more when shuffled than when zeroed. B should log both, and a fix to our model should move both.

## Related

- [Compute ladder vs BPE](../refuted/2026-11-01-compute-ladder-vs-bpe.md)
- [Deep-main compute split](../refuted/2026-10-31-deep-main-compute-split.md)
- [Cartesia reference notes](../../../docs/references/cartesia-tokens-to-bytes.md)
