# Experiment: byte and BPE controls

Status: refuted

Date proposed: 2026-10-31
Algorithm: none — standard method (isotropic byte Transformer; BPE Transformer with an in-corpus tokenizer)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

At this project's scale (about 36M parameters, 786M training bytes), where does a 2-stage byte H-Net land against the two standard controls, a byte-level Transformer and a BPE Transformer, at matched forward FLOPs per byte and the same training bytes? This is a control for the end-of-chunk router runs, and it gives the bpb numbers an outside reference.

## Hypothesis

The 2-stage H-Net reaches lower validation bpb than the BPE Transformer at matched FLOPs/byte, because it learns its chunks end to end (Cartesia Part I: 1.28x compute multiplier for 1 stage). Both reach much lower bpb than the byte Transformer, because compression lets them spend compute on coarse positions (Part II: byte is about 9x worse on compute-to-loss).

## Predictions

Main prediction (only this one answers the Question):

- On edu validation bpb at the end of training, the better 2-stage H-Net arm of the router experiment is lower than the BPE Transformer by more than the H-Net seed spread.

Other predictions (at most two):

- Ordering on edu validation bpb: byte Transformer is the worst of the three.
- The H-Net minus BPE bpb gap is larger on Go validation than on edu validation (Part I: the H-Net gain is larger on code).

If supported, we do: keep the H-Net path and plan the long emergence run on it.

If not supported, we do: before any Phase 2 work, find the cause. First check per-stage LR and ratio-loss weight (the controls get the same LR but have no ratio loss), then the router form. The BPE control becomes the reference every H-Net change must beat.

Gates (the run is broken if one fails; gates are not predictions):

- Forward FLOPs/byte of each control is within 5% of 22.4M, the midpoint of the 2-stage eoc (21.3M) and lookahead (23.5M) arms, by the shared FLOP counter. For BPE, FLOPs per byte = FLOPs per token / measured bytes per token on edu val.
- Same training bytes as the H-Net arms (786M), same edu 0.9 / go 0.1 mix by bytes, same val text, finite loss.
- BPE bpb counts every byte of the val text: sum of token CE over a document / (ln 2 x bytes in that document).

## Method

```
byte control:  HNetLM(stages=[]) at byte level, global causal attention over 4096 bytes,
               width and depth chosen so fwd FLOPs/byte ~ 22.4M
bpe control:   tokenizer = BPE trained on our edu+go train text (no hand pretokenization change:
               standard byte-level BPE with GPT-style pretokenization, vocab 32k)
               model = the H-Net main network shape (d 512) at token level, layers set so fwd FLOPs/byte ~ 22.4M
               seq = 1024 tokens; steps set so total training bytes = 786M
both: same optimizer, LR schedule, batch rows, seeds 0 and 1, wandb group byte-and-bpe-controls
eval: edu / go / sql val bpb; BPE also logs CE per token and bytes per token
```

- Runs after the router arms on the same 5090, sequentially.
- The H-Net side of the comparison is the router experiment's 2-stage arms (3 seeds each); no new H-Net runs.

## Results

**Not supported.** The better 2-stage H-Net arm (`lookahead`) is *worse* than the BPE Transformer on edu val bpb. Paired by seed: +0.141 (s0), +0.159 (s1), mean +0.150 bpb. BPE seed spread is 0.003, and H-Net seed spread is 0.032.

All runs used the local RTX 5090 and the same 786M training bytes (BPE: 11,255 steps x 16 x 1024 tokens, about 790M bytes at the mix's 4.29 bytes/token). Per-run numbers and wandb links: [summaries.json](../results/2026-10-31-byte-and-bpe-controls/summaries.json). Samples: [samples.md](../results/2026-10-31-byte-and-bpe-controls/samples.md). The H-Net rows come from the [router experiment](../refuted/2026-10-30-end-of-chunk-router.md).

| Model | edu bpb (seeds) | Go bpb | SQL bpb | Params | Fwd FLOPs/byte | MB/s |
|---|---|---|---|---|---|---|
| BPE Transformer | 1.273 / 1.271 | 0.864 / 0.860 | 1.671 / 1.688 | 62.5M | 22.1M | 1.38 |
| lookahead 2-stage H-Net | 1.414 / 1.429 / 1.446 | 0.955 / 0.985 / 0.981 | 1.795 / 1.853 / 1.798 | 36.6M | 23.5M | 0.84-0.90 |
| byte Transformer | 1.463 / 1.440 | 0.887 / 0.896 | 1.734 / 1.716 | 4.95M | 22.4M | 1.63-1.65 |
| eoc 2-stage H-Net | 1.510 / 1.467 / 1.542 | 0.983-1.018 | 1.837-1.868 | 36.6M | 20.3-21.4M | 0.84-0.97 |

The same edu numbers in BPE units (CE per BPE token = bpb x ln 2 x 4.556 edu bytes per token): BPE 4.02 nats/token, `lookahead` H-Net 4.47-4.57, byte Transformer 4.55-4.62.

Edu val bpb through training (seed 0; WSD decay starts at 80% of the steps):

| step | 1500 | 3000 | 6000 | 9000 | end |
|---|---|---|---|---|---|
| BPE | 1.625 | 1.485 | 1.398 | 1.367 | 1.273 |
| lookahead 2-stage | 1.883 | 1.721 | 1.585 | 1.523 | 1.414 |
| byte | 2.090 | 1.916 | 1.668 | 1.558 | 1.463 |
| H-Net minus BPE | +0.258 | +0.236 | +0.187 | +0.156 | +0.141 |

The H-Net gap to BPE shrinks for the whole run, and the byte Transformer closes on the H-Net from below.

Gates: FLOPs/byte are within 5% of 22.4M (byte -0.2%, BPE -1.2%, `lookahead` H-Net +4.9%). Training bytes are equal within 0.5%. Every val byte counts for BPE. Loss is finite. All gates hold.

### Other predictions

- Byte Transformer worst: not supported. It sits inside the `lookahead` H-Net seed range on edu, and it beats every H-Net arm on Go and SQL. It has 1/7 of the parameters.
- Larger H-Net minus BPE gap on Go: not supported. The gap is +0.112 on Go and +0.158 on edu. Relative to BPE the H-Net is 13.0% worse on Go and 12.4% worse on edu, which is the same.

### Diagnostic after the results (spike, no prior prediction)

The spike's [coarse-path ablation](../../spikes/2026-10-30-two-stage-hnet-smoke/README.md#coarse-path-ablation-diagnostic-no-prior-prediction) explains the byte-Transformer tie. In every 2-stage checkpoint, the main network output can be swapped for another document's at a cost of +0.001 to +0.043 bpb. The 2-stage H-Net is in effect a stage-0 + stage-1 local model. Its main network (8 layers, d 512) does almost no work.

## Verdict

**Not supported.** At 786M bytes and matched FLOPs/byte, the best 2-stage H-Net is +0.150 bpb worse than BPE (paired mean, 2 seeds). That is 50x the BPE seed spread and 5x the H-Net seed spread.

As planned: the BPE Transformer is the reference that every H-Net change must beat. Find the cause before any Phase 2 work.

## Updated hypothesis

The first cause to fix is the dead main network, not the router and not the budget. A 2-stage H-Net whose main network does no work cannot show a hierarchy win at any data size. A long run would spend its night on a local model. The shrinking gap (+0.26 to +0.14) is real, but it may only be the stage-0/stage-1 local model still learning.

Next: make main-network use a training-time probe (Constitution Principle 3), then find why the main network goes unused at 2 stages but not at 1 stage (`lookahead` 1-stage: +1.0 bpb when shuffled). Candidate causes to test, in this order: the stage-1 decoder's residual path wins early and the main network never catches up (residual scale and main LR), and the main network sees about 540 positions per row with weak gradient through two EMA/STE layers.

## Related

- [End-of-chunk router experiment](../refuted/2026-10-30-end-of-chunk-router.md)
- [Constitution](../../../docs/constitution.md), Success criterion 1.
