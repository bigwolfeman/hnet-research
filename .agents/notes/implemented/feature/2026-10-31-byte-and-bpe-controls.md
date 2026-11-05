# Agent Note: byte and BPE control models

Status: implemented

## Problem

The 2-stage H-Net bpb has no outside reference. [The planned experiment](../../../../lab/experiments/planned/2026-10-31-byte-and-bpe-controls.md)
needs a byte Transformer and a BPE Transformer at the same forward FLOPs per byte (22.4M) and the same 786M training bytes.

## Decision

- **Byte control** is `HNetLM` with `stages: []`: d 256, 6 layers, global attention over 4096 bytes, 4.95M parameters,
  22.35M forward FLOPs/byte by the shared counter (0.2% off). No new model code. d 320 and 384 gave 20.8M and 20.7M at the
  nearest layer counts (7% and 8% off), so d 256 is the only width in range with a layer count that fits the gate.
- **BPE control** is the same class with `model.kind: token`: vocab 32768, d 512, 9 layers, seq 1024 tokens, batch 16,
  62.5M parameters (untied embedding and head). 100.8M forward FLOPs/token, 4.527 edu-val bytes/token, so 22.27M/byte (0.6% off).
  8 layers gave 20.6M (8% off). The 32768-way head costs 33.5M FLOPs/token, a third of the total.
- **Tokenizer**: HF `tokenizers` byte-level BPE, GPT-2 regex, 1 GB sample of train text (edu 0.9 / go 0.1 by bytes, 64 strided segments),
  EOS between docs. `scripts/prepare_bpe.py` writes uint16 memmaps for all five `.bin` files and `meta.json`, and asserts that the byte
  lengths of the tokens add up to the file size, so every byte is counted once. EOS stands for the 0 separator byte (1 byte).
- **Bytes per token**: edu train 4.528, go train 2.800, edu val 4.527, go val 2.897, sql val 2.077. Train mix 4.265.
- **bpb** is sum of token CE / (ln 2 x bytes the target tokens stand for), on fixed non-overlapping 1024-token windows (256 per val set, about
  the 1.05M bytes the byte models eval on). It also logs `val/{d}/ce_per_token` and `val/{d}/bytes_per_token`.
- **Loader**: `MixLoader(dtype=uint16)` draws a row with probability proportional to byte weight / bytes per token, so the byte mix stays 0.9 / 0.1.
  `train.max_steps` 11255 gives 786.4M bytes; `train.py` asserts the plan is within 1% of `data.token.target_train_bytes`.
- The H-Net path is unchanged. The golden tests pass.

## Alternatives considered

- Evaluate the BPE model on the whole val file: 15M tokens per set is too slow every 500 steps. Fixed windows keep cost near the byte models'.
- Count only document bytes (not the separator) in the bpb denominator: the H-Net predicts the separator byte, so leaving it out would favor BPE.
- Tied embedding and head for the BPE model: not chosen. The H-Net arms are untied, and the experiment fixes FLOPs, not parameters.
- A separate `TokenLM` class: rejected. An isotropic Transformer over ids is `HNetLM` with no stages.

## Consequences

The byte control has 4.95M parameters against about 36M for the H-Net. FLOPs are matched, parameters are not, so the byte
control is the weakest arm by construction (it spends its FLOPs on attention over 4096 positions). Read the comparison as equal compute, not equal size.
