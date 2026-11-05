# Project Constitution

**Status**: Ratified

**Version**: 1.2.0
**Ratified**: 2026-10-30
**Last Amended**: 2026-11-05

This file is the long-term track for the project. Standing orders and file layout live in [AGENTS.md](../AGENTS.md).

## Vision

We build a byte-level language model with a learned hierarchy of 2 or more chunking stages. Its coarsest stage is a looped (weight-tied, iterated) network. The loop runs over a sequence that is about 8x shorter than the bytes. Thus each loop iteration costs a fraction of a byte-level or token-level iteration. The fine stages run once, to encode bytes up into chunks and to decode chunks back down to bytes.

The bet: a looped model gets more depth per FLOP when it loops only at the coarse level. A hierarchy that learns semantic chunks gives the loop a meaningful unit to think over. The result must beat an unlooped model of the same FLOPs per byte.

The project has two phases. Phase 1 recreates the multi-stage H-Net behavior from the Cartesia posts ([reference](references/cartesia-tokens-to-bytes.md)) as far as our compute allows. It uses the reference layout: Mamba-2 outer stages and a Transformer main network. Phase 2 makes the main stage loop and measures the amortized gain. Phase 2 does not start until Phase 1 meets its success criteria.

## Problem

Looped LMs add depth without parameters, but each loop costs a full pass over every position. At byte resolution that cost is very high. At BPE resolution the loop runs over fixed, hand-made units. A learned hierarchy can make the looped sequence short and semantic. Nobody has published a looped H-Net, and Cartesia has not released their improved multi-stage router.

## Users and operators

The researcher runs and reads the experiments. Agents implement, run, and write up. A wrong result costs GPU nights and can send Phase 2 down a false path. That is why every claim needs a control.

## Scope

**In scope**

- A 2-stage (and later N-stage) byte H-Net in the reference layout: Mamba-2 encoders and decoders (stage 1 adds one sliding-window attention block on each side) and a global-attention Transformer main network. Why Mamba-2: [stack layouts and Mamba-2](../.agents/notes/implemented/feature/2026-11-02-stack-layouts-mamba2.md).
- Router variants, starting with lookahead (2025) vs end-of-chunk (our reading of the new H-Net).
- Ratio control per stage, with targets near 4x and 2x.
- Baselines at matched FLOPs/byte: byte Transformer, BPE Transformer, 1-stage H-Net.
- Probes for hierarchy quality (see Success criteria).
- Phase 2: a looped main stage with a loop count R, compared against unlooped models.
- Side quest (see below): Lorentz-model hyperbolic equivalents of the H-Net's Euclidean parts.

**Out of scope (non-goals)**

- Reverse engineering the blog's unpublished router. We build on the 2025 lookahead router (the blog's "LH-Net") and test our own router guesses as experiments.
- Reproducing the Cartesia scale (3B, 1T tokens). We reproduce directions, not numbers.
- Vision, DNA, audio.
- Prompt-boundary corrections for BPE baselines.

## Success criteria

Phase 1 is done when both of these hold:

- **A. Enough of the blog to build on, proven at our scale.** Every item below is graded by an experiment (supported, not supported, or can't tell) at the largest scale we can run. Items 1 to 5 must be supported. Items 6 and 7 are graded and reported, but they do not gate Phase 2: the blog credits them to its unpublished router, and our router is the 2025 one.
- **B. Throughput.** The training step is profiled, and the clear same-math speedups are taken and recorded in an Agent Note.

Checklist:

1. **Compute win.** At matched FLOPs/byte, the 2-stage H-Net reaches lower validation bpb than the 1-stage H-Net and the BPE Transformer.
2. **Stable ratios.** Each stage holds its target compression through training without collapse (all-boundary or no-boundary).
3. **Word-level boundaries.** Stage-1 boundaries align with word boundaries better than a matched fixed-stride chunker does, and stage-1 chunks merge words into multi-word units. Stage 0 finds word starts with the spaces deleted (the semantic chunking score, `sem/*`).
4. **Domain compression.** The ratio of H-Net bytes-per-chunk to BPE bytes-per-token (cl100k, measured on the same val text) is higher on Go than on English, as in Cartesia Figure 6. The same holds on held-out SQL.
5. **End-to-end boundaries matter.** With the same architecture and ratio, learned boundaries give lower bpb than boundaries from an entropy-threshold proxy model (Cartesia Part II, Table 2).
6. **Morpheme-like stage-0 boundaries.** Inside words, stage-0 boundaries fall at morpheme boundaries more often than at other positions (the blog's `track|ers` vs LH-Net's `r|hino`).
7. **In-context tokenization.** On made-up words repeated in a paragraph (our probe set, and the blog's Figure 11 texts where we can recover them), chunks per occurrence go down as the word repeats. Logged as a curve over training.

Phase 2 is done when a looped coarse stage beats an unlooped model of equal FLOPs/byte on validation bpb, and the gain grows with R up to some point.

## Side quests

Side quests are in scope but do not gate Phase 1 or Phase 2. Each one gets its own planned experiment with a Euclidean control at matched FLOPs/byte.

### Lorentz hyperbolic H-Net

**Question**: Do hyperbolic representations in the Lorentz (hyperboloid) model help an H-Net? A chunk hierarchy (bytes in chunks, chunks in larger chunks) is tree-shaped. Hyperbolic space embeds trees with lower distortion than Euclidean space of the same dimension.

**Arms**: pure Euclidean (the control), a 20% Lorentz / 80% Euclidean mix, and pure Lorentz. The split comes from the user's own reimplementation of HELM ("HELMET", [arXiv 2505.24722](https://arxiv.org/abs/2505.24722)) on token models. There, 50/50 gave the best loss, and the compute cost of the Lorentz operations moved the best loss-per-FLOP point to about 20/80.

**H-Net angle**: a Lorentz operation costs per position, and an H-Net's inner stages see fewer positions (about 1/3 and 1/9 of the bytes here). So the compute pressure that moved token models from 50/50 to 20/80 is weaker at the coarse stages. The best split may differ by stage. The user recalls a clever Lorentz optimization trick for H-Nets from an earlier conversation, but not what it was. H-Nets are underexplored, so look for H-Net-specific speedups before accepting the token-model cost curve.

**Candidate places**: the byte embedding and LM head, the router's boundary score (Lorentz distance in place of cosine similarity), and the chunk representations that pass between stages. Pick the first place with the user before any code.

**Timing**: tangential to looping. It can run alongside Phase 2, on the Phase 1 base model.

## Constraints

- Hardware: one RTX 5090 (32 GB, about 26 GB free) for full runs. A DGX Spark and a 3070 are available for side jobs.
- Power: the machine cannot run the GPU above about 530 W and a CPU load at the same time. Data loading must be cheap on the CPU: pre-encoded uint8 memmaps, no per-step tokenization.
- Full training runs happen at night. Daytime is for smoke tests at small size.
- Stack: PyTorch, bf16 compute (fp32 master weights and grads where needed), Hydra configs, wandb with the full resolved config on every run. Python env via `uv`.
- Attention: PyTorch SDPA, flex_attention, or PyTorch's own varlen flash attention (`torch.nn.attention.varlen`). Do not depend on the flash-attn package. Mamba-2: mamba_ssm's Triton kernels, with causal_conv1d built for sm_120.
- Data: FineWeb-Edu (on disk) plus Go from `bigcode/starcoderdata` (The Stack v1.2 derived; Stack v2 ships only Software Heritage IDs). SQL from the same source is eval-only and never in training, so it is an out-of-domain check. Go and SQL had the largest H-Net-vs-BPE compression gaps in Cartesia Figure 6.

## Principles

### Principle 1: Every stage is one class

**The rule**: An H-Net stage is one module type that wraps an inner network. The inner network is either another stage or the main network.

**Why**: Recursion is the point of the architecture. Special-casing stage 0 or stage 1 makes N-stage and looped variants painful.

**How to apply**: `Stage(encoder, router, inner, decoder)`. The main network and the looped main network have the same interface as a stage.

### Principle 2: Match FLOPs per byte, and count them

**The rule**: Every comparison states FLOPs/byte for every model, counted by one shared function, with causal attention FLOPs only.

**Why**: H-Nets move compute between stages, and the compression ratio changes what a "token" costs. Without one FLOP counter, claims drift.

**How to apply**: One `flops_per_byte(cfg, measured_ratios)` utility. Every wandb run logs it. Use measured ratios, not targets.

### Principle 3: Hierarchy probes run inside training

**The rule**: Boundary alignment and in-context tokenization probes run as periodic eval hooks. They are not post-hoc scripts.

**Why**: The semantic behaviors appear late in training. A curve over steps shows the direction early and shows when the behavior starts.

**How to apply**: A fixed probe set in the repo, evaluated every N steps, logged to wandb as scalars plus a rendered chunk sample.

### Principle 4: Router variants are config, not forks

**The rule**: Router type, ratio target, and ratio-loss weight are Hydra config fields per stage.

**Why**: The router is the least-known part of the new H-Net. We will swap it often.

**How to apply**: A router registry keyed by name. Each run's config records the router for every stage.

## Architecture sketch

```
bytes ─▶ Stage0.enc (Mamba-2 x4) ─▶ router0 ─▶ chunks (~3 bytes)
                                                   │
                  Stage1.enc (SWA attn + Mamba-2 x4) ─▶ router1 ─▶ chunks (~9 bytes)
                                                                 │
                                         Main (global Transformer; Phase 2: looped R times)
                                                                 │
                  Stage1.dec (Mamba-2 x4 + SWA attn) ◀─ dechunk1 (+ residual) ◀─┘
                                                   │
bytes ◀─ Stage0.dec (Mamba-2 x4) ◀─ dechunk0 (+ residual) ◀─┘
```

Production code goes in `src/hnet/` after human review. Spikes live in `lab/spikes/`. Design decisions are Agent Notes in `.agents/notes/`.

## When this constitution applies

- All new work that could change what the project is
- Features, refactors, and promotions into production

**Override**: only with explicit user authorization and a one-line rationale recorded in the change. If the vision itself changed, amend this file with the user. Do not silently ignore it.

## Governance

Amend with the user when the vision or a principle is wrong. Bump:

- MAJOR: the project is a different thing
- MINOR: new principle or new in-scope capability
- PATCH: clarification

Record substantial amendments in an Agent Note under `.agents/notes/`.
