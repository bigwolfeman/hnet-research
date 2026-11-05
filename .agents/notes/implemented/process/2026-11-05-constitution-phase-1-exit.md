# Agent Note: Constitution 1.2.0, Phase 1 exit and current architecture

Status: implemented

## Problem

The constitution had drifted from the project. It said "Transformers in every stage" and listed Mamba stages as a non-goal, while the model has used Mamba-2 outer stages since [stack layouts and Mamba-2](../feature/2026-11-02-stack-layouts-mamba2.md), which met the constitution's own evidence rule. Its Phase 1 criteria required morpheme-like stage-0 cuts and in-context tokenization. Our model shows neither (AUROC 0.53-0.57; chunks per new word 4.70 to 4.65 over 5 repeats), and the blog credits both to its unpublished router. On 2026-11-05 the user set a different bar: enough of the blog to build on, proven as well as our compute allows, plus a well-optimized training step.

## Decision

[docs/constitution.md](../../../../docs/constitution.md) 1.2.0, agreed with the user on 2026-11-05:

- Vision, scope and sketch describe the reference layout (Mamba-2 outer stages, SWA attention at stage 1, Transformer main). The SSM non-goal is replaced by "do not reverse engineer the blog's unpublished router".
- Phase 1 exits on A (checklist items 1-5 supported, items 6-7 graded and reported) and B (throughput profiled, clear same-math speedups taken).
- New gating item 5: learned boundaries beat entropy-proxy boundaries at the same architecture and ratio (blog Part II, Table 2).
- Morpheme cuts and in-context tokenization become report-only items 6 and 7. Item 7 also uses the blog's Figure 11 texts (recovered to `ignored/blogs-raw/figdata/`).
- Attention may use PyTorch's varlen flash attention (packed inner stages).
- 1.1.0 (same day) added the Lorentz side quest; see [Lorentz hyperbolic H-Net](../../proposed/feature/2026-11-05-lorentz-hyperbolic-hnet.md).

## Alternatives considered

- **Keep criteria 3 and 5 as gates.** Phase 2 would wait on a router the blog does not publish, and our end-of-chunk guess lost by 0.076 bpb. The user chose to build on what we can reproduce.
- **Drop items 6 and 7.** They are the blog's signature behaviors. Grading them keeps the gap visible, and a later router change can be measured against them.
- **No end-to-end-vs-entropy item.** Without it, "the learned chunking works" rests on probes alone. The blog's Table 2 test checks it in bpb at the same architecture and ratio.

## Consequences

- Phase 1 now needs: seeds vs BPE on edu (item 1), the entropy-proxy comparison (item 5), and a graded report on the blog's own figure texts (items 3, 6, 7).
- A later router that passes items 6 and 7 is a measurable upgrade, not a Phase 1 blocker.
