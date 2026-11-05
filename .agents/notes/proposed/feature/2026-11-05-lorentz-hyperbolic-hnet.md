# Agent Note: Lorentz hyperbolic H-Net

Status: proposed

## Problem

Every representation in our H-Net is Euclidean: the byte embedding, the router's cosine similarity, the chunk vectors that move between stages, and the LM head. The model's structure is a hierarchy. Bytes group into chunks, and chunks group into larger chunks. A tree of depth k with branching b has b^k leaves. Euclidean volume grows polynomially with radius, so a tree embeds with distortion that grows with its size. Hyperbolic volume grows exponentially, so trees embed with low distortion in a few dimensions. The user asked for this test on 2026-11-05. It was added to [the constitution](../../../../docs/constitution.md) as a side quest (version 1.1.0).

## Proposal

Three arms on one changed part: pure Euclidean (the control), 20% Lorentz / 80% Euclidean, and pure Lorentz. Compare at matched FLOPs/byte with the usual probes (bpb per domain, `sem/*`, in-context tokenization).

Source of the 20/80 split: the user's own reimplementation of HELM ("HELMET") on token models (2026-11-05). There, 50/50 gave the best loss per step, and the cost of the Lorentz operations moved the best loss-per-FLOP point to about 20/80. It is a compute-curve result, not a loss result. No published paper reports this split (literature check, 2026-11-05).

H-Net angle: Lorentz operations cost per position. Our inner stages see about 1/3 and 1/9 of the byte positions, so the compute pressure toward 20/80 is weaker there, and the best split may differ by stage. The user recalls an earlier conversation with a clever Lorentz optimization trick for H-Nets, but not what it was. Look for H-Net-specific speedups before accepting the token-model cost curve.

References found by that check:
- HELM, "Hyperbolic Large Language Models via Mixture-of-Curvature Experts" (arXiv 2505.24722, NeurIPS 2025). Fully Lorentz LLM at 100M and 1B, with no Euclidean-split ablation. It is the closest source for Lorentz LM building blocks. Its stability note: the 1B models were unstable with full embedding training, so it trains only the space-like part of the embeddings.
- "Hyperbolic Large Language Models" survey (arXiv 2509.05757): mixed hyperbolic-Euclidean designs are the most common, but it gives no split policy.
- HypLoRA (arXiv 2410.04010): pretrained LLM token embeddings measure as tree-like (delta-hyperbolicity).
- No prior work applies hyperbolic geometry to dynamic chunking (H-Net, BLT, MEGABYTE) or to a boundary router. Candidate first steps, not yet chosen:

```
mixed vector:    h = [h_E (0.8 d dims, Euclidean) ; x (0.2 d dims, a Lorentz point)]
lorentz point:   x = (x_0, x_s),  <x, x>_L = -x_0^2 + |x_s|^2 = -1/c,  x_0 > 0
lift:            x_s = h (Euclidean features),  x_0 = sqrt(1/c + |x_s|^2)
distance:        d(x, y) = acosh(-c <x, y>_L) / sqrt(c)
mixed score:     s(h, h') = <h_E, h'_E> - w * d(x, x')^2        (one combine rule; the paper's rule wins if it differs)
router (cand.):  p_t = sigmoid(a * d(q_t, k_{t-1}) - b)       in place of (1 - cos(q_t, k_{t-1})) / 2
embed/head (cand.): logits_v = -d(h_t, e_v)^2                 hyperbolic nearest-prototype head
```

The design is open. The first place to change gets picked with the user before any code lands.

## Alternatives considered

- **Poincaré ball model.** Same geometry, but its distance and gradients are numerically unstable near the boundary. The Lorentz model is better conditioned in fp32, and that matters in bf16 training. The user asked for Lorentz.
- **Hyperbolic everywhere at once (fully hyperbolic layers).** It changes too many things in one run, and a loss would have no single cause.
- **50/50 split.** Best loss per step in the user's token-model runs, but worse per FLOP. Add it as a fourth arm only if the per-stage cost argument above makes it cheap at the coarse stages.
- **No change: a learned hierarchy may not need hyperbolic geometry.** This is the null hypothesis that the Euclidean control tests.

## Acceptance criteria

- A planned experiment in `lab/experiments/planned/` with one main prediction against the Euclidean control, at matched FLOPs/byte.
- The Lorentz operations live in one shared module with tests: points on the hyperboloid, distance symmetry, and finite gradients in bf16 autocast.

## Risks

- Numerical: acosh near 1 and x_0 growth in bf16. Keep the Lorentz math in fp32. HELM saw 1B instability with fully trained hyperbolic embeddings.
- The curvature c is a new hyperparameter. Fix it, or learn one scalar per stage. Do not sweep it.
- FLOPs: the distance costs one extra dot product per pair. Count it in `flops_per_byte`.
