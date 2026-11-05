# Agent Note: 2025 init rule and partial RoPE

Status: implemented

## Problem

Our init claimed to follow the 2025 H-Net `_init_weights`, and it did not, in two places:

- **Residual count.** Output projections (`out_proj`, `fc2`) get std 0.02 / sqrt(n), where n counts residual-stream additions over this level and every enclosing level. The 2025 code counts one addition for a block without an MLP and two for a block with one (`Isotropic.height`). Ours counted two for every block. All-attention stacks (`T` blocks, the blog split) give the same n. Mamba-2 blocks without an MLP (`m`) get projections up to 1.41x too small.
- **Residual bias.** The 2025 code zeroes `residual_proj.weight` and leaves `residual_proj.bias` at nn.Linear's default init. Ours zeroed both.

A per-tensor comparison against the reference found both. With the reference's exact step-0 weights, our code moved most of the way to the reference's results: seed 1's main-network use rose 10x, and the seed spread in bpb fell 3x ([reference init in our code](../../../../lab/experiments/refuted/2026-11-02-reference-init-in-our-code.md)).

Separately, the reference applies RoPE to 32 of 64 head dims (`rotary_emb_dim` 32). Ours always rotated all of them.

## Decision

`lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/`:

```
Stack.height = sum(2 if block has MLP else 1)                       (blocks.py)
Main.init_weights:  n = n_res + stack.height
Stage.init_weights: n = n_res + encoder.height + decoder.height     (stage.py; inner levels get n)
Stage.residual_proj: weight zeroed, bias = nn.Linear default
model.rope_dim (null = head_dim): apply_rope rotates the first rope_dim dims of each head, the rest pass through
```

- `reflayout_2stage` sets `rope_dim: 32`, the reference value.
- After the fix, the step-0 std of every tensor class with at least 64 elements is within 3% of the reference's (averaged over 4 seeds, 56 classes).
- The residual-bias change applies to every model, the blog split too. Its RNG draws do not change (nn.Linear draws the bias at construction either way), so the only change is the bias value.

## Alternatives considered

- **Config flags for the old behaviour** (`init_residual_count: per_block | two_per_block`, `residual_bias_init: zero | default`). Rejected: the old behaviour is a bug against the code's own stated intent, not a design choice. Runs before this change reproduce from their recorded commit (wandb `env.git`). Flags would keep two init rules alive.
- **Fix the residual count only.** Rejected: the bias is a second divergence from the 2025 code that we found in the same comparison, and the transplant test changed both at once. Leaving one in would make the next result ambiguous.
- **Match the reference's RoPE pair convention too** (rotate halves). Not needed: interleaved and half pairs differ by a fixed permutation of q/k dims, so they give the same model family and the same init distribution.

## Consequences

- Runs at or before commit f4f49b7 used the old init: all runs of [our H-Net with Mamba-2](../../../../lab/experiments/refuted/2026-11-02-our-hnet-with-mamba2.md), [reference layout in our code](../../../../lab/experiments/refuted/2026-11-02-reference-layout-in-our-code.md), and every blog-split run. For T-only models, only the residual bias differs.
- `tests/golden_util.py` zeroes the residual bias of its GPU model, because the goldens were recorded with the old init. The golden test guards forward/backward numerics, not init.
- Tests: `tests/test_layouts.py` covers `Stack.height`, the out_proj std per level, the residual weight and bias, and partial RoPE (tail dims pass through, rotation keeps norms, identity at position 0).
