# Agent Note: Stack layouts and Mamba-2 outer stages

Status: implemented

## Problem

Our 2-stage H-Net has Transformer encoders and decoders only. In the reference H-Net, swapping its Mamba-2 outer stages for Transformer blocks cut main-network use about 16x and gave the dead signature that our model shows ([difference ablation](../../../../lab/experiments/refuted/2026-11-01-reference-hnet-difference-ablation.md)). To test the fix in our own model, its stacks must be able to hold Mamba-2 blocks, at matched FLOPs, with the probes, compile path, and FLOPs counter unchanged for every existing arm.

## Decision

A `Stack` is built from a layout string in the reference H-Net's grammar (`lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/blocks.py`):

```
parse_layout("T1m4") -> [(attn, mlp), (mamba, no mlp) x 4]     t/T attention, m/M Mamba-2; upper case adds SwiGLU
stage config: enc_layout / dec_layout (e.g. "M2"); without them "T<enc_layers>" / "T<dec_layers>" (model.stage_layouts)
Block: x += Mixer(RMSNorm(x)) ; x += SwiGLU(RMSNorm(x)) if mlp     Mixer = Attention | Mamba2 (hnet/ssm.py)
FLOPs: flops.layout_flops counts every layout (attention terms, or mamba2_flops in recurrent form, + 6 d h per MLP);
       the reference FLOPs path uses the same function
model.ssm: Mamba-2 settings (configs/model/base.yaml; the reference values d_state 128, d_conv 4, expand 2, headdim 64)
```

- `parse_layout` is the one parser: the model builder and both FLOPs counters use it.
- Old configs built the same model when this change landed (since then the [init rule fix](../bug-fix/2026-11-02-init-rule-and-partial-rope.md) changes `residual_proj.bias` for every model). The T-only blog split from the new code is bit-equal to the committed code in parameters, logits, chunk counts, FLOPs/byte, and state_dict key order (check in [paper LR in our H-Net](../../../../lab/experiments/refuted/2026-11-02-paper-lr-in-our-hnet.md), amendment). Old checkpoints load.
- Mamba-2 is mamba_ssm's `Mamba2`, the module the reference uses. `hnet/ssm.py` imports it lazily, so T-only arms do not need mamba_ssm. A stub stands in for the Mamba-1 CUDA module that `mamba_ssm/__init__` imports; any Mamba-1 call raises.
- `compile_blocks` compiles attention blocks whole and only the MLP of Mamba-2 blocks (Mamba-2 runs fused Triton kernels).
- If a stage sets both `enc_layers` and `enc_layout`, the layout wins. Experiment configs that inherit the blog split keep `enc_layers` and add a layout; `describe()` and wandb log the layout that was built.

## Alternatives considered

- **Use the reference model (`model.kind: ref_hnet`) as our 2-stage design.** That remains possible, but it changes everything at once (router, dechunk, residual, init, norms) and runs at 113 ms/step against 55. The layout swap changes one thing in our model and keeps our probes and compile path.
- **A per-stage `mixer: mamba2` flag.** Simpler, but it cannot express mixed stacks such as the reference's `T1m4`, and a later port of the reference layout would need a second mechanism.
- **Write our own Mamba-2 (SSD) kernels.** Rejected for now: mamba_ssm's Triton kernels run on sm_120 with CUDA 13, and they are the kernels the reference was trained with. A port would add a second unknown to the comparison.

## Consequences

- The spike venv needs mamba_ssm @ a6a1dae (MAMBA_SKIP_CUDA_BUILD) and causal_conv1d built for sm_120 (recipe: `lab/spikes/2026-11-01-reference-hnet-probe/README.md`), plus transformers below 4.50 and tokenizers below 0.22 for mamba_ssm's imports.
- `configs/model/base.yaml` has a `model.ssm` key, so every run's wandb config now records it, Mamba or not.
- Tests: `tests/test_layouts.py`. It covers the layout grammar, FLOPs equal to the old counter for T layouts, an unchanged state_dict for explicit T layouts, and, with a GPU and mamba_ssm, Mamba-2 causality, right-padding safety, and finite gradients that reach every Mamba-2 parameter.
- First use: [our H-Net with Mamba-2](../../../../lab/experiments/refuted/2026-11-02-our-hnet-with-mamba2.md).
