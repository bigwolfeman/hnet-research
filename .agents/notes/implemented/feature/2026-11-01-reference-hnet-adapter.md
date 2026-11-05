# Agent Note: Reference H-Net adapter in our trainer

Status: implemented

## Problem

Our 2-stage H-Net never uses its main network. The released reference model (`hnet_2stage_L`) uses it heavily ([reference checkpoint main-network use](../../../../lab/experiments/successes/2026-11-01-reference-checkpoint-main-use.md)). To split "scale" from "our model and recipe", the reference code must train at our scale, with our data, batch, schedule, evals, and probes. goombalab/hnet ships model code only: no trainer, no data pipeline. It depends on flash-attn (no build for torch 2.14 + CUDA 13 + sm_120) and on mamba_ssm and causal_conv1d. Its package is also named `hnet`, the same as ours.

## Decision

`model.kind: ref_hnet` builds `RefHNetLM` (`lab/spikes/2026-10-30-two-stage-hnet-smoke/hnet/refmodel.py`). It wraps the reference `HNetForCausalLM` behind the `HNetLM` interface, so `train.py`, the evals, the probes, the WSD ladder, and resume run unchanged.

```
build_model(cfg): kind ref_hnet -> RefHNetLM(cfg)  (imported lazily: only that kind needs the reference environment)
RefHNetLM.forward(x, lengths) -> LMOutput(logits, [StageAux per stage])
    StageAux from the reference router outputs: counts, frac, mean boundary prob, byte positions of start anchors,
    ratio loss = the reference load_balancing_loss formula over valid positions
stages() -> handles; handle.ablate = zero | shuffle patches that level's residual_func (the probe's coarse path)
stage_param_groups() -> the reference _apply_lr_multiplier ownership (depth-s HNet -> s, deeper overrides)
flops_per_byte(kind ref_hnet) -> reference layouts, Mamba-2 counted in recurrent form
```

- The reference package is vendored as `ignored/ref/refnet` by `lab/spikes/2026-11-01-reference-hnet-probe/vendor_refnet.py`. That script checks the pinned commit (3673fe1) and only rewrites the absolute `hnet.` imports.
- flash-attn is replaced by a torch shim (`lab/spikes/2026-11-01-reference-hnet-probe/shims/`), tested against float32 math.
- The environment is that spike's venv (mamba_ssm without its Mamba-1 CUDA build, causal_conv1d patched for sm_120, transformers below 4.50, tokenizers 0.21). Run with `PY=<that venv>/bin/python scripts/run_arms.sh ...`.

Weight transplant (added 2026-11-02, for [reference init in our code](../../../../lab/experiments/refuted/2026-11-02-reference-init-in-our-code.md)):

```
ref_state_to_ours(ref, ours) -> our HNetLM state dict holding the reference weights, level by level (hnet/refmodel.py);
    SwiGLU fc1 [value | gate] -> gate_up [gate | value]; strict coverage, type and shape checks
scripts/ref_init_transplant.py: seed -> the reference arm's step-0 weights in our layout, with forward checks
train.init_from: load such a file strictly before training (train.py); path and sha256 go into the run config
```

## Alternatives considered

- **A standalone trainer for the reference model.** Rejected: it would duplicate data loading, evals, probes, the ladder, and resume. Any difference between the two trainers would confound the comparison.
- **Load the reference package under an alias at import time** (rename the `sys.modules` entries). Rejected: it depends on import order and breaks on any function-level import. The generated copy is explicit and checked.
- **Build flash-attn from source.** Rejected: there is no sm_120 + CUDA 13 support at the pinned version, the build is long and CPU-heavy (power budget), and the constitution keeps flash-attn out of the project.
- **Port Mamba-2 outer stages into our own Stage code first.** Deferred until the reference run says whether the reference design fixes the main network. Porting first would mix two changes: the architecture and our implementation of it.

- **Transplant by matching parameter names with string rewrites.** Rejected: the two trees nest differently (the reference keeps `pad_dimension` on the child level and names its stacks `layers`), and a rename table hides type mismatches. The level-by-level walk checks block types and MLP presence.

## Consequences

- The reference model runs at 102 ms/step against 52 ms/step for our blog split at similar FLOPs/byte (smoke run, eager, no torch.compile). Mamba-2 at d 256 is launch-bound. Speed work waits until the design is chosen.
- `RefHNetLM` requires bf16 autocast (the reference attention asserts bf16). Training and eval already use it.
- Tests: `tests/test_refmodel.py` (reference venv and GPU; skipped elsewhere). It covers aux consistency, the ratio loss against the reference `load_balancing_loss`, causality, probe ablation, parameter groups, and a training step.
