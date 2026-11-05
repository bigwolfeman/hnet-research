# Experiment: reference init in our code

Status: refuted

Date proposed: 2026-11-02
Algorithm: none (step-0 weights copied from the reference model; pseudocode in Method)
Owner: Claude session (hnet-hierarchy vlt thread)

## Question

Is the initialization the implementation difference that makes our code's main network seed-fragile?

Context: with the reference configuration, our code gave a half-live main network for seed 0 and a dead one for seed 1 (stage-1 shuffle − zero −0.020 and −0.061). The reference code gave live ones for both seeds (+0.021, +0.031). The split shows at the first eval ([reference layout in our code](../refuted/2026-11-02-reference-layout-in-our-code.md)). A per-tensor comparison of the step-0 weights found two systematic differences.

- **Output-projection scale.** Both codes scale `out_proj` and `fc2` by 1/sqrt(n_residuals), counted across levels. The reference counts one residual for a block with no MLP and two for a block with an MLP. Ours counts two for every block. With the reference layout, our stage-0 projections start 1.41x smaller, stage 1 1.36x, and main 1.18x. For all-attention layouts, such as the blog split, the two counts agree.
- **Residual bias.** The reference zeroes `residual_proj.weight` and keeps PyTorch's default `residual_proj.bias` (std 0.036). We zero both.

The other tensors match in distribution.

## Hypothesis

The difference is the initialization. Smaller output projections in the Mamba-2 outer stages make the encoder's early updates weaker, so the decoder settles on its own residual before the main network's input carries content. The reference's init avoids this in both seeds.

## Predictions

Main prediction (only this one answers the Question):

- Our code, started from the reference model's exact step-0 weights, ends at 12,000 steps with the live stage-1 signature (`probe/coarse_shuffle_s1` above `probe/coarse_zero_s1`) in both seeds, 0 and 1. Both above: supported. Either at or below: not supported.

Other predictions (at most two):

- At step 2,000, seed 1 of the transplant arm has `coarse_shuffle_s0` above our fresh-init seed 1's (0.058), the early split that F showed.
- Both transplant seeds' edu bpb at the end is within the seed spread (0.059) of the reference arm's for the same seed (1.3178, 1.3185).

If supported, we do: the cause is initialization. Fix our init to the reference rule (per-block residual count; decide the residual bias with a test of its own), confirm with fresh init on two seeds, then bisect the configuration from the reference layout toward the blog split.

If not supported, we do: the cause is in the forward pass or the training step, not in the init. The next candidates are partial RoPE (the reference rotates 32 of 64 head dims), the order of norm and residual dtypes (the reference adds residuals in fp32 inside a fused norm), and torch.compile of our attention blocks.

Gates (the run is broken if one fails; gates are not predictions):

- The transplant files load strictly into our model, and their checks pass: our SwiGLU on mapped weights equals the reference's (max error 0.0), and the stage-0 encoder output (Mamba-2 only) agrees with the reference's to 0.015 and the router probabilities to 0.0022 (bf16, both seeds). The mapping test (`tests/test_refmodel.py::test_ref_state_maps_into_our_model`) passes.
- Edu ratios within 15% of 3 x 3 at the end; edu fwd FLOPs/byte within 5% of 22.27M. Bytes 786,432,000 per arm. Finite loss.

## Method

```
init files (scripts/ref_init_transplant.py, reference venv):
    torch.manual_seed(s); ref = build_model(ref_hnet_2stage)        # the reference arm's step-0 weights for seed s
    sd = ref_state_to_ours(ref, build_model(reflayout_2stage))      # hnet/refmodel.py
    -> ignored/experiment-artifacts/ref-init-transplant/ref_init_seed{0,1}.pt
arms: reflayout_2stage + train.init_from=<that file>, seeds 0 and 1 (seed s also sets the data order, as in the
      reference arm of seed s)
everything else = the reference-layout arms: configuration, RatioController, schedule (12,000 steps, no branches),
      eval every 2,000 steps with the probe
controls (not rerun): reference code seeds 0 and 1; our code with fresh init seeds 0 and 1
grade: probe/coarse_shuffle_s1 - probe/coarse_zero_s1 at step 12,000, both seeds
```

- Launch, from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`, one `run_arms.sh` call per seed (each needs its own `init_from`): log dirs `ignored/runs/reference-init-in-our-code/s{0,1}`, wandb group `reference-init-in-our-code`, extra overrides `train.max_steps=12000 "train.ladder_fracs=[]" train.init_from=<file>`.
- `train.init_from` loads the file strictly after the model is built and logs its path, sha256, and source checks in the run config.
- Cost: about 20 min per arm.

## Results

Both arms ran to 12,000 steps (commit f4f49b7). Log dirs `ignored/runs/reference-init-in-our-code/s{0,1}`; wandb seed 0 `qjb455zw` (init sha256 0d96118e9e79), seed 1 `t1rw7f3p` (c99553c9b20d). Artifacts: [results/](../results/2026-11-02-reference-init-in-our-code/) (`seed*-summary.json`, `eval-curves.csv`, `init-files.md`).

### Gates

| gate | seed 0 | seed 1 | result |
|---|---|---|---|
| init files load strictly; mapping checks and mapping test | SwiGLU 0.0, encoder 0.0145, router 0.0022 | 0.0, 0.0154, 0.0022 | pass |
| edu ratios at the end (3 x 3 ±15%) | 3.06 x 2.86 | 3.03 x 2.96 | pass |
| edu fwd FLOPs/byte (22.27M ±5%) | 22.68M | 22.58M | pass |
| training bytes, finite loss | 786,432,000, yes | same | pass |

### Coarse-path probe and bpb at the end (12,000 steps)

| arm | `shuffle_s1` | `zero_s1` | shuffle − zero | `shuffle_s0` | `zero_s0` | edu bpb | Go bpb |
|---|---:|---:|---:|---:|---:|---:|---:|
| our code, reference init, seed 0 | 0.190 | 0.141 | **+0.050** | 1.194 | 1.028 | **1.3105** | 0.888 |
| our code, reference init, seed 1 | 0.0815 | 0.0912 | **−0.010** | 0.902 | 0.923 | 1.3288 | 0.892 |
| our code, our init, seed 0 (F) | 0.0656 | 0.0856 | −0.020 | 1.040 | 1.041 | 1.3248 | 0.918 |
| our code, our init, seed 1 (F) | 0.0078 | 0.0690 | −0.061 | 0.427 | 0.888 | 1.3797 | 0.935 |
| reference code, seed 0 | 0.135 | 0.114 | +0.021 | 1.041 | 0.917 | 1.3178 | 0.884 |
| reference code, seed 1 | 0.132 | 0.101 | +0.031 | 1.048 | 1.021 | 1.3185 | 0.890 |

Stage-1 shuffle − zero by eval (steps 2,000 to 12,000): seed 0 +0.019, +0.030, +0.032, +0.037, +0.048, +0.050; seed 1 +0.0055, +0.0055, +0.0047, −0.0077, −0.0035, −0.0097.

- Main prediction (both seeds live at the end): seed 0 yes, seed 1 no (−0.010). **Not supported.**
- Other prediction 1 (seed 1 `shuffle_s0` at step 2,000 above our-init seed 1's 0.058): holds, 0.198.
- Other prediction 2 (edu bpb within 0.059 of the reference arm of the same seed): holds, −0.007 and +0.010.

### What the arms show

- **The reference init moves our code to the reference's results on every measure but one sign.** Seed 1's main-network use is 10x higher than with our init (0.082 against 0.0078). Its bpb is 0.051 better, 0.010 from the reference's. Its stage-0 balance is the reference's (0.90 against 0.92). Seed 0 is more live than the reference (+0.050 against +0.021) and 0.007 bpb better.
- **The two seeds now differ by 0.019 in bpb (they differed by 0.055 with our init), and both are inside the reference's spread band.** The init explains most of the seed fragility.
- **Seed 1 was live until step 6,000 and crossed below zero at step 8,000.** It ends at −0.010, a small value against the ±0.06 range the arms span. One forward difference remains between the codes: partial RoPE (the reference rotates 32 of 64 head dims; we rotate all 64), plus fp32 residual adds inside the reference's fused norm.

## Verdict

**Not supported.** From the reference's exact step-0 weights, our code gives a live stage-1 main network in seed 0 (+0.050) but not in seed 1 (−0.010). Initialization is most of the difference: the seed spread fell 3x, seed 1's main-network use rose 10x, and both seeds' bpb landed within 0.010 of the reference's. It is not all of it.

## Updated hypothesis

Two parts of our implementation differ from the reference and matter: the init and at least one forward detail. The init difference is a bug against our own stated intent (the 2025 init rule). We count two residuals for every block; the 2025 code counts one for a block with no MLP. We also zero `residual_proj.bias`; the 2025 code zeroes only the weight. The most likely remaining forward difference is partial RoPE. Next: fix both init details in our code, add partial RoPE (`rope_dim` 32 in the reference layout), and train with fresh init on two seeds: [faithful init and RoPE](../refuted/2026-11-02-faithful-init-and-rope.md).

## Related

- [Reference layout in our code](../refuted/2026-11-02-reference-layout-in-our-code.md)
- [Stack layouts note](../../../.agents/notes/implemented/feature/2026-11-02-stack-layouts-mamba2.md)
