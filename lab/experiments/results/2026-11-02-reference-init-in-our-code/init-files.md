# Pointer: transplant init files

Large binaries, not committed. Location: `ignored/experiment-artifacts/ref-init-transplant/`.

| file | built by | checks |
|---|---|---|
| `ref_init_seed0.pt` | `scripts/ref_init_transplant.py --ref ref_hnet_2stage --ours reflayout_2stage --seed 0` | SwiGLU 0.0, stage-0 encoder 0.0145, router 0.0022 |
| `ref_init_seed1.pt` | same, `--seed 1` | SwiGLU 0.0, stage-0 encoder 0.0154, router 0.0022 |

Rebuild with the reference venv from `lab/spikes/2026-10-30-two-stage-hnet-smoke/`. The sha256 of each file is in the run config (`init_from.sha256`).
