# Spike: reference H-Net probe harness

Runs the released goombalab/hnet checkpoints (and later the reference training code) on our data and our probes, on the RTX 5090. Used by [reference checkpoint main-network use](../../experiments/successes/2026-11-01-reference-checkpoint-main-use.md).

```
probe_ref.py  (this venv)          probe_ours.py  (two-stage spike venv)
  load_ref(name)  bf16               HNetLM(final.pt config)  eager, bf16 autocast
        \                                    /
         probe_data.probe_batches(edu val)   # the exact training probe batches (seed 1234, first 8 of 16)
         per stage s: rise[s, zero|shuffle] = bpb(coarse path ablated) - bpb
                      share[s] = mean ||coarse path|| / ||residual|| at the decoder input
         -> JSON in lab/experiments/results/<slug>/
```

Two processes, because both code bases name their package `hnet`. `probe_data.py` is plain numpy, so both import it.

## Files

| File | Role |
|---|---|
| `refhnet.py` | sys.path setup (shims first), reference config and checkpoint loading, byte mapping (our doc separator 0 -> reference BOS 254), stage list |
| `probe_ref.py` | Probe for reference checkpoints. Ablates by wrapping each stage's `residual_func` |
| `probe_ours.py` | Probe for our checkpoints. Calls the training probe (`probes.coarse_use`) and records the share with a `_scatter_out` replica that is asserted equal to the original |
| `probe_data.py` | The training probe batches, reimplemented in numpy |
| `shims/flash_attn/` | Torch versions of the flash-attn 2.8 calls the reference uses (attention via SDPA with flash-attn's window and bottom-right causal rules, fused-add RMSNorm, rotary, SwiGLU). kv-cache decoding raises |
| `shims/selective_scan_cuda.py` | Stub for the Mamba-1 CUDA module that `mamba_ssm/__init__` imports. H-Net uses Mamba-2 only |
| `tests/test_shims.py` | Shim vs direct float32 math |

## Environment (`.venv`, not committed)

```
uv venv --python 3.12 .venv
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install einops optree regex omegaconf numpy wandb hydra-core pytest "transformers>=4.44,<4.50"
MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-build-isolation "mamba_ssm @ git+https://github.com/state-spaces/mamba.git@a6a1dae"
# causal-conv1d @ e940ead: its setup.py hard-codes sm_53..sm_100, and CUDA 13 rejects sm_53. Local patch in
# ignored/ref/causal-conv1d/setup.py builds only CC1D_ARCHS (default 120). Build with the GPU idle (CPU load, power budget).
PATH=/opt/cuda/bin:$PATH CUDA_HOME=/opt/cuda MAX_JOBS=4 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    uv pip install --no-build-isolation ignored/ref/causal-conv1d
```

- transformers is pinned below 4.50 because `mamba_ssm.utils.generation` imports names that transformers 5 removed.
- Reference clone: `ignored/ref/hnet` at 3673fe1. Checkpoints: `ignored/ref/ckpt/` (`hnet_2stage_L.pt` sha256 5711de80..., `hnet_1stage_L.pt` b92e051c...).

## Run

```
.venv/bin/python -m pytest -q tests
.venv/bin/python probe_ref.py hnet_2stage_L <out.json>
../2026-10-30-two-stage-hnet-smoke/.venv/bin/python probe_ours.py <final.pt> <out.json>
```
