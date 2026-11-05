"""Mamba-2 mixer for layouts with m/M blocks (hnet/blocks.py). Wraps mamba_ssm's Mamba2 (Triton SSD kernels and
causal_conv1d), the same module the reference H-Net uses (goombalab/hnet @ 3673fe1, mamba_ssm @ a6a1dae).

Environment: mamba_ssm installed with MAMBA_SKIP_CUDA_BUILD, and causal_conv1d built for sm_120. The recipe is in
lab/spikes/2026-11-01-reference-hnet-probe/README.md. mamba_ssm/__init__ imports the Mamba-1 CUDA module
`selective_scan_cuda` unconditionally; H-Net uses only Mamba-2, so a stub that fails loudly stands in for it.

Pseudocode:
    mamba2(d, ssm) -> Mamba2(d_model=d, d_state, d_conv, expand, headdim, chunk_size)   # input and output (B, L, d)
"""
from __future__ import annotations

import sys
import types


def _stub_selective_scan_cuda():
    if "selective_scan_cuda" in sys.modules:
        return
    try:
        import selective_scan_cuda  # noqa: F401  a real build, or the reference probe shim on sys.path
        return
    except ImportError:
        pass
    stub = types.ModuleType("selective_scan_cuda")

    def __getattr__(name):
        if name.startswith("__"):   # module machinery (__file__, __path__, ...) probes; answer "no such attribute"
            raise AttributeError(name)
        raise NotImplementedError(f"selective_scan_cuda.{name}: Mamba-1 CUDA kernels are not built; H-Net uses Mamba-2")

    stub.__getattr__ = __getattr__
    sys.modules["selective_scan_cuda"] = stub


def mamba2(d: int, ssm: dict):
    _stub_selective_scan_cuda()
    from mamba_ssm.modules.mamba2 import Mamba2

    keys = ("d_state", "d_conv", "expand", "headdim", "chunk_size")
    missing = [k for k in keys if k not in ssm]
    if missing:
        raise ValueError(f"model.ssm is missing {missing}")
    return Mamba2(d_model=d, **{k: int(ssm[k]) for k in keys})
