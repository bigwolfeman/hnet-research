"""Stub for mamba_ssm's Mamba-1 CUDA extension. mamba_ssm/__init__ imports it unconditionally, but H-Net uses only
Mamba-2 (Triton SSD kernels + causal_conv1d). Any Mamba-1 call fails here, loudly."""


def __getattr__(name):
    raise NotImplementedError(f"selective_scan_cuda.{name}: Mamba-1 CUDA kernels are not built in this harness")
