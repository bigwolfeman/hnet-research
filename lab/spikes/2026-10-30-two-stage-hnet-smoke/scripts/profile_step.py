"""Steady-state step time, kernel-busy %, kernel-category split, peak memory, compile counts. GPU box only.

    python scripts/profile_step.py [hydra overrides...]         e.g. train.compile=true wandb.mode=disabled
Env: PROF_STEPS (timed steps, default 20), PROF_WARMUP (default 60), PROF_SYNC=1 (print every device-to-host sync
site found in 3 steps), PROF_TRACE=path.json (export a chrome trace).
"""
import collections
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

spike = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(spike))
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from hnet.data import MixLoader  # noqa: E402
from hnet.model import build_model, total_loss  # noqa: E402
from train import make_optimizer  # noqa: E402

with initialize_config_dir(str(spike / "configs"), version_base=None):
    cfg = compose("config", overrides=sys.argv[1:])
tc = cfg.train
dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")
model = build_model(OmegaConf.to_container(cfg.model, resolve=True)).to(dev)
if tc.compile:
    model.compile_stacks()
opt = make_optimizer(model, tc)
dd = Path(cfg.data.dir) if Path(cfg.data.dir).is_absolute() else spike / cfg.data.dir
src = [dict(name=d.name, file=str(dd / d.file), weight=d.weight) for d in cfg.data.train_sources]
ld = MixLoader(src, tc.seq_len, tc.batch_size, 0, dev, cfg.data.prefetch)


def step():
    x, y = ld.next()
    L = torch.full((x.shape[0],), x.shape[1], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, L)
    loss, _ = total_loss(out, y, L)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)


warm, n = int(os.environ.get("PROF_WARMUP", 60)), int(os.environ.get("PROF_STEPS", 20))
t_w = time.perf_counter()
for _ in range(warm):
    step()
torch.cuda.synchronize()
print(f"WARMUP {warm} steps {time.perf_counter() - t_w:.1f} s (includes compiles)")
t = time.perf_counter()
for _ in range(n):
    step()
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / n
print(f"STEADY step {dt * 1e3:.1f} ms  bytes/s {tc.batch_size * tc.seq_len / dt / 1e3:.0f}k  "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
try:
    from torch._dynamo.utils import counters
    print("COMPILES", {k: dict(v) for k, v in counters.items() if k in ("stats", "frames")})
except Exception as e:  # noqa: BLE001
    print("compile counters unavailable", e)

if os.environ.get("PROF_SYNC") == "1":
    sites = collections.Counter()
    orig = warnings.showwarning

    def grab(message, category, filename, lineno, file=None, line=None):
        if "synchronizing" in str(message):
            fr = [f for f in traceback.extract_stack() if "hnet" in f.filename or "train" in f.filename][-3:]
            sites[" <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(fr))] += 1
        else:
            orig(message, category, filename, lineno, file, line)

    warnings.showwarning = grab
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    for _ in range(3):
        step()
    torch.cuda.set_sync_debug_mode("default")
    print("SYNC SITES per 3 steps")
    for s, c in sites.most_common():
        print(f"  {c:3d}  {s}")

from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(5):
        step()
    torch.cuda.synchronize()
if os.environ.get("PROF_TRACE"):
    p.export_chrome_trace(os.environ["PROF_TRACE"])
k = [e for e in p.events() if e.device_type == DeviceType.CUDA]
tot = sum(e.device_time for e in k) / 5 / 1e3
print(f"KERNEL time per step {tot:.1f} ms of {dt * 1e3:.1f} ms wall ({100 * tot / dt / 1e3:.0f}% busy), {len(k) / 5:.0f} kernels/step")
cat = collections.Counter()
for e in k:
    nm = e.name.lower()
    c = ("gemm" if ("gemm" in nm or "cutlass_80_tensorop" in nm or "sm90_xmma" in nm or "cublas" in nm or "nvjet" in nm)
         else "attn_memeff_masked" if "fmha_cutlass" in nm else "attn_flash_main" if "flash" in nm
         else "attn_flex" if "flex" in nm else "triton_fused" if nm.startswith("triton")
         else "elementwise/copy" if ("elementwise" in nm or "copy" in nm) else "reduce" if "reduce" in nm else "other")
    cat[c] += e.device_time / 5 / 1e3
for c, v in cat.most_common():
    print(f"  {c:22s} {v:6.1f} ms  {100 * v / tot:4.0f}%")
oth = collections.Counter()
for e in k:
    nm = e.name.lower()
    if not any(s in nm for s in ("gemm", "cutlass_80_tensorop", "fmha", "flash", "flex", "triton", "elementwise", "copy", "reduce")):
        oth[e.name[:80]] += e.device_time / 5 / 1e3
for nm, v in oth.most_common(6):
    print(f"    other: {v:5.1f} ms  {nm}")
