"""Golden-value harness: a fixed batch through a fixed-seed model must keep the same loss, grads and routing.

Reference JSON files in tests/golden/ were produced by `scripts/make_golden.py` on the code BEFORE the throughput work.
  cpu_fp32.json : tiny model, fp32, CPU (tight tolerance; guards the chunk/EMA/glue math)
  gpu_bf16.json : smoke-size model (eoc_2stage), bf16 autocast, CUDA (bf16 tolerance; guards attention/compile changes)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
SPIKE = Path(__file__).resolve().parents[1]


def probe_bytes_batch(batch: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic real-text batch from data/probe_paragraphs.json (rows start at different offsets)."""
    paras = json.loads((SPIKE / "data" / "probe_paragraphs.json").read_text())
    blob = np.frombuffer(" ".join(p["text"] for p in paras).encode(), dtype=np.uint8)
    blob = np.tile(blob, -(-(batch * 997 + seq_len + 2) // len(blob)) + 1)
    rows = np.stack([blob[i * 997: i * 997 + seq_len + 1] for i in range(batch)])
    t = torch.from_numpy(rows.astype(np.int64))
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def smoke_cfg() -> dict:
    """The eoc_2stage smoke config as a plain dict (same composition train.py uses). The golden values pin eoc_2stage,
    so the experiment is named here and does not follow the default in configs/config.yaml."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(str(SPIKE / "configs"), version_base=None):
        cfg = compose("config", overrides=["experiment=eoc_2stage"])
    return OmegaConf.to_container(cfg.model, resolve=True)


def capture(model, x, y, device, amp: bool, lengths=None) -> dict:
    """Forward+backward once. Returns loss, ce, per-stage counts and anchor digests, per-param grad norms."""
    from hnet.model import total_loss

    model.zero_grad(set_to_none=True)
    x, y = x.to(device), y.to(device)
    if lengths is None:
        lengths = torch.full((x.shape[0],), x.shape[1], device=device)
    lengths = lengths.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(x, lengths)
    loss, ce = total_loss(out, y, lengths)
    loss.backward()
    stages = []
    for a in out.aux:
        valid = torch.arange(a.pos.shape[1], device=a.pos.device)[None, :] < a.counts[:, None]
        pos = torch.where(valid, a.pos, torch.full_like(a.pos, -1)).to(torch.int32)  # anchor byte positions, -1 = pad
        # compare only the valid prefix of each row so a changed padded width M cannot matter
        rows = [pos[b, : int(a.counts[b])].cpu().numpy().tobytes() for b in range(pos.shape[0])]
        stages.append(dict(counts=a.counts.cpu().tolist(),
                           anchors_sha1=hashlib.sha1(b"|".join(rows)).hexdigest(),
                           frac=float(a.frac), ratio_loss=float(a.ratio_loss)))
    return dict(loss=float(loss), ce=float(ce), stages=stages,
                grad_norms={n: float(p.grad.float().norm()) for n, p in model.named_parameters() if p.grad is not None},
                logits_mean=float(out.logits.float().mean()), logits_abs_mean=float(out.logits.float().abs().mean()))


def build_cpu():
    from hnet.model import HNetLM
    from tests.conftest import make_cfg, randomize

    torch.manual_seed(0)
    m = randomize(HNetLM(make_cfg("eoc", 2, multiple=32)), std=0.15, seed=0)
    x, y = probe_bytes_batch(3, 256)
    return m, x, y, torch.tensor([256, 200, 130])


def build_gpu(compile_: bool = False):
    from hnet.model import HNetLM

    torch.manual_seed(0)
    m = HNetLM(smoke_cfg())
    # The goldens were recorded with the old init, which zeroed residual_proj.bias. The 2025-faithful init keeps
    # PyTorch's default bias (same RNG draws), so zeroing it here rebuilds the recorded model exactly. This test
    # guards forward/backward numerics, not the init.
    with torch.no_grad():
        for st in m.stages():
            st.residual_proj.bias.zero_()
    m = m.cuda()
    if compile_:
        m.compile_stacks()
    x, y = probe_bytes_batch(4, 4096)
    return m, x, y


def load(name: str) -> dict:
    return json.loads((GOLDEN_DIR / name).read_text())
