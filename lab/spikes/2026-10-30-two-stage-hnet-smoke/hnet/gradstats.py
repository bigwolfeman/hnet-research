"""Per-level gradient and AdamW-update statistics: how much, and how consistent, is the signal each level gets?

Experiment: lab/experiments/refuted/2026-11-03-main-gradient-coherence.md

AdamW makes the update size of a parameter almost independent of its gradient scale. What it does not hide is the
gradient's consistency over steps: with m = EMA_b1(g) and v = EMA_b2(g^2), the bias-corrected update
u = m_hat / (sqrt(v_hat) + eps) has rms near 1 when g keeps its sign, and near sqrt((1 - b1) / (1 + b1)) (0.229 at
b1 = 0.9) when g is zero-mean noise. So rms(u) per level says whether a level is learning from a signal or
random-walking. The floor assumes a constant gradient scale. While the scale falls (early training), v remembers
larger past gradients and rms(u) reads below the floor, so compare levels and seeds with each other, not with it.

Pseudocode:
    levels = {level: params} from the optimizer's param groups (names "stage{L}_decay", "stage{L}_nodecay", "stage{L}_extra")
    grad_norms(levels)      before clipping:  ||g||_2 over each level's params
    update_rms(opt, levels) after opt.step(): sqrt(mean over the level's elements of u^2), u from the AdamW state
"""
from __future__ import annotations

import math
import re

import torch

NOISE_FLOOR_B1_09 = math.sqrt(0.1 / 1.9)   # rms(u) for zero-mean iid gradients at b1 = 0.9 (large-step limit)
_LEVEL = re.compile(r"^stage(\d+)_")


def levels(opt: torch.optim.Optimizer) -> dict[int, list[tuple[torch.nn.Parameter, dict]]]:
    """Level -> [(param, its param group)] from the group names that train.make_optimizer sets."""
    out: dict[int, list] = {}
    for g in opt.param_groups:
        m = _LEVEL.match(g["name"])
        assert m, f"param group {g['name']!r} has no stage level"
        out.setdefault(int(m.group(1)), []).extend((p, g) for p in g["params"])
    return out


@torch.no_grad()
def grad_norms(lv: dict[int, list]) -> dict[int, torch.Tensor]:
    """L2 norm of the current .grad per level (fp32, on device; no host sync)."""
    res = {}
    for L, ps in lv.items():
        gs = [p.grad.float().square().sum() for p, _ in ps if p.grad is not None]
        res[L] = torch.stack(gs).sum().sqrt() if gs else torch.zeros(())
    return res


@torch.no_grad()
def update_rms(opt: torch.optim.Optimizer, lv: dict[int, list]) -> dict[int, torch.Tensor]:
    """rms of the bias-corrected AdamW direction m_hat / (sqrt(v_hat) + eps) per level, from the optimizer state."""
    res = {}
    for L, ps in lv.items():
        sq, n = [], 0
        for p, g in ps:
            st = opt.state.get(p)
            if not st:
                continue
            b1, b2 = g["betas"]
            t = float(st["step"])
            m_hat = st["exp_avg"].float() / (1 - b1 ** t)
            v_hat = st["exp_avg_sq"].float() / (1 - b2 ** t)
            sq.append((m_hat / (v_hat.sqrt() + g["eps"])).square().sum())
            n += p.numel()
        res[L] = (torch.stack(sq).sum() / max(n, 1)).sqrt() if sq else torch.zeros(())
    return res
