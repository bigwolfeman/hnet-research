"""Chunk selection, EMA smoothing (parallel scan), STE, ratio loss.

Pseudocode:
    select_chunks(mask):                      # mask (B,L) bool, True = anchor position
        counts  = mask.sum(1)
        M       = min(L, ceil(max(counts) / multiple) * multiple)   # bounded shape set
        idx     = positions of True entries in order, first M, per row
        valid   = arange(M) < counts
    ema_scan(x, p):                           # out_j = p_j x_j + (1 - p_j) out_{j-1}
        (a_j, b_j) = (1 - p_j, p_j x_j)       # affine maps y -> a y + b
        Hillis-Steele: for s in 1, 2, 4, ...: compose element j with element j-s
        log2(M) steps, all in fp32, no loop over the sequence
    ratio_loss(p, mask, N):                   # 2025 formula
        F = mean(mask), G = mean(p)
        N/(N-1) * ((1-F)(1-G) + F*G*(N-1))    # gradient flows through G only: d/dG = N/(N-1) * (N F - 1)
    RatioController (optional, train.ratio_integral_lr > 0):  # PI control: pin each stage's ratio to its target
        P term = the fixed-weight 2025 ratio loss above
        I term : loss += lam_s * G_s ;  lam_s += lr * (N_s F_s - 1)   # signed; on device, no host sync
Math copied from goombalab/hnet (modules/dc.py, utils/train.py, models/hnet.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

P_CLAMP = 1e-4  # 2025 DeChunkLayer clamps p to [1e-4, 1 - 1e-4]


def chunk_width(counts: torch.Tensor, L: int, multiple: int) -> int:
    """Padded anchor width M. M >= min(L, multiple) so an all-empty batch stays valid."""
    return chunk_stats(counts, L, multiple)[0]


def chunk_stats(counts: torch.Tensor, L: int, multiple: int) -> tuple[int, int, int]:
    """(M, sum(counts), max(counts)) in the ONE device-to-host sync of a stage. The sum sizes a packed inner level."""
    m_max, total = (int(v) for v in torch.stack([counts.max(), counts.sum()]).tolist())
    return min(L, max(multiple, -(-m_max // multiple) * multiple)), total, m_max


def chunk_index(mask: torch.Tensor, counts: torch.Tensor, M: int):
    """Return (idx (B,M), valid (B,M)). idx lists anchor positions in order; filler entries are 0 (masked by valid).

    Same result as a stable sort of anchors first, built with a cumsum and a scatter: O(L), no sort.
    """
    B, L = mask.shape
    rank = torch.cumsum(mask, dim=1) - 1
    slot = torch.where(mask, rank, torch.full_like(rank, M))  # non-anchors land in the dump column M
    src = torch.arange(L, device=mask.device)[None, :].expand(B, L)
    idx = torch.zeros(B, M + 1, dtype=torch.long, device=mask.device).scatter(1, slot, src)[:, :M]
    valid = torch.arange(M, device=mask.device)[None, :] < counts[:, None]
    return idx, valid


def select_chunks(mask: torch.Tensor, multiple: int):
    """Return (idx (B,M), counts (B,), valid (B,M)). Costs one host sync for M."""
    counts = mask.sum(dim=1)
    M = chunk_width(counts, mask.shape[1], multiple)
    idx, valid = chunk_index(mask, counts, M)
    return idx, counts, valid


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x (B,L,...) gathered along dim 1 by idx (B,M)."""
    if x.dim() == 2:
        return torch.gather(x, 1, idx)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


SCAN_CHUNK = 16  # intra-chunk Hillis-Steele depth is log2(SCAN_CHUNK); the carry scan runs on M / SCAN_CHUNK rows


def _hillis_steele(a: torch.Tensor, b: torch.Tensor, dim: int):
    """Inclusive scan of affine maps y -> a y + b along `dim`, log2(n) compose steps."""
    n = a.shape[dim]
    step = 1
    while step < n:
        a_prev = torch.narrow(a, dim, 0, n - step)
        b_prev = torch.narrow(b, dim, 0, n - step)
        pad = [0, 0] * (a.dim() - 1 - dim)
        a_prev = F.pad(a_prev, pad + [step, 0], value=1.0)
        b_prev = F.pad(b_prev, pad + [step, 0], value=0.0)
        b = b + a * b_prev
        a = a * a_prev
        step *= 2
    return a, b


def ema_scan(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """out_j = p_j * x_j + (1 - p_j) * out_{j-1}, out_{-1} = 0. x (B,M,D) fp32, p (B,M) fp32.

    Two-level scan: Hillis-Steele inside chunks of SCAN_CHUNK, then over the chunk carries, then one fused fix-up.
    Reads/writes the (B,M,D) tensor log2(SCAN_CHUNK)+1 times instead of log2(M) times. Same math, fp32 reorder only.
    """
    p = p.float().clamp(P_CLAMP, 1 - P_CLAMP).unsqueeze(-1)
    a = 1.0 - p
    b = p * x.float()
    B, M, D = b.shape
    C = SCAN_CHUNK
    if M <= C:
        return _hillis_steele(a, b, 1)[1]
    pad = (-M) % C
    if pad:  # identity maps at the tail
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
        b = F.pad(b, (0, 0, 0, pad), value=0.0)
    N = (M + pad) // C
    a, b = _hillis_steele(a.reshape(B, N, C, 1), b.reshape(B, N, C, D), 2)
    ca, cb = _hillis_steele(a[:, :, -1], b[:, :, -1], 1)  # inclusive carries at the end of each chunk, (B,N,1)/(B,N,D)
    carry = F.pad(cb[:, :-1], (0, 0, 1, 0))               # out at the end of the previous chunk, 0 for chunk 0
    out = (b + a * carry.unsqueeze(2)).reshape(B, N * C, D)
    return out[:, :M]


def ema_naive(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Reference loop used only by tests."""
    p = p.float().clamp(P_CLAMP, 1 - P_CLAMP)
    out, prev = [], torch.zeros_like(x[:, 0].float())
    for j in range(x.shape[1]):
        prev = p[:, j, None] * x[:, j].float() + (1 - p[:, j, None]) * prev
        out.append(prev)
    return torch.stack(out, dim=1)


class _STE(torch.autograd.Function):
    """Forward: ones. Backward: identity (2025 models/hnet.py)."""

    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, g):
        return g


def ste(x):
    return _STE.apply(x)


def ratio_loss(p: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor, N: float):
    """2025 load-balancing loss over valid positions. Returns (loss, F, G): F = anchor fraction (detached),
    G = mean boundary prob (with grad; the RatioController's integral term acts on it)."""
    v = valid.float()
    n = v.sum().clamp_min(1.0)
    frac = (mask.float() * v).sum() / n
    g = (p.float() * v).sum() / n
    loss = ((1 - frac) * (1 - g) + frac * g * (N - 1)) * N / (N - 1)
    return loss, frac.detach(), g


class RatioController:
    """Integral control of each stage's ratio: a signed multiplier lam_s on the mean boundary prob G_s.

    The fixed 2025 ratio loss (weight 0.03) is a proportional term: its gradient on G is w N/(N-1) (N F - 1), so
    the LM loss holds the ratio at an offset, of a size that depends on the architecture (target 4x: 3.1x in the
    lookahead 2-stage, 3.8x with no stage-1 blocks). Arms that differ in ratio differ in FLOPs/byte. The integral
    term removes the offset in both directions: N F > 1 (chunks too small) -> lam up -> G down, and the reverse.
    Experiment: lab/experiments/refuted/2026-11-01-compute-ladder-vs-bpe.md (amendment 2026-11-01 03:00).
    Note: .agents/notes/implemented/feature/2026-10-31-ratio-controller.md (why not dual ascent on the weight).
    """

    def __init__(self, targets: list[float], lr: float, device):
        assert lr > 0 and len(targets) > 0 and all(n > 1 for n in targets), (targets, lr)
        self.lr = lr
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)
        self.lam = torch.zeros(len(targets), dtype=torch.float32, device=device)

    def update(self, fracs: list[torch.Tensor]):
        """fracs: detached anchor fraction per stage from this step's forward."""
        self.lam += self.lr * (torch.stack(fracs).float() * self.targets - 1)

    def state_dict(self) -> dict:
        return {"lam": self.lam.clone()}

    def load_state_dict(self, sd: dict):
        self.lam.copy_(sd["lam"])

