"""The exact coarse-probe batches our training used, in plain numpy (no package import, so both venvs share it).

Pseudocode (= ByteLoader(seq_len, batch).fixed(n_eval, seed 1234)[:n_probe] in the spike's hnet/data.py):
    rng = default_rng(1234)
    batch k: offs = rng.integers(0, len(mm) - seq_len - 1, batch) ; rows = mm[o : o + seq_len + 1]
    x = rows[:, :-1], y = rows[:, 1:]
"""
from __future__ import annotations

import numpy as np

EDU_VAL = "ignored/data/fineweb-edu/val.bin"   # relative to the repo root


def probe_batches(path, seq_len: int = 4096, batch: int = 16, n: int = 8, seed: int = 1234) -> list[np.ndarray]:
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    return [np.stack([mm[o: o + seq_len + 1] for o in rng.integers(0, len(mm) - seq_len - 1, batch)]).astype(np.int64)
            for _ in range(n)]
