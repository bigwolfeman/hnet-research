"""Entropy thresholds for frozen boundaries (hnet/boundaries.py) that hit target ratios on the training mix's val data.

    python scripts/calibrate_entropy.py <proxy final.pt> --ratios 3 3 [--rows 256] [--json out.json]

Pseudocode:
    rows  = val rows drawn as the trainer draws train rows: per row, a source by weight (data.train_sources), then a
            random offset; seq_len bytes each (seed fixed)
    H     = proxy next-byte entropy per byte (EntropyBoundaries.entropy)
    theta = boundaries.calibrate(H, ratios)       # stage s: 1 / prod(ratios[:s+1]) boundaries per byte, pooled
    print theta (for model.boundaries.thresholds) and the measured bytes per chunk per source and stage
Experiment: lab/experiments/planned/2026-11-05-learned-vs-entropy-boundaries.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet.boundaries import EntropyBoundaries, calibrate, load_proxy  # noqa: E402


def mix_rows(data_cfg: dict, data_dir: Path, n: int, L: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Val rows in the train mix: source by weight, random offset (as MixLoader draws train rows)."""
    val = {d["name"]: d["file"] for d in data_cfg["val_sets"]}
    srcs = [(s["name"], float(s["weight"])) for s in data_cfg["train_sources"]]
    w = np.array([x[1] for x in srcs])
    rng = np.random.default_rng(seed)
    mm = {name: np.memmap(data_dir / val[name], dtype=np.uint8, mode="r") for name, _ in srcs}
    out = []
    for k in rng.choice(len(srcs), size=n, p=w / w.sum()):
        name = srcs[k][0]
        o = int(rng.integers(0, len(mm[name]) - L))
        out.append((name, np.asarray(mm[name][o:o + L])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proxy", type=Path)
    ap.add_argument("--ratios", type=float, nargs="+", required=True, help="per-stage target ratio, e.g. 3 3")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/data/bytes.yaml")
    ap.add_argument("--data-dir", type=Path, default=None, help="default: data.dir of the data config")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = OmegaConf.to_container(OmegaConf.load(a.data_config), resolve=True)   # resolves ${oc.env:HNET_DATA_DIR,...}
    data_dir = a.data_dir or Path(dcfg["dir"])
    proxy, src = load_proxy(a.proxy, device)
    eb = EntropyBoundaries(proxy, [], src)
    rows = mix_rows(dcfg, data_dir, a.rows, a.len, a.seed)
    ents = []
    for b in range(0, len(rows), 16):
        x = torch.from_numpy(np.stack([r for _, r in rows[b:b + 16]]).astype(np.int64)).to(device)
        ents += list(eb.entropy(x, torch.full((x.shape[0],), a.len, device=device)).cpu().numpy())
    th = calibrate(ents, a.ratios)
    res = {"thresholds": th, "ratios": a.ratios, "rows": a.rows, "len": a.len, "seed": a.seed, "proxy": src}
    for s, t in enumerate(th):
        for name in sorted({n for n, _ in rows}):
            e = [h for (n, _), h in zip(rows, ents) if n == name]
            res[f"bytes_per_chunk_s{s}_{name}"] = float(sum(len(h) for h in e) / sum(int((h > t).sum()) for h in e))
    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
