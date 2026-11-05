"""How much does each stage's decoder use the coarse path? Eval-time ablation on trained checkpoints.

    python scripts/ablate_coarse.py <run_dir_or_final.pt> [...] [--batches 16]

Pseudocode:
    batches = fixed edu val batches (the training eval batches)
    probes.coarse_use(model, batches):  for stage s, mode in {zero, shuffle}: bpb(stage s ablated) - bpb
    A small rise under shuffle means the network below stage s carries little document-specific information.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnet import probes  # noqa: E402
from hnet.data import ByteLoader  # noqa: E402
from sample import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--data-dir", type=Path, default=Path("~/data/hnet-bytes").expanduser())
    ap.add_argument("--batches", type=int, default=16)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in a.checkpoints:
        model, cfg = load(path, device)
        tc = cfg["train"]
        loader = ByteLoader(str(a.data_dir / "fineweb-edu/val.bin"), tc["seq_len"], tc["batch_size"], cfg["seed"], device)
        m = probes.coarse_use(model, loader.fixed(a.batches), device, torch.bfloat16)
        cells = [f"{k.removeprefix('probe/coarse_')} {v:+.3f}" for k, v in m.items() if k != "probe/coarse_base_bpb"]
        print(f"{cfg['wandb']['name']}: base {m['probe/coarse_base_bpb']:.4f}  " + "  ".join(cells), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
