"""Does the looped main network's output depend on the document? Measured after each pass, on CPU.

    nice -n 19 <spike>/.venv/bin/python loop_passes.py <run_dir> [...] > loop-passes.txt

Pseudocode:
    h_0 = main input (stage-0 chunk vectors); h_r = stack(h_{r-1}) for r = 1..R  (Main.forward, recorded)
    at each main position j < min row length:  doc share = sum_d Var_rows(h_r[:, j, d]) / mean_rows ||h_r[:, j]||^2
    mean over j.  Anchor 0: the vector at position j is the same in every row (no document content).
"""
import sys
from pathlib import Path

import torch

SPIKE = Path(__file__).resolve().parents[4] / "lab/spikes/2026-10-30-two-stage-hnet-smoke"
sys.path.insert(0, str(SPIKE))
from hnet.data import ByteLoader  # noqa: E402
from hnet.model import build_model  # noqa: E402
from hnet.stage import Main  # noqa: E402

DATA = Path("~/data/hnet-bytes/fineweb-edu/val.bin").expanduser()


def doc_share(h: torch.Tensor, lengths: torch.Tensor) -> float:
    n = int(lengths.min())
    x = h[:, :n].float()                                     # rows, positions, d
    var = x.var(dim=0, unbiased=False).sum(-1)               # per position
    return float((var / x.pow(2).sum(-1).mean(0)).mean())


def main():
    torch.set_num_threads(2)
    for run in map(Path, sys.argv[1:]):
        ckpt = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        mc = dict(cfg["model"]); mc["attn_impl"] = "sdpa_mask"   # FlexAttention has no CPU path
        model = build_model(mc)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
        model.eval()
        main_mod = next(m for m in model.modules() if isinstance(m, Main))
        shares = []

        def fwd(h, lengths, pos):
            shares.append(doc_share(h, lengths))
            for _ in range(main_mod.loops):
                h = main_mod.stack(h)
                shares.append(doc_share(h, lengths))
            return h, []

        main_mod.forward = fwd
        tc = cfg["train"]
        x = ByteLoader(str(DATA), tc["seq_len"], 8, cfg["seed"], torch.device("cpu")).fixed(1)[0]
        with torch.no_grad():
            model(x[0] if isinstance(x, (tuple, list)) else x)
        cells = "  ".join(f"h{r}={s:.4f}" for r, s in enumerate(shares))
        print(f"{cfg['wandb']['name']:16s} loops={main_mod.loops}  doc share  {cells}", flush=True)


if __name__ == "__main__":
    main()
