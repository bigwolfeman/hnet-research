"""Write the reference H-Net's step-0 weights for one seed as a state dict of OUR model (for train.init_from).

Pseudocode:
    torch.manual_seed(seed); ref = build_model(ref experiment)      # the same RNG calls as train.py before training
    ours = build_model(our experiment)                              # same configuration in our code
    sd = ref_state_to_ours(ref, ours)                               # hnet/refmodel.py, strict coverage and shapes
    checks (GPU, bf16 autocast, the reference's own forward as the oracle):
        SwiGLU: our block MLP(x) == reference block MLP(x) after the [value | gate] -> [gate | value] swap
        stage 0: our encoder(embed(x)) == reference encoder(embeddings(x))      (Mamba-2 blocks, no attention)
        stage 0: our router boundary probs == reference routing_module probs
    save {"model": sd, "source": {...}}

usage (reference venv: refnet + mamba_ssm):
    python scripts/ref_init_transplant.py --ref ref_hnet_2stage --ours reflayout_2stage --seed 0 --out <file.pt>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from hnet.model import build_model  # noqa: E402
from hnet.refmodel import ref_state_to_ours  # noqa: E402


def model_cfg(exp: str) -> dict:
    with initialize_config_dir(config_dir=str(HERE / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={exp}", "train=ladder"])
    return OmegaConf.to_container(cfg.model, resolve=True)


@torch.no_grad()
def check(ref, ours, device):
    ref, ours = ref.to(device).eval(), ours.to(device).eval()
    g = torch.Generator().manual_seed(123)
    x = torch.randint(0, 256, (2, 1024), generator=g).to(device)
    mask = torch.ones_like(x, dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # SwiGLU of the first block that has one (stage-1 encoder of the reference layout)
        blk = next(b for st in ours.stages() for b in st.encoder.blocks if b.mlp is not None)
        lay = next(lay for h in ref._levels()[:-1] for lay in h.encoder.layers if getattr(lay, "mlp", None) is not None)
        h = torch.randn(2, 64, blk.mlp.fc2.out_features, device=device)
        a, b = blk.mlp(h).float(), lay.mlp(h).float()
        assert torch.allclose(a, b, atol=2e-2, rtol=2e-2), ("SwiGLU", (a - b).abs().max().item())
        e_ours = ours.stages()[0].encoder(ours.embed(x))
        e_ref = ref.ref.backbone.encoder(ref.ref.embeddings(x), mask=mask)
        err = (e_ours.float() - e_ref.float()).abs().max().item()
        assert err < 5e-2, ("stage-0 encoder", err)
        p_ours = ours.stages()[0].router(e_ours, torch.full((2,), 1024, device=device)).p
        p_ref = ref.ref.backbone.routing_module(e_ref, mask=mask).boundary_prob[..., -1]
        perr = (p_ours.float() - p_ref.float()).abs().max().item()
        assert perr < 2e-2, ("stage-0 router", perr)
    return dict(swiglu_max_err=(a - b).abs().max().item(), enc0_max_err=err, router0_max_err=perr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)                     # train.py: manual_seed(seed) then build_model, no RNG in between
    ref = build_model(model_cfg(a.ref))
    ours = build_model(model_cfg(a.ours))
    sd = ref_state_to_ours(ref, ours)
    ours.load_state_dict(sd, strict=True)
    errs = check(ref, ours, torch.device("cuda"))
    source = dict(ref_experiment=a.ref, ours_experiment=a.ours, seed=a.seed, checks=errs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=sd, source=source), a.out)
    print(source)


if __name__ == "__main__":
    main()
