"""Stage-1 interface diagnostic: size of the inner signal vs the residual, and main output stats."""
import sys; from pathlib import Path
R = Path("lab/spikes/2026-10-30-two-stage-hnet-smoke")
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
import torch
from functools import partial
from hnet.data import ByteLoader
from hnet.stage import Stage
from sample import load
dev = torch.device("cuda")
for path in sys.argv[1:]:
    model, cfg = load(Path(path), dev); model.eval()
    tc = cfg["train"]
    b = ByteLoader(str(Path("~/data/hnet-bytes/fineweb-edu/val.bin").expanduser()), tc["seq_len"], 4, cfg["seed"], dev).fixed(1)[0]
    stages = list(model.stages())
    rec = {}
    for s, st in enumerate(stages):
        orig = st._scatter_out
        def wrap(y, p_c, read_idx, conf, resid, dtype, ablate=None, _s=s, _o=orig):
            yy = y[..., :stages[_s].d].float()
            z = _o(y, p_c, read_idx, conf, resid, dtype, ablate)
            rec[_s] = dict(inner_rms=yy.pow(2).mean().sqrt().item(), resid_rms=resid.float().pow(2).mean().sqrt().item(),
                           z_minus_resid_rms=(z.float() - resid.float()).pow(2).mean().sqrt().item(),
                           conf_mean=conf.float().mean().item())
            return z
        st._scatter_fn = wrap
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        x = b[0]; model(x, torch.full((x.shape[0],), x.shape[1], device=dev))
    print(cfg["wandb"]["name"], cfg["wandb"].get("group"), Path(path).parent.parent.name)
    for s, st in enumerate(stages):
        print(f"  s{s}", {k: round(v, 4) for k, v in rec[s].items()}, "residual_proj |W|", round(st.residual_proj.weight.norm().item(), 3),
              "decoder in-norm w" , [round(m.weight.norm().item(),3) for n, m in st.decoder.named_modules() if n.endswith("norm") or n.endswith("norm1")][:2])
    del model; torch.cuda.empty_cache()
