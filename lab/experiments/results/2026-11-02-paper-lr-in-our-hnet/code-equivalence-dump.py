import sys, torch
sys.path.insert(0, sys.argv[1])
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hnet.model import build_model
from hnet.flops import flops_per_byte
with initialize_config_dir(config_dir=sys.argv[3], version_base=None):
    cfg = compose(config_name="config", overrides=["experiment=blog_split_2stage", "train=ladder"])
mc = OmegaConf.to_container(cfg.model, resolve=True)
mc.pop("ssm", None)   # the old code has no ssm key; the T-only model must not depend on it
torch.manual_seed(0)
m = build_model(mc).cuda().eval()
g = torch.Generator().manual_seed(1)
x = torch.randint(0, 256, (4, 4096), generator=g).cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(x)
torch.save(dict(sd={k: v.cpu() for k, v in m.state_dict().items()}, logits=out.logits.float().cpu(),
                counts=[a.counts.cpu() for a in out.aux], flops=flops_per_byte(mc, 4096, [4.0, 2.0])["fwd"]), sys.argv[2])
