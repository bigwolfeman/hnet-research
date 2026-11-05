import sys, wandb, math
api = wandb.Api()
for r in sorted(api.runs("hnet-router", filters={"group": sys.argv[1]}), key=lambda r: r.name):
    c, s = r.config, r.summary
    st = c["model"]["stages"]
    lay = [(x.get("enc_layout"), x.get("dec_layout"), x.get("window")) for x in st]
    keys = [k for k in s.keys() if "byte" in k or "param" in k]
    print(r.id, r.name, c["data"]["train_sources"][0]["file"], r.state, lay, c["model"].get("main_layout", c["model"].get("main")),
          {k: s[k] for k in keys}, "loss", s.get("train/loss"), "steps", s.get("_step"))
