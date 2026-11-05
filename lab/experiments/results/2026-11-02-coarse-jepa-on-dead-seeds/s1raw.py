import wandb
api = wandb.Api()
for rid in ["i581xzb1","xdktm0nr","191clzls","wqol9jx2","liw70jg2","chl6kcwn","yh1nptqe","hd9kqj1n","21qy39rj"]:
    try: r = api.run(f"hnet-router/{rid}")
    except Exception as e: print(rid, e); continue
    out=[]
    for h in r.scan_history():
        if h.get("probe/coarse_zero_s1") is None: continue
        out.append(f'{h["_step"]}:{h["probe/coarse_zero_s1"]:.2e}/{h["probe/coarse_shuffle_s1"]:.2e}/{h["val/edu/bpb"]:.4f}')
    print(rid, r.group, " ".join(out))
