"""Hydra training entry point. All hyperparameters live in configs/.

Pseudocode:
    cfg -> model (recursive Stage/Main), param groups per stage (lr = base * lr_mult[stage], WSD schedule)
    init_from set: model.load_state_dict(file.model, strict)   # step-0 weights from a file; sha256 logged
    grow_from set: final.pt of an N-stage run -> hnet/grow.py maps it into this (N+1)-stage model; new stage fresh
    wandb.init(config = resolved cfg + model.describe() + target flops)      # full config, reproducible
    for step in 1..max_steps:
        x, y = loader.next()                                                 # weighted source per row, random-offset memmap slice
        with bf16 autocast: out = model(x); loss = CE + sum_s w_s * ratio_loss_s
        loss.backward(); clip; AdamW step (fp32 master weights)
        ratio_integral_lr > 0: loss += sum_s lam_s * G_s ; lam_s += lr (N_s F_s - 1)   # RatioController, PI with the ratio loss
        jepa.weight > 0 (before the LM forward): (a, b) = pairs.next(); (w * CoarseJEPA.loss(model, a, b, step)).backward()  # hnet/jepa.py; own rng
        log: loss, bpb, measured ratio per stage, bytes/s, mem
        grad_stats: per level, ||grad|| before clipping and rms of the AdamW direction after the step (hnet/gradstats.py)
        every eval_every : per-domain val bpb, bytes/chunk per stage, hnet_over_bpe (hnet/evaluation.py),
                           coarse-path use per stage (probes.coarse_use)
        model.kind == token (BPE control): MixLoader over uint16 tokens (rows by byte weight / bytes per token),
            eval = fixed token windows, bpb = sum CE / (ln2 * bytes of the targets); no stages, no probes
        every probe_every: boundary alignment, in-context tokenization, rendered sample,
                           semantic chunking score (hnet/semantics.py: router-p AUROC per level, sem/*)
        step == b * (1 - decay_frac) for b in ladder_fracs * max_steps:      # WSD compute ladder, one run
            snapshot model + optimizer; decay to b; eval; log as ladder_b{b}/*; restore snapshot; continue stable
        resume_every > 0 and step % resume_every == 0:                      # power-loss safety (fixed Hydra run dir)
            resume.pt <- model, AdamW, controller, loader rng, wandb id, ladder results   (fsync, then rename; previous
            file kept as resume_prev.pt). Start: a readable resume.pt in the run dir with the same config -> continue
            at step + 1; re-evaluate and assert the val bpb matches; wandb resumes the same run id. wandb keeps
            the steps the crashed attempt logged after the checkpoint, so logs below its next step are skipped
            (they stay in the local log); rung metrics also go to the wandb summary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import wandb
from hnet import bpe, gradstats, probes, semantics, tokens
from hnet.chunking import RatioController
from hnet.data import ByteLoader, MixLoader, PairLoader
from hnet.evaluation import evaluate, evaluate_tokens
from hnet.flops import flops_per_byte, token_flops_per_byte
from hnet.grow import grow_state_dict
from hnet.jepa import CoarseJEPA, main_net
from hnet.model import LN2, HNetLM, build_model, measured_ratios, total_loss


def wsd_factor(step: int, total: int, warmup: int, decay_frac: float, min_frac: float) -> float:
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return (step + 1) / warmup
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / max(total - decay_start, 1)
    return min_frac + (1 - min_frac) * (1 - t)


def make_optimizer(model: HNetLM, tc, extra: torch.nn.Module | None = None) -> torch.optim.Optimizer:
    """AdamW with per-stage LR multipliers. `extra` (the JEPA predictor) joins the main network's level."""
    mults = list(tc.lr_mult)
    groups = model.stage_param_groups()
    assert len(mults) == len(groups) == model.n_stages + 1, f"lr_mult needs {model.n_stages + 1} entries"
    groups = [(lvl, named, f"stage{lvl}") for lvl, named in groups]
    if extra is not None:
        groups.append((model.n_stages, list(extra.named_parameters()), f"stage{model.n_stages}_extra"))
    pg = []
    for lvl, named, tag in groups:
        for decay in (True, False):
            ps = [p for _, p in named if (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)) == decay]
            if ps:
                pg.append(dict(params=ps, lr_mult=mults[lvl], weight_decay=tc.weight_decay if decay else 0.0,
                               name=f"{tag}_{'decay' if decay else 'nodecay'}"))
    return torch.optim.AdamW(pg, lr=tc.lr, betas=tuple(tc.betas), fused=torch.cuda.is_available())


RESUME_FILES = ("resume.pt", "resume_prev.pt")   # newest first


def config_diff(a, b, path="") -> list[str]:
    """Dotted paths where two nested config dicts differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in config_diff(a.get(k), b.get(k), f"{path}{k}.")]
    return [] if a == b else [path.rstrip(".")]


def load_resume(out_dir: Path, run_cfg: dict) -> dict | None:
    """Newest readable resume checkpoint in out_dir, or None if there is none. A torn newest file (power loss during
    the write) falls back to the previous one, loudly. A config mismatch is an error, never a silent fresh start."""
    for name in RESUME_FILES:
        path = out_dir / name
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001  (torn file: fall back to the previous checkpoint)
            print(f"RESUME: {path} is unreadable ({e!r}); trying the previous checkpoint", flush=True)
            continue
        diff = config_diff(ck["cfg"], run_cfg)
        assert not diff, f"{path} was written by a run with a different config: {diff}"
        print(f"RESUME from {path} at step {ck['step']}", flush=True)
        return ck
    if any((out_dir / n).exists() for n in RESUME_FILES):
        raise RuntimeError(f"resume checkpoints in {out_dir} exist but none is readable")
    return None


def save_resume(out_dir: Path, state: dict):
    """Write resume.pt durably: tmp file + fsync, keep the old file as resume_prev.pt, then rename into place."""
    tmp, cur, prev = out_dir / "resume.pt.tmp", out_dir / RESUME_FILES[0], out_dir / RESUME_FILES[1]
    with open(tmp, "wb") as fh:
        torch.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())
    if cur.exists():
        os.replace(cur, prev)
    os.replace(tmp, cur)


def is_token_cfg(mcfg: dict) -> bool:
    return mcfg.get("kind", "hnet") == "token"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def run_probes(model, val_loader, paragraphs, device, step, log, sem_levels=None):
    model.eval()
    rows = [val_loader.slice(i * 65536, 1024) for i in range(16)]
    m = probes.boundary_alignment(model, rows, device)
    m.update(probes.in_context(model, paragraphs, device))
    if sem_levels:
        m.update(semantics.semantic_report(model, sem_levels, device))
    sample = val_loader.slice(3 * 65536 + 5000, 600)
    text = probes.render(model, sample, device)[0]
    log({"probe/sample_html": wandb.Html(probes.render_html(text)),
         "probe/sample_table": wandb.Table(columns=["step", "text"], data=[[step, text]])}, step)
    model.train()
    return m, text


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    tc = cfg.train
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("high")
    amp = getattr(torch, tc.amp_dtype)
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    if tc.resume_every:
        assert tc.resume_every % tc.eval_every == 0, "resume_every must be a multiple of eval_every (resume check)"
    resume = load_resume(out_dir, run_cfg) if tc.resume_every else None
    loader_state = resume["loader"] if resume else None

    model = build_model(mcfg)
    init_src = None
    if tc.init_from:
        # step-0 weights from a file (e.g. the reference H-Net's init mapped by scripts/ref_init_transplant.py);
        # strict: every parameter must be present with the same shape. A resume, if any, overwrites them later.
        init_path = Path(tc.init_from)
        blob = torch.load(init_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"], strict=True)
        init_src = dict(path=str(init_path), sha256=hashlib.sha256(init_path.read_bytes()).hexdigest(),
                        source=blob.get("source"))
        print(f"init_from {init_path} sha256 {init_src['sha256'][:12]} source {init_src['source']}", flush=True)
    if tc.grow_from:
        # a trained N-stage final.pt -> this (N+1)-stage model: the new stage sits above main, fresh init (hnet/grow.py)
        assert not tc.init_from, "init_from and grow_from are exclusive"
        grow_path = Path(tc.grow_from)
        blob = torch.load(grow_path, map_location="cpu", weights_only=False)
        src_stages = len(blob["config"]["model"]["stages"])
        sd, fresh = grow_state_dict(blob["model"], src_stages, model)
        model.load_state_dict(sd, strict=True)
        init_src = dict(path=str(grow_path), sha256=hashlib.sha256(grow_path.read_bytes()).hexdigest(),
                        source=f"grow_from: {src_stages}-stage step {blob.get('step')} "
                               f"wandb {blob['config'].get('wandb', {}).get('name')}", fresh_keys=len(fresh))
        print(f"grow_from {grow_path} sha256 {init_src['sha256'][:12]} {init_src['source']}; "
              f"{len(fresh)} fresh keys in the new stage", flush=True)
    model = model.to(device)
    if tc.compile:
        model.compile_stacks()
    jepa = None
    if tc.jepa.weight > 0:
        assert not is_token_cfg(mcfg) and len(mcfg["stages"]), "jepa needs a model with stages"
        assert all(w % mcfg["chunk_multiple"] == 0 for w in tc.jepa.widths), "jepa widths: multiples of chunk_multiple"
        jc = tc.jepa
        jepa = CoarseJEPA(model, jc.proj_hidden, jc.proj_dim, jc.sigreg_lambda, jc.sigreg_slices, jc.sigreg_knots,
                          cfg.seed + jc.seed_offset).to(device)
    opt = make_optimizer(model, tc, jepa)
    grad_levels = gradstats.levels(opt)   # level -> params, for the per-level grad statistics (hnet/gradstats.py)

    data_dir = Path(cfg.data.dir)
    primary = cfg.data.primary_val
    is_token = is_token_cfg(mcfg)
    full_cfg = copy.deepcopy(run_cfg)
    if is_token:
        # BPE control: uint16 token memmaps, rows drawn by byte weight, val = fixed token windows with exact byte counts.
        tdir = data_dir / cfg.data.token.dir
        meta, nbytes = tokens.load_meta(tdir), tokens.load_nbytes(tdir)
        assert meta["vocab_size"] == mcfg["vocab_size"], (meta["vocab_size"], mcfg["vocab_size"])
        file_bpt = lambda f: meta["files"][Path(f).stem]["bytes_per_token"]  # noqa: E731
        sources = [dict(name=d.name, file=str(tdir / cfg.data.token.train[d.name]), weight=d.weight,
                        bytes_per_token=file_bpt(cfg.data.token.train[d.name])) for d in cfg.data.train_sources]
        train_loader = MixLoader(sources, tc.seq_len, tc.batch_size, cfg.seed, device, cfg.data.prefetch, dtype=np.uint16,
                                 state=loader_state)
        val_windows = {d.name: torch.from_numpy(tokens.eval_windows(
            np.memmap(tdir / cfg.data.token.val[d.name], dtype=np.uint16, mode="r"), tc.seq_len,
            cfg.data.token.eval_windows, tc.batch_size)).to(device) for d in cfg.data.val_sets}
        assert primary in val_windows, f"primary_val {primary!r} not in val_sets {list(val_windows)}"
        bpts = {d.name: file_bpt(cfg.data.token.val[d.name]) for d in cfg.data.val_sets}
        w = np.array([float(s["weight"]) for s in sources])
        train_bpt = float(w.sum() / (w / np.array([s["bytes_per_token"] for s in sources])).sum())  # mix bytes per token
        bytes_per_step = tc.batch_size * tc.seq_len * train_bpt
        planned = tc.max_steps * bytes_per_step
        target = cfg.data.token.target_train_bytes
        assert abs(planned / target - 1) < 0.01, f"train bytes {planned:.4g} vs target {target:.4g}: fix train.max_steps"
        full_cfg["token_meta"] = meta
        full_cfg["train_bytes_per_token"] = train_bpt
        full_cfg["planned_train_bytes"] = planned
        full_cfg["flops_at_target_ratios"] = token_flops_per_byte(mcfg, tc.seq_len, bpts[primary])
        print(f"token mix: {train_bpt:.3f} bytes/token, {planned / 1e6:.1f}M train bytes planned, val bytes/token "
              f"{ {k: round(v, 3) for k, v in bpts.items()} }", flush=True)
    else:
        sources = [dict(name=d.name, file=str(data_dir / d.file), weight=d.weight) for d in cfg.data.train_sources]
        train_loader = MixLoader(sources, tc.seq_len, tc.batch_size, cfg.seed, device, cfg.data.prefetch, state=loader_state)
        val_sets = {d.name: ByteLoader(str(data_dir / d.file), tc.seq_len, tc.batch_size, cfg.seed, device)
                    for d in cfg.data.val_sets}
        assert primary in val_sets, f"primary_val {primary!r} not in val_sets {list(val_sets)}"
        val_batches = {n: l.fixed(tc.eval_batches) for n, l in val_sets.items()}
        bpts = {d.name: bpe.bytes_per_token(data_dir / d.file, cfg.data.bpe_encoding) for d in cfg.data.val_sets}
        val_loader = val_sets[primary]
        print("bpe bytes/token", {k: round(v, 3) for k, v in bpts.items()}, flush=True)
        bytes_per_step = tc.batch_size * tc.seq_len
        target_ratios = [float(s["router"]["target_ratio"]) for s in mcfg["stages"]]
        full_cfg["flops_at_target_ratios"] = flops_per_byte(mcfg, tc.seq_len, target_ratios)
    has_probes = not is_token and model.n_stages > 0
    pairs = None
    if jepa is not None:
        pairs = PairLoader(str(data_dir / tc.jepa.pairs_dir), tuple(tc.jepa.views), tuple(tc.jepa.widths),
                           tc.jepa.pairs_per_step, cfg.seed + tc.jepa.seed_offset, device,
                           state=resume.get("pairs") if resume else None)
        full_cfg["jepa_resolved"] = dict(pairs=pairs.n, proj_params=sum(p.numel() for p in jepa.parameters()),
                                         view_bytes_per_step_max=tc.jepa.pairs_per_step * sum(tc.jepa.widths),
                                         view_bytes_over_lm_bytes_max=tc.jepa.pairs_per_step * sum(tc.jepa.widths)
                                         / (tc.batch_size * tc.seq_len))
    # the main network's own monitors (hnet/parcae.py ParcaeMain.log_stats), logged as train/<key>
    leaf_stats = getattr(main_net(model), "log_stats", None) if isinstance(model, HNetLM) else None
    ratio_ctl = None
    if tc.ratio_integral_lr:
        assert not is_token and model.n_stages, "ratio_integral_lr needs a model with stages"
        ratio_ctl = RatioController([float(s["router"]["target_ratio"]) for s in mcfg["stages"]], tc.ratio_integral_lr, device)
    paragraphs = probes.load_probe(cfg.data.probe_file) if has_probes else None
    sem_levels = semantics.probe_levels(data_dir, cfg.data.semantic) if has_probes and cfg.data.get("semantic") else None
    full_cfg["bpe_bytes_per_token"] = bpts
    full_cfg["model_resolved"] = model.describe()
    full_cfg["init_from"] = init_src
    full_cfg["env"] = dict(torch=torch.__version__, cuda=torch.version.cuda, git=git_rev(),
                           device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    # wandb rewind and fork are not enabled on our server, so resume continues the same run id.
    same_run = dict(id=resume["wandb_id"], resume="must") if resume and cfg.wandb.mode != "disabled" else {}
    wb = wandb.init(project=cfg.wandb.project, group=cfg.wandb.group, name=cfg.wandb.name, mode=cfg.wandb.mode,
                    config=full_cfg, **same_run)
    wb_floor = wb.starting_step if same_run else 0   # wandb drops steps below this (the crashed attempt logged them); wb.step is 0 here
    if same_run:
        print(f"wandb: resumed run {wb.id}; steps {resume['step'] + 1}..{wb_floor - 1} keep the crashed attempt's "
              f"history, this attempt logs from step {wb_floor}", flush=True)

    def log(d: dict, step: int):
        if step >= wb_floor:
            wb.log(d, step=step)
    print("params", full_cfg["model_resolved"]["params"], "wandb", getattr(wb, "url", None), flush=True)

    # WSD ladder: a budget b < max_steps branches off the stable phase at its own decay start, decays to b, is
    # evaluated, and the trainer restores the pre-branch state. One run gives a compute ladder (stable LR is shared).
    branch_at = {}
    for frac in tc.ladder_fracs:
        b = round(frac * tc.max_steps)
        assert 0 < b < tc.max_steps and int(b * (1 - tc.decay_frac)) > tc.warmup_steps, f"ladder budget {b} steps"
        branch_at[int(b * (1 - tc.decay_frac))] = b
    assert all(s < int(tc.max_steps * (1 - tc.decay_frac)) for s in branch_at), "ladder branches must start in the stable phase"

    def train_step(f: float, step: int):
        for g in opt.param_groups:
            g["lr"] = tc.lr * g["lr_mult"] * f
        x, y = train_loader.next()
        lengths = torch.full((x.shape[0],), x.shape[1], device=device)
        jm, jl = {}, None
        if jepa is not None:
            # backward of the JEPA term first: its activations are freed before the LM forward, so peak memory is the
            # larger of the two passes, not their sum (the sum ran out of 32 GB). Same gradient: both add into .grad.
            xa, la, xb, lb = pairs.next()
            with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda"):
                jl, jm = jepa.loss(model, xa, la, xb, lb, step)
            (tc.jepa.weight * jl).backward()
        with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == "cuda"):
            out = model(x, lengths)
        loss, ce = total_loss(out, y, lengths, ratio_ctl.lam if ratio_ctl else None)
        loss.backward()
        if jl is not None:
            loss = loss.detach() + tc.jepa.weight * jl.detach()
        if ratio_ctl:
            ratio_ctl.update([a.frac for a in out.aux])
        params = list(model.parameters()) + (list(jepa.parameters()) if jepa is not None else [])
        stats = tc.grad_stats and (step % tc.log_every == 0 or step == 1)
        gl = gradstats.grad_norms(grad_levels) if stats else {}   # before clipping (clip scales all levels alike)
        gn = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        opt.step()
        if stats:
            ur = gradstats.update_rms(opt, grad_levels)
            jm = {**jm, **{f"grad/norm_l{L}": v for L, v in gl.items()}, **{f"grad/update_rms_l{L}": v for L, v in ur.items()}}
        opt.zero_grad(set_to_none=True)
        return out, ce.detach(), loss.detach(), gn, jm

    def run_eval(step: int, label: str) -> dict:
        if is_token:
            m = evaluate_tokens(model, val_windows, nbytes, mcfg, tc.seq_len, tc.batch_size, device, amp, primary)
            ratios = []
        else:
            m, ratios = evaluate(model, val_batches, bpts, mcfg, tc.seq_len, device, amp, primary)
            if model.n_stages and tc.coarse_probe_batches:
                m.update(probes.coarse_use(model, val_batches[primary][:tc.coarse_probe_batches], device, amp))
        print(f"EVAL {label} step {step}: val bpb {m['val/bpb']:.4f} ratios {[round(r, 2) for r in ratios]} "
              f"flops/byte fwd {m['flops/fwd_per_byte']/1e6:.2f}M", flush=True)
        for dn in val_windows if is_token else val_sets:
            if is_token:
                print(f"  {dn}: bpb {m[f'val/{dn}/bpb']:.4f} ce/token {m[f'val/{dn}/ce_per_token']:.4f} "
                      f"bytes/token {m[f'val/{dn}/bytes_per_token']:.3f}", flush=True)
            else:
                print(f"  {dn}: bpb {m[f'val/{dn}/bpb']:.4f} bytes/chunk "
                      f"{[round(m[f'val/{dn}/bytes_per_chunk_s{k}'], 2) for k in range(model.n_stages)]} hnet/bpe "
                      f"{[round(m[f'val/{dn}/hnet_over_bpe_s{k}'], 3) for k in range(model.n_stages)]}", flush=True)
        return m

    def run_branch(start: int, budget: int) -> dict:
        """Decay from the stable state at `start` to `budget`, evaluate, restore. Logged at `start` (wandb steps only rise)."""
        snap_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        snap_opt = copy.deepcopy(opt.state_dict())
        snap_ctl = ratio_ctl.state_dict() if ratio_ctl else None
        snap_jepa = {k: v.detach().clone() for k, v in jepa.state_dict().items()} if jepa is not None else None
        for s in range(start + 1, budget + 1):
            train_step(wsd_factor(s - 1, budget, tc.warmup_steps, tc.decay_frac, tc.min_lr_frac), s)
        m = run_eval(budget, f"ladder b{budget}")
        m["ladder/train_bytes"] = budget * bytes_per_step
        (out_dir / f"ladder_{budget}.json").write_text(json.dumps(dict(budget_steps=budget, branch_start=start, **m), indent=1))
        if tc.save_checkpoint:
            torch.save(dict(model=model.state_dict(), config=full_cfg, step=budget), out_dir / f"ladder_{budget}.pt")
        rung = {f"ladder_b{budget}/{k}": v for k, v in m.items()}
        log(rung, start)
        wb.summary.update(rung)   # also in the summary: a resume can skip the step this rung was logged at
        model.load_state_dict(snap_model)
        opt.load_state_dict(snap_opt)
        if ratio_ctl:
            ratio_ctl.load_state_dict(snap_ctl)
        if jepa is not None:
            jepa.load_state_dict(snap_jepa)
        return m

    t_last, bytes_since, step_times = time.perf_counter(), 0, []
    t0 = t_last
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    last, ladder, start = {}, {}, 1
    if resume:
        model.load_state_dict(resume["model"])
        if jepa is not None:
            jepa.load_state_dict(resume["jepa"])
        opt.load_state_dict(resume["opt"])
        if ratio_ctl:
            ratio_ctl.load_state_dict(resume["ctl"])
        last, ladder, step_times, start = resume["last"], resume["ladder"], resume["step_times"], resume["step"] + 1
        m = run_eval(resume["step"], "resume check")   # not logged: wandb holds this step's eval already
        assert abs(m["val/bpb"] - last["val/bpb"]) < 1e-3, f"restored model evaluates to {m['val/bpb']}, saved {last['val/bpb']}"
        del resume
        t_last = t0 = time.perf_counter()
    for step in range(start, tc.max_steps + 1):
        f = wsd_factor(step - 1, tc.max_steps, tc.warmup_steps, tc.decay_frac, tc.min_lr_frac)
        out, ce_d, loss_d, gn, jm = train_step(f, step)
        bytes_since += bytes_per_step

        if step % tc.log_every == 0 or step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            dt = now - t_last
            ratios = measured_ratios(out.aux)
            # token model: ce is nats per token; bpb uses the train mix's bytes per token (an estimate per batch)
            train_bpb = ce_d.item() / LN2 / (train_bpt if is_token else 1.0)
            last = {"train/ce": ce_d.item(), "train/bpb": train_bpb, "train/loss": loss_d.item(),
                    "train/grad_norm": float(gn), "train/lr_factor": f, "train/bytes": step * bytes_per_step,
                    "perf/bytes_per_s": bytes_since / dt, "perf/step_time_s": dt / max(round(bytes_since / bytes_per_step), 1),
                    "perf/peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                    "train/ratio_total": float(np.prod(ratios))}
            for s, a in enumerate(out.aux):
                last[f"train/ratio_s{s}"] = ratios[s]
                last[f"train/ratio_loss_s{s}"] = float(a.ratio_loss.detach())
                last[f"train/p_mean_s{s}"] = float(a.p_mean.detach())
                last[f"train/ratio_lambda_s{s}"] = float(ratio_ctl.lam[s]) if ratio_ctl else 0.0
                if a.router_loss is not None:   # router "surprise": weighted next-byte CE of its aux head
                    last[f"train/router_loss_s{s}"] = float(a.router_loss.detach())
            for s, st in enumerate(model.stages() if hasattr(model, "stages") else []):
                for k in ("beta", "tau"):   # router "surprise": learned entropy scale and threshold
                    if hasattr(st.router, k):
                        last[f"train/router_{k}_s{s}"] = float(getattr(st.router, k).detach())
            last.update({f"train/{k}": float(v) for k, v in jm.items()})
            if leaf_stats is not None:
                last.update({f"train/{k}": v for k, v in leaf_stats().items()})
            log(last, step)
            if step - tc.log_every >= start:   # the first interval of a run or a resume holds compile time
                step_times.append((dt, bytes_since))
            print(f"step {step} ce {ce_d.item():.4f} bpb {train_bpb:.3f} ratios {[round(r,2) for r in ratios]} "
                  f"{bytes_since/dt/1e3:.0f} kB/s {last['perf/step_time_s']*1e3:.0f} ms/step", flush=True)
            t_last, bytes_since = now, 0

        if step % tc.eval_every == 0 or step == tc.max_steps:
            m = run_eval(step, "main")
            if has_probes and (step % tc.probe_every == 0 or step == tc.max_steps):
                pm, text = run_probes(model, val_loader, paragraphs, device, step, log, sem_levels)
                m.update(pm)
            log(m, step)
            last.update(m)
        if step in branch_at:
            ladder[branch_at[step]] = run_branch(step, branch_at[step])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_last, bytes_since = time.perf_counter(), 0   # branch time is not training throughput
        if tc.resume_every and step % tc.resume_every == 0 and step < tc.max_steps:
            save_resume(out_dir, dict(step=step, cfg=run_cfg, model=model.state_dict(), opt=opt.state_dict(),
                                      ctl=ratio_ctl.state_dict() if ratio_ctl else None, loader=train_loader.state_dict(),
                                      jepa=jepa.state_dict() if jepa is not None else None,
                                      pairs=pairs.state_dict() if pairs is not None else None,
                                      wandb_id=wb.id, ladder=ladder, last=last, step_times=step_times))
            t_last, bytes_since = time.perf_counter(), 0   # checkpoint time is not training throughput
        if tc.time_limit_s and time.perf_counter() - t0 > tc.time_limit_s:
            print("time limit reached", flush=True)
            break

    tot_bytes = sum(b for _, b in step_times)
    tot_t = sum(d for d, _ in step_times)
    summary = dict(final_val_bpb=last.get("val/bpb"), bytes_per_s=tot_bytes / max(tot_t, 1e-9),
                   step_time_ms=1e3 * tot_t / max(tot_bytes / bytes_per_step, 1),
                   peak_mem_gb=last.get("perf/peak_mem_gb"), flops_fwd_per_byte=last.get("flops/fwd_per_byte"),
                   ratios=[last.get(f"val/ratio_s{s}") for s in range(model.n_stages)], wandb_url=getattr(wb, "url", None),
                   params=full_cfg["model_resolved"]["params"], probes={k: v for k, v in last.items() if k.startswith(("probe/", "sem/"))},
                   domains={k: v for k, v in last.items() if k.startswith("val/") and k.count("/") == 2})
    summary["ladder"] = {str(b): {k: v for k, v in m.items() if k.count("/") == 2 or k.startswith(("probe/coarse", "flops/"))}
                         for b, m in sorted(ladder.items())}
    for k, v in summary.items():
        wb.summary[f"final/{k}"] = v
    if tc.save_checkpoint:
        torch.save(dict(model=model.state_dict(), config=full_cfg, step=step), out_dir / "final.pt")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))   # written last: marks the run done
    for name in RESUME_FILES:
        (out_dir / name).unlink(missing_ok=True)
    print("OUT_DIR", out_dir, flush=True)
    print("SUMMARY", json.dumps(summary, default=str), flush=True)
    wb.finish()


if __name__ == "__main__":
    main()
