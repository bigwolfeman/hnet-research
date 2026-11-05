# Agent Note: Resumable training runs

Status: implemented

## Problem

The lab machine can lose power. On 2026-11-01 a power loss stopped the first compute-ladder arm at step 141,550 of 192,000, after 2.2 hours of GPU time. `train.py` saved only rung and final checkpoints, so the whole arm had to restart. The log file on the NTFS (ntfs-3g, FUSE) data drive also ended in torn bytes, so a checkpoint that a power cut interrupts mid-write must be expected.

## Decision

`lab/spikes/2026-10-30-two-stage-hnet-smoke/train.py` writes a resume checkpoint every `train.resume_every` steps, and continues from it at start:

```
every resume_every steps (a multiple of eval_every; ladder.yaml: 8000, about 6 min):
    state = model, AdamW, RatioController lam, train-loader rng state, wandb run id, rung results, throughput log
    write resume.pt.tmp, fsync; rename resume.pt -> resume_prev.pt; rename tmp -> resume.pt
start:
    newest readable of (resume.pt, resume_prev.pt) in the Hydra run dir   # torn newest file -> previous one, loudly
    assert the saved config equals this run's config                      # never a silent fresh start
    restore; re-evaluate at the checkpoint step; assert val bpb equals the saved value (1e-3)
    wandb.init(id=<saved id>, resume="must"); floor = wandb's next step   # same run; no logs below floor
end: final.pt, then summary.json (marks the run done), then delete resume*.pt
```

- wandb keeps the steps that the crashed attempt logged after the checkpoint. train.py skips its own logs below wandb's next step and prints the skipped range, so those steps stay in the local log only. Rung metrics also go to the wandb summary (`ladder_b<steps>/*`), because a rung logged at its branch-start step can fall in that range.
- `MixLoader.state_dict()` holds the rng state right after the last batch that `next()` returned. The prefetch thread runs ahead, so every batch carries the state taken after its own draw. A loader built with `state=` continues the exact batch stream.
- `scripts/run_arms.sh` runs each arm in a fixed Hydra run dir, `<log_dir>/<name>/`, and appends to `<log_dir>/<name>.log`. Run the same command again after a crash: arms with `summary.json` are skipped, and an arm with a resume file continues.
- `resume_every: 0` (smoke, router) turns this off.

## Alternatives considered

- Save every N minutes instead of every N steps. Rejected: the checkpoint step must be an eval step for the resume check, and step-based saves give the same resume points on every rerun.
- One checkpoint file, overwritten in place. Rejected: a power cut during the write destroys the only copy. The tmp + fsync + rename + previous-file scheme keeps one good file at every moment.
- Reseed the loader from (seed, step) on resume. Rejected: this gives a different batch stream from an uninterrupted run. Saving the consumed rng state costs one small dict per batch and keeps the stream exact.
- wandb rewind (`resume_from`) or fork (`fork_from`), which would drop or branch the history at the checkpoint. Not possible: both are in private preview on our wandb server (tested 2026-11-01: error 400 for both). The same-run resume with a step floor is the remaining choice.

## Consequences

- A power cut costs at most `resume_every` steps, plus compile time at restart.
- A resume checkpoint of the 49M-param blog split is about 600 MB. Two copies stay in the run dir until the run ends.
- GPU training is not bit-deterministic (atomic adds in scatter backward), so a resumed run matches an uninterrupted one in distribution, not bit for bit. The restored state itself is exact: the eval at the checkpoint step matches the saved value.
- Tests: `tests/test_resume.py` (rotation, torn-file fallback, config guard) and `tests/test_data.py::test_state_dict_continues_exact_stream` (inline and prefetch).
