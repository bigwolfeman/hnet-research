#!/usr/bin/env bash
# "The next ladder" queue (user request 2026-11-03: all four, A last, start at 19:30).
#   no other train.py -> CUDA suite gate -> 200-step smoke of every new config -> B -> C -> D -> A
#   B grown hierarchy keeps main   lab/experiments/planned/2026-11-03-grown-hierarchy-keeps-main.md
#   C looped coarse main           lab/experiments/planned/2026-11-03-looped-coarse-main.md
#   D thin outer stages            lab/experiments/planned/2026-11-03-thin-outer-stages.md
#   A reference layout vs BPE      lab/experiments/planned/2026-11-02-reference-layout-vs-bpe.md
# A failed smoke skips that experiment (loud line in chain.log); a failed arm is logged by run_arms.sh and the queue goes on.
# Rerunning this script after a crash is safe: run_arms.sh skips finished arms and resumes from resume.pt.
set -u
ROOT=$(git rev-parse --show-toplevel)
D=$ROOT/ignored/runs/next-ladder
cd "$ROOT/lab/spikes/2026-10-30-two-stage-hnet-smoke" || exit 1
PY=.venv/bin/python
log() { echo "$(date -Is) $*" >> "$D/chain.log"; }
log "launch ($(git rev-parse --short HEAD))"
while pgrep -f "[t]rain.py experiment=" > /dev/null; do sleep 30; done
log "suite start"
if ! HNET_TEST_CUDA=1 $PY -m pytest -q > "$D/suite.log" 2>&1; then
    log "SUITE FAILED, nothing started (see suite.log)"; exit 1
fi
log "suite ok: $(tail -1 "$D/suite.log")"

smoke() {   # smoke <name> <experiment> [overrides...]: 200 steps, one eval with the coarse probe, final.pt
    local name=$1 exp=$2; shift 2
    if OMP_NUM_THREADS=2 $PY train.py experiment="$exp" train=ladder seed=0 hydra.run.dir="$D/smoke/$name" \
        train.max_steps=200 train.eval_every=200 train.resume_every=0 "train.ladder_fracs=[]" \
        wandb.project=hnet-smoke wandb.group=next-ladder-smoke wandb.name="$name" "$@" > "$D/smoke/$name.log" 2>&1; then
        log "smoke ok $name: $(grep -o '"final_val_bpb": [0-9.]*' "$D/smoke/$name.log" | tail -1) $(grep -m1 '^params' "$D/smoke/$name.log" | cut -c1-40)"
        return 0
    fi
    log "SMOKE FAILED $name (see smoke/$name.log)"; return 1
}
mkdir -p "$D/smoke"
okB=0; okC=0; okD=0
smoke grow_p1 reflayout_1stage && smoke grow_p2 reflayout_2stage train.grow_from="$D/smoke/grow_p1/final.pt" && okB=1
smoke loop2 loop2_1stage && smoke loop4 loop4_1stage && okC=1
smoke thin thin_outer_2stage && okD=1

if ((okB)); then
    log "B start"
    TRAIN=ladder scripts/run_arms.sh "$D/grown-p1" grown-hierarchy-p1 reflayout_1stage:0 reflayout_1stage:1 reflayout_1stage:2 \
        -- train.max_steps=4000 train.decay_frac=0.0 "train.ladder_fracs=[]"
    for s in 0 1 2; do
        src="$D/grown-p1/reflayout_1stage_s$s/final.pt"
        if [[ ! -f $src ]]; then log "B FAILED seed $s: no phase-1 final.pt"; continue; fi
        TRAIN=ladder scripts/run_arms.sh "$D/grown" grown-hierarchy reflayout_2stage:$s \
            -- train.max_steps=8000 "train.ladder_fracs=[]" train.grow_from="$src"
    done
    log "B done"
else log "B SKIPPED (smoke failed)"; fi

if ((okC)); then
    log "C start"
    TRAIN=ladder scripts/run_arms.sh "$D/looped" looped-coarse-main \
        loop1_1stage:0 loop2_1stage:0 loop4_1stage:0 loop1_1stage:1 loop2_1stage:1 loop4_1stage:1 \
        loop1_1stage:2 loop2_1stage:2 loop4_1stage:2 -- train.max_steps=12000 "train.ladder_fracs=[]"
    log "C done"
else log "C SKIPPED (smoke failed)"; fi

if ((okD)); then
    log "D start"
    TRAIN=ladder scripts/run_arms.sh "$D/thin" thin-outer-stages thin_outer_2stage:0 thin_outer_2stage:1 thin_outer_2stage:2 \
        -- train.max_steps=12000 "train.ladder_fracs=[]"
    log "D done"
else log "D SKIPPED (smoke failed)"; fi

log "A start"
TRAIN=ladder scripts/run_arms.sh "$ROOT/ignored/runs/reference-layout-vs-bpe" reference-layout-vs-bpe reflayout_2stage:0 reflayout_2stage:1
log "A done"
grep -h FAILED "$D"/*/index.log "$ROOT/ignored/runs/reference-layout-vs-bpe/index.log" 2>/dev/null | sed 's/^/  /' >> "$D/chain.log"
log "CHAIN DONE"
