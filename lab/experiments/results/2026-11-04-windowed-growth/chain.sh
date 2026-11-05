#!/usr/bin/bash
# Queue: windowed growth keeps main  lab/experiments/planned/2026-11-04-windowed-growth-keeps-main.md
# Starts after the next-ladder queue (A, then the thin-outer seed 2 rerun) ends.
ROOT=$(git rev-parse --show-toplevel)
D=$ROOT/ignored/runs/windowed-growth; SMOKE_GROUP=windowed-growth-smoke
until grep -q "thin s2 rerun done" $ROOT/ignored/runs/next-ladder/chain.log; do sleep 300; done
cd "$ROOT/lab/spikes/2026-10-30-two-stage-hnet-smoke" || exit 1
source scripts/chain_lib.sh
log "launch ($(git rev-parse --short HEAD))"
wait_gpu
suite_gate
smoke wg_p1 loop1_1stage || exit 1
smoke wg_p2 windowed_2stage train.grow_from="$D/smoke/wg_p1/final.pt" || exit 1
smoke wg_scratch windowed_2stage || exit 1

grow_seed() {   # phase 1 then growth for one seed; returns 1 if the phase-1 precondition fails
    local s=$1
    TRAIN=ladder scripts/run_arms.sh "$D/p1" windowed-growth-p1 loop1_1stage:$s \
        -- train.max_steps=4000 train.decay_frac=0.0 "train.ladder_fracs=[]"
    local u; u=$(shuffle_s0 "$D/p1/loop1_1stage_s$s") || { log "seed $s: phase 1 has no summary"; return 1; }
    if ! $PY -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0.0033 else 1)" "$u"; then
        log "seed $s: PRECONDITION FAILED, phase-1 shuffle s0 $u <= 0.0033"; return 1
    fi
    log "seed $s: phase-1 shuffle s0 $u"
    TRAIN=ladder scripts/run_arms.sh "$D/grown" windowed-growth windowed_2stage:$s \
        -- train.max_steps=8000 "train.ladder_fracs=[]" train.grow_from="$D/p1/loop1_1stage_s$s/final.pt"
}
log "grown start"
for s in 0 1 2; do grow_seed $s || grow_seed 3; done
log "grown done"
log "scratch start"
TRAIN=ladder scripts/run_arms.sh "$D/scratch" windowed-scratch windowed_2stage:0 windowed_2stage:1 windowed_2stage:2 \
    -- train.max_steps=12000 "train.ladder_fracs=[]"
log "scratch done"
grep -h FAILED "$D"/*/index.log >> "$D/chain.log" 2>/dev/null
log "CHAIN DONE"
