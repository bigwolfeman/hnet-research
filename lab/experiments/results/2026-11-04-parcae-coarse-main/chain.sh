#!/usr/bin/bash
# Queue: Parcae coarse main  lab/experiments/planned/2026-11-04-parcae-coarse-main.md
# Starts after the windowed-growth queue ends.
ROOT=$(git rev-parse --show-toplevel)
D=$ROOT/ignored/runs/parcae-coarse-main; SMOKE_GROUP=parcae-coarse-main-smoke
until grep -q "CHAIN DONE\|SUITE FAILED\|SMOKE FAILED" $ROOT/ignored/runs/windowed-growth/chain.log 2>/dev/null; do sleep 300; done
cd "$ROOT/lab/spikes/2026-10-30-two-stage-hnet-smoke" || exit 1
source scripts/chain_lib.sh
log "launch ($(git rev-parse --short HEAD))"
wait_gpu
suite_gate
smoke parcae_r2 parcae_r2_1stage || exit 1
smoke parcae_r6 parcae_r6_1stage || exit 1
log "arms start"
TRAIN=ladder scripts/run_arms.sh "$D/arms" parcae-coarse-main \
    parcae_r2_1stage:0 parcae_r6_1stage:0 parcae_r2_1stage:1 parcae_r6_1stage:1 parcae_r2_1stage:2 parcae_r6_1stage:2 \
    -- train.max_steps=12000 "train.ladder_fracs=[]"
log "arms done"
grep -h FAILED "$D"/*/index.log >> "$D/chain.log" 2>/dev/null
log "CHAIN DONE"
