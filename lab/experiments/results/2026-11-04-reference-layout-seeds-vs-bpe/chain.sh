#!/usr/bin/bash
# Queue: reference layout seeds vs BPE  lab/experiments/planned/2026-11-04-reference-layout-seeds-vs-bpe.md
# Starts after the Parcae coarse main queue ends.
ROOT=$(git rev-parse --show-toplevel)
D=$ROOT/ignored/runs/reference-layout-seeds-vs-bpe; SMOKE_GROUP=reference-layout-seeds-vs-bpe-smoke
until grep -q "CHAIN DONE\|SUITE FAILED\|SMOKE FAILED" $ROOT/ignored/runs/parcae-coarse-main/chain.log 2>/dev/null; do sleep 300; done
cd "$ROOT/lab/spikes/2026-10-30-two-stage-hnet-smoke" || exit 1
source scripts/chain_lib.sh
log "launch ($(git rev-parse --short HEAD))"
wait_gpu
suite_gate
smoke bpe bpe_transformer || exit 1
smoke reflayout reflayout_2stage || exit 1
log "arms start"
TRAIN=ladder scripts/run_arms.sh "$D/arms" reference-layout-seeds-vs-bpe bpe_transformer:1 \
    -- train.max_steps=180080 data.token.target_train_bytes=12582912000
TRAIN=ladder scripts/run_arms.sh "$D/arms" reference-layout-seeds-vs-bpe reflayout_2stage:2
log "arms done"
grep -h FAILED "$D"/*/index.log >> "$D/chain.log" 2>/dev/null
log "CHAIN DONE"
