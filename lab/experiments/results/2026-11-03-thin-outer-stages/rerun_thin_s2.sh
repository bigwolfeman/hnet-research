#!/usr/bin/bash
# Rerun thin_outer_2stage seed 2 after the chain (incl. A) is done. First run: CUDA OOM near step 60
# (early stage-1 ratio swing, 24.45 GB allocated + 2.54 GB reserved-unallocated). Fix: expandable segments.
ROOT=$(git rev-parse --show-toplevel)
D=$ROOT/ignored/runs/next-ladder
until grep -q "CHAIN DONE" $D/chain.log; do sleep 300; done
cd "$ROOT/lab/spikes/2026-10-30-two-stage-hnet-smoke" || exit 1
echo "$(date -Iseconds) thin s2 rerun start" >> $D/chain.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN=ladder scripts/run_arms.sh "$D/thin" thin-outer-stages thin_outer_2stage:2 \
    -- train.max_steps=12000 "train.ladder_fracs=[]"
echo "$(date -Iseconds) thin s2 rerun done" >> $D/chain.log
