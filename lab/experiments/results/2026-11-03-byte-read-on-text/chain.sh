#!/usr/bin/env bash
# Queue: no other training model on the GPU -> suite gate -> 5 read arms on text.
# Plan: lab/experiments/planned/2026-11-03-byte-read-on-text.md
set -u
D=ignored/runs/byte-read-on-text
cd lab/spikes/2026-10-30-two-stage-hnet-smoke || exit 1
while pgrep -f "train.py" > /dev/null; do sleep 30; done
echo "$(date -Is) suite start" >> "$D/chain.log"
if ! HNET_TEST_CUDA=1 .venv/bin/python -m pytest -q > "$D/suite.log" 2>&1; then
    echo "$(date -Is) SUITE FAILED, arms not started (see suite.log)" >> "$D/chain.log"; exit 1
fi
echo "$(date -Is) suite ok: $(tail -1 "$D/suite.log")" >> "$D/chain.log"
TRAIN=ladder scripts/run_arms.sh "$D/read" byte-read-on-text reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 reflayout_2stage:3 reflayout_2stage:4 -- +model.stages.0.main_read.head_dim=64 train.max_steps=12000 "train.ladder_fracs=[]"
grep -q FAILED "$D/read/index.log" && { echo "$(date -Is) ARM FAILED (see read/index.log)" >> "$D/chain.log"; exit 1; }
echo "$(date -Is) CHAIN DONE" >> "$D/chain.log"
