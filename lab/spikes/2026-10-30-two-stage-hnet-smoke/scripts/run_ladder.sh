#!/usr/bin/env bash
# Compute ladder vs BPE: three long WSD runs in sequence on one GPU, each with rungs at 1/16 and 1/4 of its budget.
#   scripts/run_ladder.sh <log_dir>
# Experiment: lab/experiments/refuted/2026-11-01-compute-ladder-vs-bpe.md
# Safe to rerun after a crash or power loss: finished arms are skipped, a partial arm resumes (run_arms.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG_DIR=$1; GROUP=${GROUP:-compute-ladder-vs-bpe-r2}
export TRAIN=ladder
# BPE: 180,080 steps x 16 x 1024 tokens x 4.2647 mix bytes/token = 12.58B bytes (16 x the 786.4M H-Net budget)
scripts/run_arms.sh "$LOG_DIR" "$GROUP" blog_split_2stage:0
scripts/run_arms.sh "$LOG_DIR" "$GROUP" bpe_transformer:0 -- train.max_steps=180080 data.token.target_train_bytes=12582912000
scripts/run_arms.sh "$LOG_DIR" "$GROUP" matched_1stage:0
echo "$(date -Is) LADDER DONE" | tee -a "$LOG_DIR/index.log"
