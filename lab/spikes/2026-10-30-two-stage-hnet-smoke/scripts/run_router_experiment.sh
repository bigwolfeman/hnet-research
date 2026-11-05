#!/usr/bin/env bash
# Arms of lab/experiments/refuted/2026-10-30-end-of-chunk-router.md, main-prediction arms first.
cd "$(dirname "$0")"
exec ./run_arms.sh "${LOG_DIR:-../../../ignored/runs/end-of-chunk-router}" end-of-chunk-router \
  eoc_2stage:0 lookahead_2stage:0 eoc_2stage:1 lookahead_2stage:1 \
  eoc_1stage:0 lookahead_1stage:0 eoc_1stage:1 lookahead_1stage:1 \
  eoc_lag_2stage:0 eoc_2stage:2 lookahead_2stage:2 -- "$@"
