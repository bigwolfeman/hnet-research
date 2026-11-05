#!/usr/bin/env bash
# Queue of lab/experiments/planned/2026-10-31-byte-and-bpe-controls.md. Starts after the router experiment ends.
#   nohup setsid scripts/run_controls_queue.sh > <log> 2>&1 &
# Waits for the router experiment's last index.log line ("ALL DONE <arms...>", which lists eoc_2stage:2),
# then runs the four control arms one after another from THIS checkout with the main checkout's venv.
set -uo pipefail
cd "$(dirname "$0")/.."
MAIN=${MAIN_CHECKOUT:-$(git rev-parse --show-toplevel)}
WAIT_LOG=${WAIT_LOG:-$MAIN/ignored/runs/end-of-chunk-router/index.log}
WAIT_RE=${WAIT_RE:-'ALL DONE .*eoc_2stage:2'}
export PY=${PY:-$MAIN/lab/spikes/2026-10-30-two-stage-hnet-smoke/.venv/bin/python}
export HNET_DATA_DIR=${HNET_DATA_DIR:-$HOME/data/hnet-bytes}
LOG_DIR=${LOG_DIR:-$MAIN/ignored/runs/byte-and-bpe-controls}
mkdir -p "$LOG_DIR"
echo "$(date -Is) WAITING for /$WAIT_RE/ in $WAIT_LOG (pid $$)" | tee -a "$LOG_DIR/queue.log"
until grep -Eq "$WAIT_RE" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
echo "$(date -Is) router experiment finished, starting controls" | tee -a "$LOG_DIR/queue.log"
exec scripts/run_arms.sh "$LOG_DIR" byte-and-bpe-controls byte_transformer:0 byte_transformer:1 bpe_transformer:0 bpe_transformer:1 -- "$@"
