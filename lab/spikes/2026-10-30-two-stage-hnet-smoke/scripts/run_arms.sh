#!/usr/bin/env bash
# Run training arms one after another on one GPU.
#   scripts/run_arms.sh <log_dir> <wandb_group> <experiment:seed>... [-- <extra hydra overrides>]
# Each arm runs in a fixed Hydra run dir, <log_dir>/<name>/. Rerunning the same command is safe after a crash or a
# power loss: an arm with summary.json is skipped, and an arm with resume.pt continues from it (train.resume_every).
# A failed arm is logged loudly (FAILED in <log_dir>/index.log) and the next arm starts.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}   # PY=<other venv python> runs this checkout's code with another venv
TRAIN=${TRAIN:-router}   # train config group (configs/train/<TRAIN>.yaml)
LOG_DIR=$(realpath -m "$1"); GROUP=$2; shift 2
ARMS=(); while (($#)) && [[ $1 != -- ]]; do ARMS+=("$1"); shift; done
(($#)) && shift
mkdir -p "$LOG_DIR"
for arm in "${ARMS[@]}"; do
  exp=${arm%%:*}; seed=${arm##*:}; name="${exp}_s${seed}"; run_dir="$LOG_DIR/$name"
  if [[ -f $run_dir/summary.json ]]; then
    echo "$(date -Is) SKIP  $name (summary.json exists)" | tee -a "$LOG_DIR/index.log"; continue
  fi
  verb=START; [[ -f $run_dir/resume.pt || -f $run_dir/resume_prev.pt ]] && verb=RESUME
  echo "$(date -Is) $verb $name" | tee -a "$LOG_DIR/index.log"
  echo "===== $(date -Is) $verb =====" >> "$LOG_DIR/$name.log"
  OMP_NUM_THREADS=2 "$PY" train.py experiment="$exp" train="$TRAIN" seed="$seed" hydra.run.dir="$run_dir" \
    wandb.project=hnet-router wandb.group="$GROUP" wandb.name="$name" "$@" >> "$LOG_DIR/$name.log" 2>&1
  rc=$?   # read before any other expansion: $(date) below would reset $?
  if ((rc == 0)); then
    echo "$(date -Is) DONE  $name $(grep -o '"final_val_bpb": [0-9.]*' "$LOG_DIR/$name.log" | tail -1)" | tee -a "$LOG_DIR/index.log"
  else
    sig=""; ((rc > 128)) && sig=" = signal $((rc - 128))"
    echo "$(date -Is) FAILED $name (exit $rc$sig), see $LOG_DIR/$name.log" | tee -a "$LOG_DIR/index.log"
  fi
done
echo "$(date -Is) ALL DONE ${ARMS[*]}" | tee -a "$LOG_DIR/index.log"
