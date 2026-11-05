#!/usr/bin/env bash
# Queue: no other training model on the GPU -> suite gate -> 3 read arms (MainRead at stage 0) ->
# boundary_gain.py (stage 1 part all on read + plain; stage 0 part read on read).
# Plan: lab/experiments/planned/2026-11-03-bytes-read-main-by-content.md
set -u
D=ignored/runs/bytes-read-main-by-content
P=ignored/runs/main-gradient-coherence
cd lab/spikes/2026-10-30-two-stage-hnet-smoke || exit 1
while pgrep -f "train.py" > /dev/null; do sleep 30; done
echo "$(date -Is) suite start" >> "$D/chain.log"
if ! HNET_TEST_CUDA=1 .venv/bin/python -m pytest -q > "$D/suite.log" 2>&1; then
    echo "$(date -Is) SUITE FAILED, arms not started (see suite.log)" >> "$D/chain.log"; exit 1
fi
echo "$(date -Is) suite ok: $(tail -1 "$D/suite.log")" >> "$D/chain.log"
TRAIN=ladder scripts/run_arms.sh "$D/read" bytes-read-main-by-content reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.0.enc_layout=t4 model.stages.0.dec_layout=t4 model.stages.0.window=4 model.stages.1.enc_layout=T1t4 model.stages.1.dec_layout=t4T1 model.stages.1.window=1 +model.stages.0.main_read.head_dim=64 train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000
grep -q FAILED "$D/read/index.log" && { echo "$(date -Is) ARM FAILED (see read/index.log)" >> "$D/chain.log"; exit 1; }
echo "$(date -Is) arms done; boundary_gain start" >> "$D/chain.log"
OMP_NUM_THREADS=2 .venv/bin/python scripts/boundary_gain.py "$D"/read/reflayout_2stage_s{0,1,2} "$P"/synth/reflayout_2stage_s{0,1,2} --stage 1 --part all --out "$D/bg-s1-all.csv" --batches 8 > "$D/bg-s1-all.out" 2>&1
rc1=$?
OMP_NUM_THREADS=2 .venv/bin/python scripts/boundary_gain.py "$D"/read/reflayout_2stage_s{0,1,2} --stage 0 --part read --out "$D/bg-s0-read.csv" --batches 8 > "$D/bg-s0-read.out" 2>&1
rc2=$?
echo "$(date -Is) CHAIN DONE (boundary_gain exit $rc1 $rc2)" >> "$D/chain.log"
