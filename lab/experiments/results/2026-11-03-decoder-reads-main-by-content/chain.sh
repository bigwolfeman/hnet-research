#!/usr/bin/env bash
# Queue: wait for main gradient coherence to end (one training model on the GPU) -> suite gate -> 3 read arms ->
# boundary_gain.py on the read arms and the plain synthetic arms of main gradient coherence.
# Plan: lab/experiments/planned/2026-11-03-decoder-reads-main-by-content.md
set -u
D=ignored/runs/decoder-reads-main-by-content
P=ignored/runs/main-gradient-coherence
cd lab/spikes/2026-10-30-two-stage-hnet-smoke || exit 1
echo "$(date -Is) waiting for main gradient coherence" >> "$D/chain.log"
until grep -q "CHAIN DONE\|FAILED" "$P/chain.log" 2>/dev/null; do sleep 60; done
while pgrep -f "train.py" > /dev/null; do sleep 30; done
echo "$(date -Is) suite start" >> "$D/chain.log"
if ! HNET_TEST_CUDA=1 .venv/bin/python -m pytest -q > "$D/suite.log" 2>&1; then
    echo "$(date -Is) SUITE FAILED, arms not started (see suite.log)" >> "$D/chain.log"; exit 1
fi
echo "$(date -Is) suite ok: $(tail -1 "$D/suite.log")" >> "$D/chain.log"
TRAIN=ladder scripts/run_arms.sh "$D/read" decoder-reads-main-by-content reflayout_2stage:0 reflayout_2stage:1 reflayout_2stage:2 -- data=synth_recall model.stages.0.enc_layout=t4 model.stages.0.dec_layout=t4 model.stages.0.window=4 model.stages.1.enc_layout=T1t4 model.stages.1.dec_layout=t4T1 model.stages.1.window=1 +model.stages.1.main_read.head_dim=64 train.max_steps=6000 "train.ladder_fracs=[]" train.eval_every=1000
echo "$(date -Is) arms done; boundary_gain start" >> "$D/chain.log"
OMP_NUM_THREADS=2 .venv/bin/python scripts/boundary_gain.py "$D"/read/reflayout_2stage_s{0,1,2} "$P"/synth/reflayout_2stage_s{0,1,2} --out "$D/boundary-gain-cells.csv" --batches 8 > "$D/boundary-gain.out" 2>&1
echo "$(date -Is) CHAIN DONE (boundary_gain exit $?)" >> "$D/chain.log"
