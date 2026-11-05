#!/usr/bin/bash
# Regenerates summary.txt and val-curves.csv here, and loop-passes.txt (CPU). Run from the repo root.
S=lab/spikes/2026-10-30-two-stage-hnet-smoke; R=lab/experiments/results/2026-11-03-looped-coarse-main; L=ignored/runs/next-ladder/looped
$S/.venv/bin/python $S/scripts/pull_runs.py --group looped-coarse-main --curves $R/val-curves.csv > $R/summary.txt
nice -n 19 $S/.venv/bin/python $R/loop_passes.py $L/loop{1,2,4}_1stage_s0 $L/loop{1,2,4}_1stage_s2 > $R/loop-passes.txt
