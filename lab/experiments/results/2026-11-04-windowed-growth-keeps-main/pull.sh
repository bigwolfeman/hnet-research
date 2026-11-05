#!/usr/bin/bash
# Regenerates summary.txt and val-curves.csv here. Run from the repo root.
S=lab/spikes/2026-10-30-two-stage-hnet-smoke; R=lab/experiments/results/2026-11-04-windowed-growth-keeps-main
$S/.venv/bin/python $S/scripts/pull_runs.py --group windowed-growth-p1 --group windowed-growth --group windowed-scratch \
    --curves $R/val-curves.csv > $R/summary.txt
