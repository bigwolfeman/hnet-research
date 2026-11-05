#!/usr/bin/bash
# Regenerates summary.txt and val-curves.csv here. Run from the repo root.
S=lab/spikes/2026-10-30-two-stage-hnet-smoke; R=lab/experiments/results/2026-11-03-grown-hierarchy-keeps-main
$S/.venv/bin/python $S/scripts/pull_runs.py --group grown-hierarchy-p1 --group grown-hierarchy \
    --id akqhdkkz --id ipcxldpl --id zplsyk1o --id 5d3o336d --id 7dffta9c --curves $R/val-curves.csv > $R/summary.txt
