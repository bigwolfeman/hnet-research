#!/usr/bin/bash
# Regenerates summary.txt and val-curves.csv here. Run from the repo root. Control: the R = 1 seeds of looped coarse main.
S=lab/spikes/2026-10-30-two-stage-hnet-smoke; R=lab/experiments/results/2026-11-04-parcae-coarse-main
$S/.venv/bin/python $S/scripts/pull_runs.py --group parcae-coarse-main --id zhpmgzlf --id mzq49grx --id zssjv0k6 \
    --curves $R/val-curves.csv > $R/summary.txt
