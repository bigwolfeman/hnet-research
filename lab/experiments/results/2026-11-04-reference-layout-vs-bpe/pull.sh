#!/usr/bin/bash
# Regenerates summary.txt and val-curves.csv here. Run from the repo root.
S=lab/spikes/2026-10-30-two-stage-hnet-smoke; R=lab/experiments/results/2026-11-04-reference-layout-vs-bpe
$S/.venv/bin/python $S/scripts/pull_runs.py --group reference-layout-vs-bpe --curves $R/val-curves.csv > $R/summary.txt
