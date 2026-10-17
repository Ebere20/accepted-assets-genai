#!/usr/bin/env bash
# Reproduce every observed and simulated result, from the repository root.
set -euo pipefail
mkdir -p data results figures
python src/pilot_analysis.py          # Tier 1: observed pilot -> data/pilot_observed.csv, results/pilot_summary.json
python src/generate_simulation.py     # Tier 2: simulated extension -> data/simulated_tasks_50000.csv (seed 20260930)
python src/analyse_simulation.py      # results/*.csv, results/summary.json, figures/*.png
python src/write_data_dictionary.py   # data/data_dictionary.csv
# Optional: Excel workbook with calibration audit and summary tables
# python src/build_workbook.py        # data/simulated_tasks_50000.xlsx
