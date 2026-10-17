# From Fast Drafts to Accepted Assets: code and data

This repository holds the code and data behind the article **"From Fast Drafts to Accepted Assets: Prompt Engineering, Retrieval-Augmented Generation and Human-in-the-Loop Review for Generative AI Marketing Content Workflows"** (manuscript submitted to *Information Technology and Management*).

The study compares four ways of producing marketing content: manual writing, prompt-only generation, retrieval-augmented generation (RAG), and RAG with a six-gate human-in-the-loop review (RAG+HITL). It measures each one up to the point of an **accepted asset**, meaning content a client would approve, rather than stopping at the first AI draft.

## Two evidence tiers, kept strictly apart

| Tier | What it is | File | Label in data |
|---|---|---|---|
| 1. Observed pilot | 10 briefs × 4 workflows (English), plus 10 Yoruba RAG+HITL outputs, for one fictional client in a coursework scenario. All outputs were timed, audited for errors and scored by two external experts. | `data/pilot_observed.csv` (Supplementary Table S1) | `OBSERVED (empirical pilot)` |
| 2. Simulated extension | 50,000 workflow–task records (5 workflows × 10,000 paired tasks, 78 variables), calibrated to the pilot and to published effect sizes. Used to explore other content types, sectors, markets and languages. | `data/simulated_tasks_50000.csv` | `SYNTHETIC` on every row |

**The simulated data are not measurements.** They show what the model projects under its stated assumptions. Do not report them as observed results.

The Hybrid RAG arm in the simulation is an assumption-driven ablation and was not observed.

## Repository layout

```
src/
  pilot_analysis.py         Tier 1: observed pilot data and exact paired tests
  generate_simulation.py    Tier 2: simulated dataset (seed 20260930)
  analyse_simulation.py     Calibration audit, descriptive tables, regressions, figures
  write_data_dictionary.py  Data dictionary for the simulated dataset
  build_workbook.py         Optional Excel workbook (dataset, calibration audit, tables)
data/
  pilot_observed.csv        Observed per-brief pilot records (Supplementary Table S1)
  simulated_tasks_50000.csv Simulated dataset, every row labelled SYNTHETIC
  data_dictionary.csv       Definitions of all 78 variables
results/                    Tables and summaries produced by the analysis scripts
figures/                    Figures produced by analyse_simulation.py
run_all.sh                  Runs the whole pipeline in order
```

## Reproducing the results

Requires Python 3.11 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

The full pipeline runs in under a minute on a laptop. The random seed is fixed (`20260930`), so the simulated dataset and every file in `results/` regenerate identically.

## Calibration

The anchor cell of the simulation (pilot client, Lagos, English social posts, off-peak) reproduces all 60 observed pilot measures within ±2 standard errors. The check is written to `results/A_calibration_audit.csv` and reported in Appendix A of the article.

## Scenario note

The case setting is a fictional, coursework-based agency scenario, and the pilot client ("EcoFresh Africa Ltd.") is fictional. No real company's data are included. The two expert raters' scores are anonymised.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0.

## Citation

See `CITATION.cff`. Please cite the article once published.
