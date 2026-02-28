# Directory Structure Convention

This repository uses a conservative, script-first layout focused on reproducible data prep and training.

## Canonical Layout

- `scripts/`: runnable training/inference entrypoints.
- `data/scripts/`: dataset preparation and conversion utilities.
- `notebooks/`: exploratory and tutorial notebooks.
- `artifacts/`: generated datasets, model checkpoints, and temporary caches (gitignored).
- `tasks/`: lightweight execution tracking and lessons.
- `.cursor/plans/`: local planning docs.

## Current Structure (Target)

- `scripts/train_paligemma_origa.py`
- `data/scripts/convert_origa.py`
- `notebooks/dataset.ipynb`
- `notebooks/how_to_finetune_paligemma_on_detection_dataset.ipynb`
- `artifacts/origa_paligemma/` (runtime output root)

## Rules

1. Keep reusable logic in Python modules; keep `scripts/` as thin CLIs where practical.
2. Put generated files under `artifacts/` and avoid committing large outputs.
3. Keep one clear entrypoint per workflow:
   - dataset conversion: `data/scripts/convert_origa.py`
   - train + optional prepare: `scripts/train_paligemma_origa.py`
