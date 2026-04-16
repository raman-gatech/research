# Vaccine and Antidote Semester Runs

This folder is a curated snapshot of my semester run artifacts for the `Vaccine` and `Antidote` experiments.

Included here:

- `reports/`: completed-jobs tables, hyperparameter metadata, and presentation summary
- `ppt_assets/`: slide-ready tables and plots
- `scripts/`: exact Slurm scripts used for the completed runs and current follow-up sweeps
- `logs/`: selected `.out` and `.err` files for completed jobs
- `results/`: evaluation outputs used in the summary tables

Notes:

- Only my run artifacts and presentation content are included here.
- Large checkpoints are not included.
- The Antidote `poison_ratio=0.2` configuration was run twice (`4051776` and `4051777`) and shares the same artifact path, so it should be treated as a rerun rather than a separate configuration.
- Current follow-up dense-ratio jobs were resubmitted on `H200` on April 16, 2026; see `reports/current_submission_status_2026-04-16.md`.
