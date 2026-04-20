# Final Presentation Artifacts

This folder is generated from completed vaccine and antidote runs found in the repository.

## Contents

- `data/all_detected_runs.csv`: codebase-wide audit of candidate log files.
- `data/completed_success_runs.csv`: successful completed runs only.
- `data/presentation_runs.csv`: deduplicated canonical runs used in the plots.
- `cards/`: summary cards for title or takeaway slides.
- `plots/`: presentation-ready figures.
- `tables/`: slide-friendly tables and run catalogs.
- `vaccine/`: staged `.out` and `.err` files for successful vaccine-side runs.
- `antidote/`: staged `.out` and `.err` files for successful antidote-side runs.
- `reports/`: markdown summaries for speaker notes and handoff.

## Refresh

Run:

```bash
python /home/hice1/rswaminathan38/scratch/Research/final_presentation/generate_artifacts.py
```
