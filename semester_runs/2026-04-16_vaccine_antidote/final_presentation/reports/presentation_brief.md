# Presentation Brief

## What Completed

- Vaccine alignment runs are complete for `rho=2`, `rho=5`, and `rho=10`.
- Vaccine downstream baseline reruns are complete for poison ratios `0.1`, `0.5`, and `0.8` at `rho=2`.
- Antidote poison-ratio sweep is complete for `0.1`, `0.2`, `0.5`, `0.8`, and `1.0` with `dense_ratio=0.1`.
- Antidote dense-ratio follow-ups are complete for `0.05`, `0.1`, and `0.2` at `poison_ratio=0.2`.

## Headline Results

- Vaccine alignment poison moderation improves from `42.80` at `rho=2` to `34.00` at `rho=10`.
- Vaccine downstream baseline keeps SST2 accuracy above `93%` across the completed poison-ratio reruns, with poison moderation scores clustered around `79.6` to `82.2`.
- Antidote maintains strong SST2 accuracy through poison ratio `0.5` and then drops to `6.31%` at poison ratio `1.0`.
- In the dense-ratio sweep, the best downstream accuracy is `95.18%` at `dense_ratio=0.05`, while the strongest safety improvement is `58.00` at `dense_ratio=0.2`.

## Recommended Slide Order

1. `cards/summary_cards.png`
2. `plots/overview_dashboard.png`
3. `plots/safety_vs_utility.png`
4. `plots/canonical_metric_matrix.png`
5. `plots/vaccine_alignment.png`
6. `plots/vaccine_poison_ratio.png`
7. `plots/antidote_poison_ratio.png`
8. `plots/antidote_dense_ratio.png`
9. `plots/status_breakdown.png`
10. `plots/completion_timeline.png`
11. `tables/presentation_runs_table.png`
12. `tables/completed_run_catalog.png`
13. `tables/superseded_attempts_table.png`

