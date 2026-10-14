# Results master (paper + SI)

Updated from 19 experiment records.

See generated tables:

- [S1 experiment registry](tables/S1_experiment_registry.md)
- [S2 next-tool top-1](tables/S2_next_tool_top1.md)
- [S3 claims ledger](tables/S3_claims_ledger.md)

## Inclusion rules

1. Main-text numbers cite `EXP-*` with `status: primary` (or `preliminary` with explicit caveat).
2. `status: smoke` never supports architecture claims (budget < 50k steps).
3. Every new daemon / hypotheses completion must call `python scripts/record_experiment.py register --from-hypotheses-summary` then `build-tables`.
4. Distillation teachers ≤ 9B; record teacher id in `training` / `baselines`.
