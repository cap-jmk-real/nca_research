# Repository layout

| Path | Contents |
| --- | --- |
| `nca_text/` | Model and training code: `model.py` (TextNCA, GraphNCA, flat policy), `data.py`, `train.py`, `ssd.py`, `hypotheses.py`, `tb_log.py`. |
| `stage0/` | Benchmark runners (`run_e1_benchmark.py`, `run_e2_router_ablation.py`, `run_full_history_benchmarks.py` and others), data preparation (`make_splits.py`, `e1_prepare_bfcl_splits.py`), audits and `stage0_results/` with the committed result JSON files. |
| `scripts/` | Entry-point shell and Python scripts: `run_e1.sh`, `run_e2.sh`, `test_stage0_e1_e2_dry_run.sh`, `curate_traces.py`, `reproduce_headline.py`, `summarize_full_history.py`, `record_experiment.py`. |
| `tests/` | Unit tests run in CI: `test_e2_checkpoint_resolution.py`, `test_stage0_preflight.py`, plus `test_e1v2_eval.py`. |
| `config/`, `configs/` | Benchmark gates, specialist and publishing configuration, the dual-system E2 config. |
| `data/` | Small committed samples (`curated/`, `extra/`). Full data stays outside the repository. |
| `docs/` | This documentation site, and earlier working notes (literature, benchmarks, plans). |
| `paper/` | Manuscript sources, figures, generated tables and the supporting-information registry `paper/si/`. |
| `formal_proofs/` | Lean 4 proofs referenced in the paper. See `formal_proofs/README.md`. |
| `.github/workflows/` | `stage0-dry-run.yml` (E1 and E2 smoke chain) and `docs-sbom.yml` (docs build, SBOM, README commands). |
| `tools/` | `run_readme_blocks.py`, which executes the CI-marked commands in the README and docs. |

## Result files

| File | Produced by |
| --- | --- |
| `stage0/stage0_results/full_history_benchmarks.json` | `stage0/run_full_history_benchmarks.py` (Hermes table, seeds 0 to 2) |
| `stage0/stage0_results/e1_benchmark.json`, `e1_seed*.json` | `scripts/run_e1.sh` |
| `stage0/stage0_results/e2_router_ablation.json`, `e2_seed*.json` | `scripts/run_e2.sh` |
| `stage0/stage0_results/reviewer_suite_summary.json` | `stage0/run_reviewer_experiments.py` |
| `paper/si/experiments/EXP-*.json` | `scripts/record_experiment.py` (one record per registered run) |

Every number in the paper cites an `EXP-*` record. See `paper/README.md`.
