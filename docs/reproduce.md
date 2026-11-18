# Reproduce the paper results

Two levels are supported. The table recomputation reads committed results. The smoke run executes the real pipeline at toy size. Neither retrains the paper models. The full runs are listed at the end.

## 1. Recompute the Hermes table from stored results

<!-- ci:run -->
```bash
python scripts/summarize_full_history.py
```

Output:

```text
test turns: 3530, seeds: [0, 1, 2]
HistGradientBoosting    69.47 +- 0.26
RandomForest            67.88 +- 0.34
GraphNCA_tree           65.27 +- 0.31, 59182 params
GraphNCA_meta           65.33 +- 0.26, 59207 params
GraphNCA_star           66.19 +- 0.15, 59182 params
GraphNCA_1d             64.64 +- 0.79, 59182 params
```

Values are mean and sample standard deviation of per-seed test top-1 in percent. The source is `stage0/stage0_results/full_history_benchmarks.json`, registered as `EXP-20261102-001`.

## 2. Smoke run of the E1 and E2 pipeline

The smoke run uses the public BFCL split, 150 training steps and one seed. It checks that the code paths run and that these files appear: `stage0/stage0_results/e1_checkpoints/seed0_star.pt`, `seed0_meta.pt`, `seed0_1d.pt` and `stage0/stage0_results/e2_router_ablation.json`. It does not reproduce paper numbers, and it overwrites `e1_benchmark.json` and `e2_router_ablation.json` in the working tree.

<!-- ci:run -->
```bash
export NCA_DATA_ROOT="$PWD/.nca_data"
export RESEARCH_BACKUP_OPTIONAL=1
bash scripts/test_stage0_e1_e2_dry_run.sh
git checkout -- stage0/stage0_results
```

Expected last line: `stage0 E1→E2 dry-run chain OK`.

## 3. Full runs

These need a GPU and hours of compute, so CI does not run them.

```bash
export NCA_DATA_ROOT=/path/to/writable/dir
bash scripts/run_e1.sh
bash scripts/run_e2.sh
```

The Hermes full-history table needs the curated Hermes splits at `$NCA_DATA_ROOT/stage0_splits/curated` (`train.jsonl`, `val.jsonl`, `test.jsonl`), written by `stage0/make_splits.py` from the files that `scripts/curate_traces.py` produces.

```bash
python stage0/run_full_history_benchmarks.py --out full_history_benchmarks.json
python scripts/summarize_full_history.py --results "$NCA_DATA_ROOT/full_history_benchmarks.json"
```

The summary of the full run should agree with the table above up to run-to-run variation.

## Which numbers each level checks

| Level | Checks |
| --- | --- |
| Table recomputation | The aggregation from per-seed values to the paper's Hermes means and standard deviations. |
| Smoke run | That E1 and E2 execute end to end and write their outputs. No accuracy claim. |
| Full run | Training from scratch and the accuracy numbers themselves. |

`scripts/reproduce_headline.py` prints a fixed ledger of headline values and recomputes nothing.
