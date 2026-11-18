# nca_research

Code, results and manuscript sources for *Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection* (Julian M. Kleber).

[![CI](https://github.com/cap-jmk-real/nca_research/actions/workflows/docs-sbom.yml/badge.svg)](https://github.com/cap-jmk-real/nca_research/actions/workflows/docs-sbom.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

## What this is

An agent harness has to pick the next tool at every turn. This repository tests whether a small graph neural cellular automaton (GNCA, 35k to 59k parameters) can make that choice from a relational graph of the recent trace, on CPU in under a millisecond, and how it compares with Random Forest and HistGradientBoosting baselines on the same splits.

The repository holds:

- `nca_text/`: the model and training code (TextNCA, GraphNCA, flat baseline).
- `stage0/`: the benchmark runners, data preparation, audits and the committed result files in `stage0/stage0_results/`.
- `paper/`: manuscript sources and the supporting-information registry (`paper/si/`).
- `formal_proofs/`: the Lean 4 proofs referenced in the paper.

A full map is in [docs/layout.md](docs/layout.md). The documentation site is built with MkDocs Material from `docs/` and `mkdocs.yml`.

## Quickstart

Python 3.11 or newer. The CPU build of PyTorch is enough for the checks below.

<!-- ci:run -->
```bash
python -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt scikit-learn huggingface_hub scipy
python -m unittest tests.test_e2_checkpoint_resolution tests.test_stage0_preflight -v
```

## Reproducing the paper numbers

The Hermes top-1 numbers in the paper come from `stage0/stage0_results/full_history_benchmarks.json`. The first command recomputes the table from that committed file in a second and needs no GPU.

<!-- ci:run -->
```bash
python scripts/summarize_full_history.py
```

Expected output:

```text
test turns: 3530, seeds: [0, 1, 2]
HistGradientBoosting    69.47 +- 0.26
RandomForest            67.88 +- 0.34
GraphNCA_tree           65.27 +- 0.31, 59182 params
GraphNCA_meta           65.33 +- 0.26, 59207 params
GraphNCA_star           66.19 +- 0.15, 59182 params
GraphNCA_1d             64.64 +- 0.79, 59182 params
```

These are mean and sample standard deviation of test top-1 (percent) over seeds 0, 1 and 2, on 3,530 curated Hermes test turns. They match the Hermes values in the manuscript abstract and experiment record `EXP-20261102-001`. The tree variant with K=8 and label smoothing, reported as 68.92 +- 0.47 in the abstract, is stored in `stage0/stage0_results/reviewer_suite_summary.json` under `hermes_factorial_and_contractive`.

### Smoke run and full run

A smoke run checks that the pipeline executes and writes its output files. It does not reproduce paper numbers. The full run does.

| Run | Command | Checks | Time |
| --- | --- | --- | --- |
| Smoke | `bash scripts/test_stage0_e1_e2_dry_run.sh` | E1 and E2 code paths on the public BFCL split with 150 steps, one seed. Checks that star, meta and 1d checkpoints and `e2_router_ablation.json` are written. | about a minute on a CI runner (CPU) |
| Full E1 and E2 | `bash scripts/run_e1.sh` then `bash scripts/run_e2.sh` | 150,000 steps, seeds 0 to 2, writes `e1_benchmark.json` and `e2_router_ablation.json`. | 150,000 training steps per run, GPU recommended |

The smoke run overwrites `stage0/stage0_results/e1_benchmark.json` and `e2_router_ablation.json` in your working tree. Restore them with `git checkout -- stage0/stage0_results` afterwards.

<!-- ci:run -->
```bash
export NCA_DATA_ROOT="$PWD/.nca_data"
export RESEARCH_BACKUP_OPTIONAL=1
bash scripts/test_stage0_e1_e2_dry_run.sh
git checkout -- stage0/stage0_results
```

The full Hermes table needs the curated Hermes splits under `$NCA_DATA_ROOT/stage0_splits/curated` (`train.jsonl`, `val.jsonl`, `test.jsonl`). `stage0/make_splits.py` writes them from the curated trace files that `scripts/curate_traces.py` produces. With the splits in place:

```bash
export NCA_DATA_ROOT=/path/to/writable/dir
python stage0/run_full_history_benchmarks.py --out full_history_benchmarks.json
python scripts/summarize_full_history.py --results "$NCA_DATA_ROOT/full_history_benchmarks.json"
```

`scripts/reproduce_headline.py` prints a fixed ledger of headline values. It does not recompute anything. Use `summarize_full_history.py` for a check against stored results.

Every command in a code block marked for CI in this README and in `docs/reproduce.md` is executed by `tools/run_readme_blocks.py` in the `docs-sbom` workflow, so the quickstart and the smoke run cannot drift from what CI runs.

## Data and results

Large datasets, caches and checkpoints are not in the repository. `data/curated/` and `data/extra/` hold small samples only. The public benchmark data is downloaded on first use. See [docs/data-root.md](docs/data-root.md) for the storage rules and [docs/layout.md](docs/layout.md) for where each result file lives.

## Citation

Cite with the metadata in [CITATION.cff](CITATION.cff). Author: Julian M. Kleber, ORCID [0000-0001-5518-0932](https://orcid.org/0000-0001-5518-0932). Contact: julian.m.kleber@gmail.com.

## Licence

Code: MIT, see [LICENSE](LICENSE). The manuscript text is released under CC BY 4.0.
