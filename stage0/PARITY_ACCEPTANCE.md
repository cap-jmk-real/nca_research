# Input-parity acceptance (engine)

Reproduce the published SOTA baseline numbers **before** trusting new `--text_feat` runs.

## Prerequisites

- `NCA_DATA_ROOT=~/nca_data` (or your usual data root)
- Frozen splits: `~/nca_data/stage0_splits/curated/{train,val,test}.jsonl` (57,675 / 3,465 / 3,530 rows, session-level, seed `20261026`)
- Existing selection artifacts: `~/nca_data/stage0_results/sota_baselines_20261026/gbdt_s0.json` (from the original `sota_gbdt.py` grid search)

## Check (seed 0)

From `~/nca_research/stage0`:

```bash
cd ~/nca_research/stage0
~/nca_venv/bin/python sota_gbdt.py 0
```

In `~/nca_data/stage0_results/sota_baselines_20261026/gbdt_s0.json` expect:

| Model | `test.top1` |
|-------|-------------|
| HistGB | **69.77** |
| logreg | **66.97** |

(`hgb` / `logreg` blocks; val grid unchanged.)

Then row-level preds (uses `gbdt_s0.json` hyperparams):

```bash
~/nca_venv/bin/python sota_v2.py 0
```

Sanity: curated test top-1 from `rowpreds_v2_s0.json` for `hgb` matches the HistGB test figure above.

## Neural default unchanged

```bash
~/nca_venv/bin/python train_eval.py --arch nca1d --seed 0 --steps 20000 --tag parity_check
```

Omit `--text_feat` (defaults to `hash48`). Observations must match pre-PR `build_trace_obs` / `build_obs`.

## Parity queue (append only)

```bash
cd ~/nca_research
python3 scripts/append_stage0_parity_queue.py
cd stage0 && ./queue.sh train
```

Do **not** insert lines into `queues/train.txt` — `queue.sh` IDs are `line:md5(line)`; only append.

## Analysis after parity runs

```bash
~/nca_venv/bin/python analyze_parity.py
```

Hermes OOD uses **session-level** bootstrap (same session clustering as `eval_hermes_ood.py`).
