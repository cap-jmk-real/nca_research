# TensorBoard

**Date:** 2026-10-03

Training curves and benchmark scores are on the external SSD. Event files are not in this repo.

| | |
| --- | --- |
| Log directory | `/Volumes/Extreme SSD/nca_research/results/tb/` |
| URL | [http://127.0.0.1:6006/](http://127.0.0.1:6006/) |
| Launch | `scripts/tensorboard.sh` |

The script binds TensorBoard to `127.0.0.1`. If the Extreme SSD is not mounted, it exits and does not write under the repo or the home directory. The `tensorboard` package lives in `/Volumes/Extreme SSD/nca_research/tb-packages`. Pip’s cache and temp files for that install are under `/Volumes/Extreme SSD/nca_research/results/`.

## Runs

| Run | Path | What it shows |
| --- | --- | --- |
| `train` | `tb/train` | The 1d next-tool run. Live rows come from `results/logs/text_nca_1d.jsonl`. |
| `train_fineenvs` | `tb/train_fineenvs` | The Hermes-prior run that mixes FineEnvs common-task rows. Live rows come from `results/logs/text_nca_1d_fineenvs.jsonl`. Extra tags: `fineenvs/top1_nca`, `fineenvs/top1_flat`. |
| `train_grid` | `tb/train_grid` | The earlier grid-run summary in `results/text_nca_metrics.json` (epoch tail, not the full curve). |
| `benchmarks` | `tb/benchmarks` | One scalar per benchmark. Empty until scores are logged. |

## Training tags

Written when the jsonl row has that field:

- `loss/nca`
- `loss/flat`
- `val/top1_nca`
- `val/top1_flat`
- `val/top1_always_terminal`
- `perf/ms_per_step`
- `perf/samples_per_sec` (from the row, or `batch / (ms_per_step / 1000)` when the log has a step time)
- `lr`

`train_grid` uses the same tags. Its always-terminal series is the `majority` field in the old metrics file. That file’s history tail only has NCA loss and NCA top-1 per epoch; flat top-1, always-terminal, and the NCA step timing are a single summary point.

The 150k-step process that was already running does not import this logger. `scripts/tensorboard.sh` tails `text_nca_1d.jsonl` and appends new validation rows. The next `python3 -m nca_text.train` calls `nca_text.tb_log.log_train_row` at each validation, and `--eval-only` logs the final validation top-1.

## Benchmark tags

`scripts/log_benchmarks.py` writes one tag per key. A file or an inline object both work. `step` is the TensorBoard step (default 0), not a score.

```bash
python3 scripts/log_benchmarks.py --json '{"BFCL": 0.42, "NESTFUL": 0.31, "tau2": 0.15, "AppWorld": 0.22}'
python3 scripts/log_benchmarks.py scores.json
```

`scores.json` may be `{"step": 1, "scores": {"BFCL": 0.42}}`. No benchmark scores were on disk when this was added, so `tb/benchmarks` starts empty.

## How to rerun

```bash
scripts/tensorboard.sh
```

That refreshes events from the SSD metrics, starts a follower if one is not already running, and prints the URL. Stop the server with Ctrl-C. The follower pid is `tb/follow.pid`.

To import logs without serving:

```bash
python3 -m nca_text.tb_log --backfill
python3 -m nca_text.tb_log --follow
```
