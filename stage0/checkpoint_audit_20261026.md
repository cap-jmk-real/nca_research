# Engine research jobs: checkpoint audit, 26 Oct 2026

Read-only audit at 13:10 Berlin time (UTC+1); the engine clock is UTC-4, so add 5 hours to engine times. Nothing was changed. This file is the only thing written.

| # | Job | Status | Checkpoint | Resume in the code | Classification |
|---|---|---|---|---|---|
| 1 | q_train: B (nca1d/flat/gru/tx ×3 seeds), H1/H3, T (nca2d/graph), G1p (WD×3 seeds), H1b_nocw (flat/gru/tx ×3 seeds): 45 jobs | **done**, 45/45 OK, 0 failed; queue idle since 12:35 | `~/nca_data/stage0_runs/<ts>_<tag>_curated_s<seed>/{best.pt,last.pt}`, final step, 08:07–12:31 | **No.** train_eval.py has no `--resume`, only saves best/last, and `mkdir(exist_ok=False)` | already done |
| 2 | q_train: Track D distillation D1/D2 (9 jobs) | **held**, commented out (teacher below floor) | none | no | not started; fresh start only if the gate is reopened |
| 3 | q_teacher: qwen test, gemma test, qwen val, qwen 5k pilot | **done**, 4/4 OK; idle since 10:32 | outputs `teachers_20261026/*`, `teachers_20261026_pilot5k/*` (no model checkpoint needed) | no (inference) | already done |
| 4 | q_teacher: qwen 50k train | **held**, commented out (35.0% < floor 52.9%) | none | no | not started; fresh start only after a prompt rethink |
| 5 | q_denoise: DenoiseOpt phase 2, run_all `--cases`; search seed 2026072701 (hybrid_lstm, 5000 iters) | **running** on GPU since 10:22 | `~/reel_runs/denoiseopt_20261026T0522/repo/brand/artifacts/phase2_experiments/search_2026072701/hybrid_lstm/{checkpoint.json,policy.pt,champ_cell.pt}`, iter **1650/5000**, 13:09; champ 0.9715 (plateau since iter 877) | **Yes.** bench_meta_approaches_5k resumes from checkpoint.json by default (fresh only with `--force-fresh`); run_all reuses summaries (`--skip-if-summary true`) | can resume from a checkpoint |
| 6 | DenoiseOpt search seed 1902771841 | **done** (reused summary from `meta_approach_compare/hybrid_lstm/summary.json`, per config) | summary | n/a | already done (reused) |
| 7 | DenoiseOpt search seed 2026072702 | **queued** inside run_all, after seed 2026072701 | none yet | yes, once it starts | needs a fresh start: it has never run, so it has no checkpoint (will resume automatically if interrupted later) |
| 8 | DenoiseOpt holdout eval, α sweep (0.5–0.9), compute table, pre-registered cases | **queued** inside run_all, after the searches | none. The holdout_eval / alpha_sensitivity / compute_budget files in the artifacts folder are **old (21 Oct)** and get overwritten | evaluation only (minutes, no training) | needs a fresh start: evaluation steps, nothing to checkpoint |
| 9 | q_denoise_post: post.sh (waits for PID 1560423, copies results, paired bootstrap CI, restic backup) | **running/waiting** (polls every 300 s) | n/a | n/a | will finish by itself; no action |
| 10 | q_sota: GBDT seed 0 | **done** 12:49 | `sota_baselines_20261026/gbdt_s0.json` (result; model not saved) | **No** (argv seed, writes JSON) | already done |
| 11 | q_sota: GBDT seed 1 | **running** on CPU since about 12:52 | none | no | if interrupted: fresh start (no model checkpointing; about 20 min per seed) |
| 12 | q_sota: GBDT seed 2 | **queued** (same for-loop) | none | no | needs a fresh start: not started yet; cheap |
| 13 | Hermes FT grid: `hermes_ft/train_ft.py`, 720 configs (6 models × scratch/ft × drop/other × 2 lr × 3 seeds × 5 folds), CPU Pool 12 | **running**, **410/720** done; **orphan** (PPID 1, no tmux session), since about 12:43 | per-config result JSON in `~/nca_data/stage0_results/hermes_ft_20261026/runs/`; "ft" variants load the curated `best.pt` from stage0_runs | **Job-level resume:** skips configs whose JSON exists. No in-run checkpoint (1500 steps per config) | can resume (config level). Only the configs in progress are lost if stopped |
| 14 | Parallel `curate` tmux (curate_engine + make_splits), PID 3978454 | process still listed, started about 10:11 | n/a | n/a | not part of the queues; check whether it is stale (not touched) |
| 15 | Announced NCA follow-ups (paired GBDT vs NCA/transformer, GBDT on Hermes, GBDT without result text) | **not found**: neither in a queue file nor as a process | none | — | still need to be queued (sota_gbdt has no resume) |

## Findings and risks
- **Nothing in the NCA training has to be rerun.** All 45 q_train runs are finished and have best.pt/last.pt. train_eval.py has **no** resume support, so an interrupted run would start from step 0.
- **queue.sh identifies jobs by `line number:md5(line)`.** If lines are inserted or deleted in queue files, already finished jobs get a new ID and are **run again**. Only append to queue files.
- The DenoiseOpt seed-2026072701 search has been on a plateau since iter 877 (champ 0.9715 < N2N gate 0.975). About 3350 iters remain (about 2.5 to 3 min per 50 iters, so about 3 h), then seed 2026072702 runs fresh (about 7 to 8 h at the same rate), then holdout/α. Resume is safe: checkpoint every 25 iters.
- The Hermes FT grid runs without tmux or a log owner (orphan). A restart with `NPROC=12 python train_ft.py` in `hermes_ft/` continues where it stopped.
- Old phase-2 result files (21 Oct) are still in the artifacts folder. Do not cite them as new results.
