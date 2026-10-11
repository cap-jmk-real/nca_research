---
name: nca-auto-research
description: >-
  Runs the nca_research auto-research loop: Klaut gateway literature search,
  benchmark gate scoreboard, hypothesis backlog prioritization, and train/eval
  cycles until the text NCA beats small local LLMs on allowed benchmarks.
  Use when extending NCA agent research, distillation (≤9B cap), Klaut gateway,
  auto-research, benchmark gates, or beating Qwen/Ollama baselines on Hermes or
  BFCL/NESTFUL/tau2/AppWorld.
---

# NCA auto-research (Klaut + gates)

## Prerequisites

- Machine on the same LAN as **engine** (`192.168.10.29`) for live Klaut calls, or rely on SSD cache.
- External SSD mounted: `/Volumes/Extreme SSD/nca_research`.
- Env: `NCA_DATA_ROOT=/Volumes/Extreme SSD/nca_research/results`, `TMPDIR` on SSD.

Read first:

- [docs/auto-research-loop.md](../../../docs/auto-research-loop.md)
- [docs/hypotheses-backlog.md](../../../docs/hypotheses-backlog.md)
- [config/benchmark_gates.json](../../../config/benchmark_gates.json)

## Objective

The score that matters is accuracy the runnable harnesses record: curated val next-tool, local Hermes, BFCL AST, BFCL irrelevance, Nestful full sequence. A new hypothesis names one of those. τ2 airline, τ2 retail, and AppWorld stay blocked until those harnesses can run. Do not mark them passed and do not invent scores. Do not edit `config/benchmark_gates.json` unless the change is paired with real NCA and LLM numbers from the same harness. A curated-val gain alone is not success while another runnable harness is still failing or unscored.

## Each cycle (do in order)

1. **Measure** — Run `python scripts/auto_research.py` from repo root.  
   - Add `--klaut-search` when engine is reachable. That runs 2–3 short queries for the top failing gate (`GAP_TO_QUERY` in `scripts/auto_research.py`). It does not start or stop training.  
   - Add `--klaut-report` only when intake set `report_recommended` (a new architecture family with code).

2. **Read output** — Open `results/auto_research/latest.json` on the SSD. Note `failing_gates`, `harness_status`, `suggested_hypotheses`, `backlog_ranked`. The intake ledger is `results/auto_research/klaut_intake.json`.

3. **Literature** — Use the intake from step 1. Do not search only `next tool prediction imitation learning agent traces`. Reject a hit with no code that is not about text NCA or next-tool prediction; the ledger stores the query, hit id, and reason so the next tick does not report those papers as new. Add a backlog row only for a concrete code-backed change that is not already listed, names a runnable harness, and does not require stopping a 150k run. If nothing qualifies, leave `docs/hypotheses-backlog.md` unchanged.

4. **Pick one hypothesis** — Choose the highest-priority id in `suggested_hypotheses` that is not yet measured. If empty, take priority 1 from [hypotheses-backlog.md](../../../docs/hypotheses-backlog.md).

5. **Implement minimally** — Match existing `nca_text/` style. No scope creep.

6. **Train/eval — long enough** — Default **≥150k** AdamW steps for any architectural claim (matches mix13). ≤8k is plumbing only; 20k is not a verdict. Recent Sudoku / iterative-reasoning work (e.g. Giannoulis et al. NeurIPS 2025, arXiv:2509.22023) shows structured policies need long imitation before they settle — treat NCA the same. Daemon: `NCA_HYP_STEPS=150000`. Log to SSD TensorBoard.

7. **Score paired baselines** — Whenever benchmarks change, run **both**:  
   - `python scripts/run_text_benchmarks.py` (NCA checkpoint)  
   - `python scripts/score_local_next_tool.py` (Qwen2.5-1.5B)  
   Do not update `benchmark_gates.json` pass/fail without LLM comparison on the same task.

8. **Record for the paper (required)** — Register an SI experiment and rebuild tables:
   ```bash
   python scripts/record_experiment.py register --from-hypotheses-summary   # or backfill
   python scripts/record_experiment.py build-tables
   ```
   Fill `claims_supported` / `claims_not_supported` in the new `paper/si/experiments/EXP-*.json`.  
   Cite the `EXP-*` id in `docs/hypotheses-backlog.md` and never put a main-text number without an id.  
   Smoke (`status: smoke`, &lt;50k steps) must not support architecture claims.  
   Also re-run `auto_research.py`.

9. **Commit regularly (required)** — Do **not** batch a whole day of work into one commit.
   - After each finished hypothesis / SI register / dataset compile milestone: `git add` relevant paths, commit with a conventional message, `git push -u origin <branch>`.
   - Prefer one logical unit per commit (one EXP, one script, one doc note).
   - Branch: `cursor/<name>-1754` (current long-run: `cursor/auto-research-klaut-1754`).
   - Always include `paper/si/` updates when metrics change.
   - On every 6h supervision tick: if `git status` is dirty, commit and push before ending the turn.

## Distillation cap (9B)

- Allowed: label/trace imitation from **`qwen3.5:9b`** or smaller; observation text cleanup; verifier-filtered rows ([FineEnvs](../../../docs/multi-harness-and-message-passing.md)); **specialist LM students** (Python / TypeScript / Shell / Orchestrator) via SFT/LoRA or logit KL from ≤9B teachers ([specialist-coder-stack.md](../../../docs/specialist-coder-stack.md)).
- Forbidden: distilling 9B **weights or hidden states** into `TextNCA`; unlimited 27B teacher SFT into respond-heavy mix.

## Specialist coder path

After NCA long runs, or in parallel on CPU/disk: compile datasets → distill on engine → eval harnesses (`LiveCodeBench`, `SWE-bench_Lite`, `Terminal-Bench`, `tsc_repo_smoke`). Plug order: orchestrator → specialist → tools → verifier. Register every harness score as `EXP-*`.

## Klaut API (summary)

| Action | Command |
| --- | --- |
| Ping | `python scripts/klaut_gateway.py ping` |
| Search | `python scripts/klaut_gateway.py search "your query"` |
| Report | `python scripts/klaut_gateway.py report "topic sentence"` |

Full paths: `POST /v1/research/papers/search`, `POST /v1/research/reports` on `KLAUT_RESEARCH_URL` (default engine NodePort).

Cache: `/Volumes/Extreme SSD/nca_research/results/cache/klaut/`.

## Stop rule

Stop iterating when `auto_research.py` reports **no failing gates** except documented blockers (τ2/AppWorld install). Then add **qwen3.5:9b** as baseline and tighten gates per [auto-research-loop.md](../../../docs/auto-research-loop.md).

## Minimize tokens (agent + algorithms)

See [efficiency.md](../../../docs/efficiency.md).

- Supervision ticks: **12h**. First action: `python scripts/status_lean.py`. Reply with that block only.
- Do not re-read docs or dump `console.log` / `daemon.log` each tick.
- Distill/eval LMs: route/tool tags only (`num_predict` ≤ **24**). Never ask teachers for essays.
- LLM baselines: truncate obs to 160 chars (same as NCA `TEXT_CAP`).
- TextNCA: classification only; sparse val (5k); do not increase K or sequence length without a gate win.
- If NCA/specialists use more tokens/compute than a small transformer for worse metrics, prefer the cheaper baseline and log that in SI.

## Do not

- Skip the flat baseline when claiming cellular binding helps.
- Treat Klaut synthesis narrative as measured fact — verify against retrieved paper records.
- Run full 150k training without a cheaper 8k curated ablation when the hypothesis is architectural.
- Treat a curated-val gain as success while local Hermes, BFCL, or Nestful is still failing or unscored.
- Mark τ2 or AppWorld passed, or invent scores for them.
- Edit `benchmark_gates.json` without paired NCA and LLM numbers from the same harness.
