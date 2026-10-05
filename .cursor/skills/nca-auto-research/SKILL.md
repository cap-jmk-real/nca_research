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

## Each cycle (do in order)

1. **Measure** — Run `python scripts/auto_research.py` from repo root.  
   - Add `--klaut-search` when engine is reachable.  
   - Add `--klaut-report --topic "..."` only when starting a **new architecture family** (slow).

2. **Read output** — Open `results/auto_research/latest.json` on the SSD. Note `failing_gates`, `suggested_hypotheses`, `backlog_ranked`.

3. **Literature** — For the top failing gate, if Klaut was offline, run:  
   `python scripts/klaut_gateway.py search "<GAP_TO_QUERY string from auto_research.py>"`  
   Prefer papers with code; record DOI/arXiv in the hypothesis section you touch.

4. **Pick one hypothesis** — Choose the highest-priority id in `suggested_hypotheses` that is not yet measured. If empty, take priority 1 from [hypotheses-backlog.md](../../../docs/hypotheses-backlog.md).

5. **Implement minimally** — Match existing `nca_text/` style. No scope creep.

6. **Train/eval** — Curated or mix per hypothesis doc; log to SSD TensorBoard when training >1k steps.

7. **Score paired baselines** — Whenever benchmarks change, run **both**:  
   - `python scripts/run_text_benchmarks.py` (NCA checkpoint)  
   - `python scripts/score_local_next_tool.py` (Qwen2.5-1.5B)  
   Do not update `benchmark_gates.json` pass/fail without LLM comparison on the same task.

8. **Record** — Append under the hypothesis in `hypotheses-backlog.md`: date, metric delta, link to SSD artifact. Re-run `auto_research.py`.

9. **Commit** — One hypothesis per commit when possible; push branch `cursor/<name>-1754`.

## Distillation cap (9B)

- Allowed: label/trace imitation from **`qwen3.5:9b`** or smaller; observation text cleanup; verifier-filtered rows ([FineEnvs](../../../docs/multi-harness-and-message-passing.md)).
- Forbidden: distilling 9B **weights or hidden states** into `TextNCA`; unlimited 27B teacher SFT into respond-heavy mix.

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

## Do not

- Skip the flat baseline when claiming cellular binding helps.
- Treat Klaut synthesis narrative as measured fact — verify against retrieved paper records.
- Run full 150k training without a cheaper 8k curated ablation when the hypothesis is architectural.
