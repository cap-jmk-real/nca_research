# Specialist coder stack (Python → TypeScript → Shell → Orchestrator)

**Date:** 2026-10-05

Extend the Hermes / TextNCA program into a **plugged specialist** coding system. Distillation is allowed under the same **≤ 9B** teacher cap (`qwen3.5:9b` on engine). Students stay small enough for M1 + 3060. Results register as `EXP-*` under [paper/si/](../paper/si/README.md).

Config: [config/specialists.json](../config/specialists.json).

## Architecture

```text
                    ┌─────────────────────┐
  user / issue ───►│ Orchestrator (≤3B)  │── route label
                    └──────────┬──────────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌────────────┐      ┌────────────┐      ┌────────────┐
    │ Python     │      │ TypeScript │      │ Shell      │
    │ specialist │      │ specialist │      │ specialist │
    └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
          └───────────────────┼───────────────────┘
                              ▼
                    tools: terminal / read / write / patch
                              ▼
                    open harness verifiers (tests, tsc, Docker)
```

Optional: TextNCA supplies a **next-tool prior** (13 Hermes labels) to the orchestrator. It does **not** ingest teacher hidden states ([auto-research-loop.md](auto-research-loop.md) distillation rules).

## Build order

1. **Python specialist** — richest open trajectories + LiveCodeBench / SWE-Python.
2. **TypeScript specialist** — filter coding dumps for `.ts`/`.tsx`; `tsc` smoke.
3. **Shell specialist** — terminal-heavy Hermes + OpenHands bash; Terminal-Bench when sandboxed.
4. **Orchestrator** — learn `specialist_id` + high-level plan; plug the three students.

## Distillation (allowed)

| Allowed | Forbidden |
| --- | --- |
| SFT / LoRA students on teacher **completions** or **verified traces** (teacher ≤9B) | Distilling >9B teachers into students |
| KL on student vs 9B **logits** for specialist LMs (not into TextNCA) | Copying 9B weights into the NCA grid |
| Verifier-filtered FineEnvs / SWE resolved trajectories as labels | Unfiltered 27B completion floods |

Teacher endpoint: Ollama on engine `http://192.168.10.29:11434` model `qwen3.5:9b`.

## Datasets to compile

Sources already staged ([agent-coding-datasets.md](agent-coding-datasets.md)):

| Source | SSD path | Use |
| --- | --- | --- |
| OpenHands SFT | `datasets/coding/openhands-sft` | multi-step tools |
| SWE-smith trajectories `tool` | `datasets/coding/swe-smith-trajectories-tool` | python-heavy bugs |
| SWE-rebench OpenHands | `datasets/coding/swe-rebench-openhands` | large resolved/unresolved |
| R2E-Gym SFT | `datasets/coding/r2e-gym-sft` | editor + search |
| Hermes curated / extra | `datasets/curated`, `datasets/extra` | shell + mixed tools |

**Note (2026-10-05):** First compile from SWE-smith / OpenHands is **Python-heavy**
(thousands of rows). TypeScript hits are rare in those dumps — expect to add
TS-specific Hub sources (e.g. JS/TS SWE forks, internal `tsc` repair traces)
before claiming a TypeScript specialist. Shell comes from terminal-tagged Hermes
and OpenHands `execute_bash` in a later `--with-shell` pass.

Compiler: `scripts/compile_specialist_datasets.py` writes language-tagged jsonl under:

`/Volumes/Extreme SSD/nca_research/datasets/specialists/{python,typescript,shell,orchestrator}/`

Each row (minimum):

```json
{
  "specialist": "python",
  "session_id": "...",
  "lang_score": 0.0,
  "goal": "...",
  "messages_or_steps": [],
  "target_tool": "patch",
  "resolved": null,
  "source": "swe-smith-trajectories-tool",
  "path_hints": ["src/foo.py"]
}
```

Orchestrator rows add `"route_label": "python"|"typescript"|"shell"`.

## Evaluation harnesses (open source)

Priority order in config; score **after** a specialist checkpoint exists.

| Harness | Specialist | Notes |
| --- | --- | --- |
| [LiveCodeBench](https://livecodebench.github.io/) | python | Contests; follow their release tags |
| [SWE-bench Lite](https://www.swebench.com/lite.html) | python (+ later ts slices) | Docker; start with few `instance_id`s |
| [Terminal-Bench](https://www.tbench.ai/) / Harbor | shell | Needs sandbox |
| `tsc_repo_smoke` | typescript | Local: apply patch, `tsc --noEmit`, unit tests if present |
| Multi-specialist smoke | orchestrator | Synthetic tasks that require routing then one specialist |

Paired baselines: same harness with (a) `qwen3.5:9b` alone, (b) single student without orchestrator, (c) full plug. Register each as `EXP-*`.

## Training recipe (long enough)

Same floor as NCA ablations where applicable: **≥150k** optimizer steps or equivalent epoch budget on specialist SFT before claiming wins ([auto-research-loop.md](auto-research-loop.md) §Long training). Smoke compiles of datasets do not count as model results.

## Relation to TextNCA

| Component | Role |
| --- | --- |
| TextNCA | Cheap next-tool prior / research substrate |
| Specialists | Generative code/shell policies |
| Orchestrator | Routes language + owns multi-step plan |

Paper claims about coding performance cite specialist/orchestrator EXP records, not only Hermes top-1.

## Immediate next commands

```bash
# 1) Compile python then typescript specialist corpora from staged coding dumps
python scripts/compile_specialist_datasets.py --langs python typescript --max-rows 50000

# 2) When engine is up: teacher-filter / distill (≤9B)
# python scripts/distill_specialist.py --lang python --teacher qwen3.5:9b

# 3) Evaluate (install harness first)
# python scripts/eval_specialist_harness.py --lang python --harness LiveCodeBench --limit 20
```

Daemon / auto-research: after NCA 150k queue items, prefer specialist dataset compile + python distill when engine answers.
