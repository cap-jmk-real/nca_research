# Curated next-tool traces

**Date:** 2026-10-03

Training rows for next-tool prediction, built from the public Hermes harness traces. The local `~/.hermes` pilot (35 sessions) is the test distribution and is not in these files. The script never opens that database. Extra rows from other train splits, and the sets that were not mixed in, are in [extra-traces.md](extra-traces.md).

## Source

[DJLougen/hermes-agent-traces-filtered](https://huggingface.co/datasets/DJLougen/hermes-agent-traces-filtered) loaded cleanly, so the full [lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) corpus was not downloaded or refiltered.

| | |
| --- | --- |
| Revision | `f91a8ef669194408a706741cd6f32a1686a8fd78` |
| File | `data/train.jsonl` |
| Bytes | 365,720,348 |
| SHA-256 | `030d5bab253492df280d96dfdd7a14b1ce6862b555f152a9595aea31b3e6c5b7` |
| License | Apache-2.0 |
| Rows parsed | 3,679 sessions, 0 JSON errors |

This slice is the quality-filtered Kimi subset of `lambda/hermes-agent-reasoning-traces` (3,679 of 7,646). GLM-5.1 trajectories are not included. No download failed.

## Row schema

One row is one kept decision in a session.

| Field | Meaning |
| --- | --- |
| `session_id` | Dataset trajectory id |
| `step_index` | 0-based index of kept decisions in that session |
| `goal` | Latest user message, cut at 500 characters |
| `recent_steps` | Up to the last 4 kept steps, each `{"tool", "result"}` |
| `target_tool` | Next kept tool name, or `respond` when the assistant turn has no tool call |

Tool results are cut at 500 characters. When one assistant turn calls several tools, later calls in that turn see the earlier ones in `recent_steps` with result `[pending]` until the following tool message fills them in. Measured longest stored goal is 497 characters; the result cap is binding (stored results reach 500).

## Filters

Kept targets: `terminal`, `read_file`, `write_file`, `patch`, `execute_code`, `search_files`, `web_search`, `todo`, `memory`, `skill_view`, `session_search`, `process`, `delegate_task`, and `respond`.

Every `browser_*` call is dropped from targets and from `recent_steps`. 6,268 browser calls were dropped: `browser_snapshot` 1,697, `browser_navigate` 1,519, `browser_scroll` 1,151, `browser_click` 892, `browser_vision` 412, `browser_close` 178, `browser_type` 143, `browser_press` 128, `browser_back` 66, `browser_console` 48, `browser_get_images` 34.

**Aliases:** none. The tool schema already uses the kept names (`process` is `process`, `todo` is `todo`). These schema tools were not mapped, because their descriptions are different tools:

- `clarify` (71 calls) asks the user a question.
- `skill_manage` (50) creates, updates, or deletes a skill.
- `skills_list` (54) lists skills. `skill_view` is kept separately (104 calls).

`web_search` is in the label set and has 0 calls. This slice's tool list does not include it. Strings such as `web_search` inside prompts or file text were not tool calls.

210 further calls were dropped as names outside the kept set. Most are the three skill/clarify tools above. The rest are one-off names that are not in the tool schema (`shell`, `grep`, `find`, `find_files`, and similar, one to three calls each). The full excluded histogram is in `data/curated/stats.json`.

41 sessions produced no kept decision and are in neither split. 3,638 sessions remain. No session id is in both splits. Local Hermes sessions included: 0.

## Split and counts

A session goes to validation when the first 8 bytes of `sha256(session_id)`, read as a big-endian integer, are divisible by 10. Every other session goes to train. The hash is on the session id, so a session cannot land in both files.

| | Sessions | Rows |
| --- | ---: | ---: |
| Train | 3,256 | 57,675 |
| Val | 382 | 6,995 |

Target counts:

| Tool | Train | Val |
| --- | ---: | ---: |
| `terminal` | 30,259 | 3,798 |
| `read_file` | 8,086 | 951 |
| `write_file` | 7,231 | 838 |
| `search_files` | 6,194 | 767 |
| `respond` | 2,854 | 331 |
| `patch` | 1,670 | 197 |
| `process` | 672 | 50 |
| `todo` | 254 | 21 |
| `execute_code` | 224 | 12 |
| `skill_view` | 88 | 16 |
| `session_search` | 82 | 6 |
| `delegate_task` | 33 | 5 |
| `memory` | 28 | 3 |
| `web_search` | 0 | 0 |

Train plus validation for each kept tool equals that tool's call count in the source file.

## Where the files are

Full jsonl is on the external data root from [data-root.md](data-root.md), not in git:

- `/Volumes/Extreme SSD/nca_research/datasets/curated/train.jsonl` (77,855,680 bytes)
- `/Volumes/Extreme SSD/nca_research/datasets/curated/val.jsonl` (9,407,308 bytes)
- `/Volumes/Extreme SSD/nca_research/results/curated-stats.json`

The repo holds `data/curated/stats.json` and `data/curated/sample.jsonl` (100 rows, 133,801 bytes, the first 100 training rows). `data/curated/*.jsonl` is gitignored except that sample. The Hugging Face cache is `HF_HOME` under the same data root.

## Rerun

From the repo root, with the Extreme SSD mounted:

```bash
python3 scripts/curate_traces.py
```

`NCA_DATA_ROOT` overrides the default `/Volumes/Extreme SSD/nca_research`. The script refuses a data root on the internal disk, under the home directory, or inside this repo. It sets `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, and `HF_DATASETS_CACHE` on that volume before downloading. Needs `huggingface_hub` only. `--inspect-only` writes a tool-schema report and does not emit train or val rows.
