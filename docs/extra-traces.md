# Extra next-tool traces

**Date:** 2026-10-03

More training rows for the same next-tool task as [curated-traces.md](curated-traces.md). These files sit beside the existing curated split. The script does not open `~/.hermes`, does not write `datasets/curated/train.jsonl` or `val.jsonl`, and does not start a training run.

The row schema, truncation caps, browser drop, and session split are the ones in `scripts/curate_traces.py`.

## What was added

[lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces), revision `b92885e4f0161d4b2536512710e004d4892cac6e`, license Apache-2.0 (the dataset card's License section). Both configs are real Hermes-harness trajectories in ShareGPT form (`from` / `value`, `<tool_call>`, `<tool_response>`).

| Slice | File | Rows in file | SHA-256 | Bytes |
| --- | --- | ---: | --- | ---: |
| Kimi | `data/kimi/train.parquet` | 7,646 | `d4a53d84935d0bffe7054b12c0591c34fe31fc96115602394b3f517b4631dd02` | 508,262,558 |
| GLM-5.1 | `data/glm-5.1/train.parquet` | 7,055 | `d0344c77348cf092dce002256ed6ddbbd95bb568fbc90eb2cc8ffff3c9c3086d` | 599,571,360 |

The Kimi file's 7,646 ids are unique. All 3,679 ids in [DJLougen/hermes-agent-traces-filtered](https://huggingface.co/datasets/DJLougen/hermes-agent-traces-filtered) revision `f91a8ef669194408a706741cd6f32a1686a8fd78` (`data/train.jsonl`) are inside that Kimi file, and none are in the GLM file. Those 3,679 sessions are the already curated split, so this pass keeps the other 3,967 Kimi sessions and all 7,055 GLM sessions. Kept ids do not intersect. After curation, no extra session id appears in the curated train or val file (those files were only read; their mtimes stayed at 17:13).

No download of this corpus failed.

## Row schema and caps

Same fields as the curated split: `session_id`, `step_index`, `goal`, `recent_steps`, `target_tool`.

| Cap | Value |
| --- | --- |
| Goal | 500 characters |
| Tool result | 500 characters |
| Recent steps | 4 |
| Pending result | `[pending]` until the tool message fills it |

Measured on the combined files: train goals reach 500 characters and results reach 500; val goals reach 486 and results reach 500. No `recent_steps` list is longer than 4. No kept target or stored tool name starts with `browser_`.

A session is validation when the first 8 bytes of `sha256(session_id)`, read as a big-endian integer, are divisible by 10. Every other session is train. The hash is the raw dataset id.

Kept targets are `terminal`, `read_file`, `write_file`, `patch`, `execute_code`, `search_files`, `web_search`, `todo`, `memory`, `skill_view`, `session_search`, `process`, `delegate_task`, and `respond`. Aliases: none. `clarify`, `skill_manage`, and `skills_list` stay dropped, same as the curated writeup. `web_search` is in the label set and has 0 calls in both new slices.

## Counts

| Slice | Sessions kept | Sessions with a row | Train sessions | Val sessions | Train rows | Val rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi remainder | 3,967 | 3,902 | 3,504 | 398 | 32,833 | 3,629 |
| GLM-5.1 | 7,055 | 6,618 | 6,002 | 616 | 55,708 | 5,788 |
| Combined | 11,022 | 10,520 | 9,506 | 1,014 | 88,541 | 9,417 |

65 Kimi-remainder sessions and 437 GLM sessions had no kept decision, so they are in neither split. Local Hermes sessions included: 0. JSON parse errors: 0.

Browser calls dropped: 4,558 (Kimi remainder) and 6,287 (GLM). Unmapped calls dropped: 147 and 1,287. The GLM unmapped mass is `skills_list` 885, `clarify` 361, `skill_manage` 39, plus two malformed call strings (one each). The Kimi-remainder unmapped mass is `skill_manage` 55, `skills_list` 53, `clarify` 32, and a handful of one-off names (`shell` 3, and single calls of `execute`, `foreground`, `picture`, and a broken `read_file` string). Full histograms are in `data/extra/stats.json`.

Combined target counts:

| Tool | Train | Val |
| --- | ---: | ---: |
| `terminal` | 41,017 | 4,399 |
| `write_file` | 22,129 | 2,343 |
| `search_files` | 8,190 | 795 |
| `execute_code` | 4,377 | 492 |
| `todo` | 3,422 | 376 |
| `respond` | 3,291 | 364 |
| `read_file` | 2,338 | 236 |
| `session_search` | 1,034 | 154 |
| `skill_view` | 1,026 | 97 |
| `patch` | 805 | 79 |
| `delegate_task` | 453 | 50 |
| `memory` | 304 | 25 |
| `process` | 155 | 7 |
| `web_search` | 0 | 0 |

Per-slice histograms are in the stats file. GLM carries most of `execute_code`, `todo`, `skill_view`, `session_search`, and `delegate_task`. The Kimi remainder carries most of `write_file`, `read_file`, and `patch`.

## Where the files are

On the external data root from [data-root.md](data-root.md):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `/Volumes/Extreme SSD/nca_research/datasets/extra/lambda-kimi-remainder/train.jsonl` | 28,009,066 | `b4ea8b017a9d3d60127c2b87c11057ff6f8afbf9aeec23cc2646fb392e487ee7` |
| `/Volumes/Extreme SSD/nca_research/datasets/extra/lambda-kimi-remainder/val.jsonl` | 3,047,829 | `fb8a17a146141a8db24c43fa05babc9ce0bfa6b958de84507b112f5ef5a4af69` |
| `/Volumes/Extreme SSD/nca_research/datasets/extra/lambda-glm-5.1/train.jsonl` | 67,131,510 | `1a7d7a21e3e66ae3333c03df755573f9599b6af6904d8c92c7ca4eb7a0411995` |
| `/Volumes/Extreme SSD/nca_research/datasets/extra/lambda-glm-5.1/val.jsonl` | 7,186,196 | `cf24eebfd1794bccfc566c9c7d4a1f8ec1163ba0d516ad07f814109b62a56b1a` |
| `/Volumes/Extreme SSD/nca_research/datasets/extra/train.jsonl` | 95,140,576 | `7afa51e0e2a9db3db47f0298c54f04045cabfb400d049f70bca228acaa61db2c` |
| `/Volumes/Extreme SSD/nca_research/datasets/extra/val.jsonl` | 10,234,025 | `13011f84b60dc29b6078b84549bc3822af3d67d56a881d8c9aa149a276b904ac` |

The combined files are the two slices concatenated (Kimi remainder, then GLM). Session ids are unchanged, so the split hash matches the per-slice files.

Parquet originals are in the Hugging Face cache on the same volume (`HF_HOME=/Volumes/Extreme SSD/nca_research/datasets/hf_home`). Selected session jsonl, before the next-tool cut, is under `datasets/extra/raw/`. `pyarrow` for that read is installed at `datasets/extra/pydeps` (25.0.1). Copy of the stats: `/Volumes/Extreme SSD/nca_research/results/extra-stats.json`.

The repo holds `data/extra/stats.json` and `data/extra/sample.jsonl` (100 rows, 119,503 bytes: the first 100 Kimi-remainder training rows). `data/extra/*.jsonl` is gitignored except that sample.

## Rerun

From the repo root, with the Extreme SSD mounted:

```bash
python3 scripts/curate_extra_traces.py
```

`NCA_DATA_ROOT` overrides the default `/Volumes/Extreme SSD/nca_research`, with the same refusal rules as `scripts/curate_traces.py`. The script sets `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, and `HF_DATASETS_CACHE` on that volume before downloading. If `pyarrow` cannot be imported, it installs the wheel into `datasets/extra/pydeps` on that volume and restarts once.

## Excluded

These were opened far enough to decide. None of them were copied into the extra train or val files.

**Duplicates of lambda.** [sroecker/hermes-agent-traces-chatml](https://huggingface.co/datasets/sroecker/hermes-agent-traces-chatml) revision `9f872dadacac8e6820ad54af5a46c42a3808f70c`, Apache-2.0, README says 18,487 rows labeled `hermes-traces` or `nous-fc`. The Hermes half repeats this corpus, and the Nous half is the function-calling mix called out below. [OpenTraces/lambda-hermes-agent-reasoning-opentraces](https://huggingface.co/datasets/OpenTraces/lambda-hermes-agent-reasoning-opentraces) revision `617a8e0c6942b392c017e5d7f4e05af83ec6b515`, CC-BY-4.0. Its README stats are 7,645 traces from `moonshotai/kimi-k2.5`. Hugging Face search also lists many repos whose names match the lambda corpus (AdversaLLC, ThreeSixNine, insym, and others). Those file trees were not opened; the canonical files above are the ones that were read.

**Too small, and not tool traces.** [cfahlgren1/hermes-agent-traces](https://huggingface.co/datasets/cfahlgren1/hermes-agent-traces) revision `eb8f64e44276fa097ac090b98afa7ab0f3b124b7`. `sessions.jsonl` is 43,048 bytes and 5 sessions. The opened lines are one-turn smoke prompts ("Reply with exactly one short sentence...") with no tool call. The dataset card has no license. [cfahlgren1/hermes-agent-trace-samples-2026-06-05](https://huggingface.co/datasets/cfahlgren1/hermes-agent-trace-samples-2026-06-05) revision `80c32a72cc94906492d67f34353f477f9e38b182`, Apache-2.0, five public test exports. The opened session `20260605_092232_6d39db` has `tool_call_count` 0.

**Unlicensed derivative of sessions already curated.** [axolotl-ai-co/hermes-agent-traces-augmented](https://huggingface.co/datasets/axolotl-ai-co/hermes-agent-traces-augmented) revision `a22187d78e5601cbf955c668129df8b2ace848d0`. The card has no license. The one row opened through the dataset viewer has id `0c699abf-bc77-454a-8197-d56a2294098a`, `augmented: true`, and that id is in the filtered Kimi slice. The parquet (108,167,701 bytes) was not downloaded.

**One-shot API schemas, left out as primary finetune data.** NousResearch/hermes-function-calling-v1, Salesforce/xlam-function-calling-60k, Team-ACE/ToolACE, and Nanbeige/ToolMind. Descriptions of why their action names are API schemas are in [agent-trace-datasets.md](agent-trace-datasets.md). They were not downloaded in this pass.

**BFCL.** Official [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`, Apache-2.0. The file list is `BFCL_v3_*.json` eval categories (non-live simple/multiple/parallel, live, and multi-turn). There is no train file. Those JSON files were not downloaded. [hjshah/bfcl_non_multi_turn](https://huggingface.co/datasets/hjshah/bfcl_non_multi_turn) revision `31fdf665c1c7b0dcbb208f0d69c627a24f2a199c` names its only split `train` (card: 3,641 rows). Two opened rows are ids `irrelevance_0` and `irrelevance_1`, `multi_turn: false`, with functions such as `determine_body_mass_index`. That is the official eval set under a train label, so the parquet was not downloaded. Split rule: only a split that the publisher holds out from the leaderboard counts as train. A file renamed to `train` that still uses eval ids does not.

**NESTFUL.** [ibm-research/nestful](https://huggingface.co/datasets/ibm-research/nestful) revision `f1ed64619bc1babce637267673937d39c269fa07`, Apache-2.0. The card calls it a benchmark and the only split is named `train` with 1,861 examples, which is the whole published set. The opened row's tools are `subtract`, `rectangle_perimeter`, `square_edge_by_perimeter`, `add`, `volume_cylinder`, `inverse`, `log`, and `multiply`. Those names do not match the Hermes set. The parquet was not saved into `datasets/extra`. Split rule: this `train` file is the evaluation benchmark, so it stays out.

**tau2-bench.** [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) publishes per-domain `split_tasks.json` files. The ones opened are airline (`train` 30, `test` 20, `base` 50), retail (`train` 74, `test` 40, `base` 114), and telecom (`train` 74, `test` 40, `small` 20, `base` 114, `full` 2,285). Airline `tools.py` methods that were opened are domain APIs: `book_reservation`, `cancel_reservation`, `search_direct_flight`, `get_user_details`, `transfer_to_human_agents`, and the rest of that class. `tasks.json` holds every id, including test, and was not downloaded. [HuggingFaceH4/tau2-bench-data](https://huggingface.co/datasets/HuggingFaceH4/tau2-bench-data) is MIT domain data "for agentic evaluation," not trajectories. [AgentSuite/tau2-bench-trajectories](https://huggingface.co/datasets/AgentSuite/tau2-bench-trajectories) revision `07264f1d253a618fc20966670eddc44246212916` is 30 model files, 278 tasks each, with an `eval_result` field. 278 is the sum of the airline, retail, and telecom `base` lists (train ids and test ids together). Those jsonl files were not downloaded. Split rule: a trajectory would have to be limited to the `train` ids in `split_tasks.json`, and its tool names would have to be the Hermes names. The published rollouts are full `base` eval dumps, and the opened airline tools are booking APIs.

**AppWorld.** [stonybrooknlp/appworld](https://github.com/stonybrooknlp/appworld) (Apache-2.0) says it releases everything for `train` and `dev`, and only evaluation programs for `test_normal` and `test_challenge`. Task archives were not downloaded. The opened train row of [hamishivi/appworld_env_train](https://huggingface.co/datasets/hamishivi/appworld_env_train) has tools `['appworld']` and a contact-creation task. That single tool does not map onto the Hermes names.

A Klaut paper search at `http://192.168.10.29:30487/v1/research/papers/search` returned HTTP 200. The abstracts that came back (xLAM, BUTTON, ToolMind, and an agent-monitoring paper) did not point at another Hermes-harness dump. Datasets from those abstracts were not opened in this pass, apart from the ToolMind exclusion already listed.
