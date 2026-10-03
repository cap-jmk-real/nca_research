# Agent trace datasets

**Date:** 2026-10-03

The training row in [hermes-first-run.md](hermes-first-run.md) is a Hermes decision: a user goal, recent tool steps, and the next tool name. A useful public dataset has that shape. A function-calling corpus of one-shot API picks does not.

## Use this

**[lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces)** (Apache 2.0). 14,701 trajectories, about 1.62 GB, two configs:

| Config | Teacher model | Trajectories | Tool calls |
| --- | --- | --- | --- |
| `kimi` | Kimi-K2.5 | 7,646 | 106,222 |
| `glm-5.1` | GLM-5.1-FP8 | 7,055 | 68,328 |

Each row is a ShareGPT conversation (`system`, `human`, `gpt`, `tool`) plus a JSON tool list, a task string, and a category. Tool calls ran for real inside the Hermes harness: terminal, files, browser, memory, delegation, todos, code execution. The publishers filtered out all-error traces, traces shorter than two turns, and hallucinated tool names. Repository tasks use permissively licensed code (Apache 2.0, MIT, BSD-3-Clause, Unlicense). Kimi outputs are under that model's Modified MIT terms, which allow training use.

This is the set to add after the 35 local sessions have a baseline. Map tool names onto the local vocabulary before mixing. The public browser tools are `browser_navigate`, `browser_click`, and `browser_snapshot`. The local log's browser mass is `browser_exec`. Treat those as different labels until a mapping exists.

A quality-filtered slice of the Kimi config is [DJLougen/hermes-agent-traces-filtered](https://huggingface.co/datasets/DJLougen/hermes-agent-traces-filtered): 3,679 of 7,646 rows, kept for longer reasoning, valid JSON tool calls, and visible self-correction. Use it if the full Kimi config teaches shallow "call first" behavior. It is a derivative, not a second source of tasks.

[ThreeSixNine/hermes-agent-reasoning-traces](https://huggingface.co/datasets/ThreeSixNine/hermes-agent-reasoning-traces) exposes the same card and the same `lambda/...` load path. Load `lambda/hermes-agent-reasoning-traces`.

## Do not use as the first corpus

These are real tool-use data. Their action names are not Hermes tools, so a readout trained on them does not transfer to the local agent.

| Dataset | Scale | What it actually is |
| --- | --- | --- |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | About 9k parsed rows across single-turn, multi-turn, and a 5k Glaive slice | Training mix for Hermes 2 Pro. ShareGPT turns that emit `<tool_call>` against arbitrary API schemas. One or a few calls, not a harness session. |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60,000 verified calls, 3,673 APIs | Single-shot function calling from APIGen (DeepSeek-V2 and Mixtral). Gated. No persistent agent loop. |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 11,300 dialogs, 26,507 APIs | Synthetic API dialogs, 2–12 turns. |
| [Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) | 160k synthetic plus 200k rows augmented from xLAM, Glaive, ToolACE, When2Call, BUTTONInstruct, APIGen-MT, and τ-bench | Large API-agent mix with turn-level filters. Still API tools, not `terminal` / `patch` / `read_file`. |
| AgentBank (Song et al., EMNLP 2024 Findings) | 51,287 trajectories, 16 tasks | Interaction traces across skill types, including embodied tasks. Different environments and action spaces. |
| [pagarsky/agent-trace](https://huggingface.co/datasets/pagarsky/agent-trace) | 1,400 traces | MBPP and NL2Bash rollouts from Qwen3-0.6B and Qwen3-1.7B, with timing and resource telemetry. Too small, and the tools are code and bash only. |
| AllenAI SERA coding trajectories, as cleaned in [AmanPriyanshu/tool-reasoning-sft-CODING-allenai-SERA-data-cleaned-rectified](https://huggingface.co/datasets/AmanPriyanshu/tool-reasoning-sft-CODING-allenai-SERA-data-cleaned-rectified) | 211,360 SWE-agent trajectories | Coding agents with `bash`, `str_replace_editor`, and `submit`. Right shape for a later coding head. Wrong labels for the Hermes first run. |

## How to mix with the local log

1. Keep the local 35 sessions as the test distribution, or hold out entire local sessions. The public set is Kimi and GLM playing tasks they were given, not this machine's history.
2. Train labels are the intersection of tool names: `terminal`, `read_file`, `write_file`, `search_files`, `patch`, `execute_code`, `todo`, `memory`, `delegate_task`, plus `respond` when the assistant turn has no tool call.
3. Leave browser actions out of the first joint training until `browser_exec` and the public browser trio are explicitly mapped.
4. One public trajectory becomes many rows, one per decision, with the same session-level split used for the local log.
