# Agent and coding datasets

**Date:** 2026-10-03

Public multi-step trajectories for a later finetune whose labels are Hermes tools (`terminal`, `read_file`, `write_file`, `patch`, `execute_code`). This note is separate from the Hermes-harness slices in [curated-traces.md](curated-traces.md). Nothing here re-downloads `DJLougen/hermes-agent-traces-filtered` or writes `datasets/extra/`.

The gateway at `http://192.168.10.29:30487` answered. OpenAlex hits included AgentBank (`W4404783870`), AgentGym (`W4399455192`), and SWE-smith (`W4417525566`). AgentTrek and R2E-Gym were not in the top hits of those queries. Licenses, revisions, and counts below are from the Hub cards, GitHub files, and local parquet scans opened the same day.

Row counts for staged sets were counted from the parquet files on the SSD. Hub size-API figures are marked as such.

## Staged on the SSD

`scripts/stage_agent_coding_data.py` pins these revisions, writes under `/Volumes/Extreme SSD/nca_research/datasets/coding/`, and keeps the Hugging Face cache in `/Volumes/Extreme SSD/nca_research/hf-cache/`. The `tool` split of SWE-smith is the function-calling serialization. The `xml` and `ticks` splits are other renderings of the same collection and were left on the Hub.

`datasets/agent/` has no corpus. The agent sets opened below use web, embodied, or API actions.

| Dataset | Path | License | Revision | Measured |
| --- | --- | --- | --- | --- |
| [SWE-Gym/OpenHands-SFT-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | `datasets/coding/openhands-sft` | MIT, on the dataset card | `4aaa5a4a4b5861f4799d2336908760c190ac3b17` | 491 rows, 10,382,267 bytes, 1 parquet |
| [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) `tool` split | `datasets/coding/swe-smith-trajectories-tool` | MIT | `08e109b4a59eaeebf80e4675cd125d42e7ac99a4` | 24,100 rows, 1,077,275,999 bytes, 8 parquet files |
| [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | `datasets/coding/swe-rebench-openhands` | CC-BY-4.0 | `35455389ab51bf5e2306bfd436ef72d0f98bf882` | 67,074 rows, 2,079,557,168 bytes, 1 parquet |
| [R2E-Gym/R2EGym-SFT-Trajectories](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories) | `datasets/coding/r2e-gym-sft` | Apache-2.0, with the caveat below | `63ab4eb37668f8be0104133c21d896bedbcf8404` | 3,231 rows, 55,952,174 bytes, 1 parquet |

All four are multi-step. Each row is one agent trajectory (many tool calls), not one function-call JSON.

SWE-smith `tool`: 9,427 rows have `resolved=true` and 14,673 have `resolved=false`. The card says the publisher finetuned on 5,017 trajectories. That sentence is theirs. The file on disk has 24,100 rows.

Nebius: 32,161 rows have `resolved=1` and 34,913 have `resolved=0`. `exit_status` is `submit` on 60,739 rows. The rest stop on a max-iteration error (6,003), a stuck-loop error (252), or a smaller set of timeouts and API errors. These rollouts are on [nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench) issues. That card also has a `test` split of 21,336 task instances. Drop any overlapping `instance_id` before scoring SWE-rebench. This is not the SWE-bench test set.

R2E-Gym: the trajectory README has features and a train split and no `license` field. [R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) and [R2E-Gym-V1](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-V1) declare `apache-2.0` on their cards. The [GitHub LICENSE](https://github.com/R2E-Gym/R2E-Gym/blob/main/LICENSE) is Apache-2.0. The copy on disk is staged under that reading.

SWE-Gym's code repository LICENSE is Apache-2.0. The staged SFT card itself says MIT. The code repo is not the dataset license.

### Hermes tool mapping

Counts are tool calls inside the staged trajectories, measured by scanning assistant turns. `str_replace_editor` and `file_editor` are split by the `command` argument. A few calls use a command string outside the editor's enum (one SWE-smith call, one Nebius call). Those are included in the scan and omitted from the table.

| Source action | Calls | Hermes tool |
| --- | ---: | --- |
| `execute_bash` (OpenHands SFT) | 2,441 | `terminal` |
| `bash` (SWE-smith) | 107,050 | `terminal` |
| `execute_bash` (Nebius) | 2,197,279 | `terminal` |
| `execute_bash` (R2E-Gym) | 13,912 | `terminal` |
| `str_replace_editor` `view` (SFT / smith / Nebius) | 3,487 / 72,020 / 1,108,353 | `read_file` when `path` is a file. A directory `view` is a listing, same action name |
| `file_editor` `view` (R2E-Gym) | 14,852 | `read_file` |
| `str_replace_editor` `create` (SFT / smith / Nebius) | 622 / 26,148 / 373,890 | `write_file` |
| `file_editor` `create` (R2E-Gym) | 3,817 | `write_file` |
| `str_replace_editor` `str_replace` (SFT / smith / Nebius) | 1,604 / 57,762 / 338,895 | `patch` |
| `str_replace_editor` `insert` (SFT / smith / Nebius) | 105 / 165 / 15 | `patch` |
| `file_editor` `str_replace` (R2E-Gym) | 10,949 | `patch` |
| `search` (R2E-Gym) | 5,347 | `search_files` |
| `task_tracker` (Nebius) | 57,030 | `todo` |
| `submit` (SWE-smith) | 14,333 | `respond` |
| `finish` (SFT / Nebius / R2E-Gym) | 454 / 60,749 / 3,134 | `respond` |
| `think` (Nebius) | 180,365 | no Hermes tool |
| `undo_edit` | 64 SFT, 41 smith, 166 Nebius | no Hermes tool |

SFT calls are XML `<function=...>` blocks in `messages`. SWE-smith `tool` calls are `tool_calls[].function`. Nebius calls are `trajectory[].tool_calls`, with `arguments` stored as a JSON string. R2E-Gym calls are XML in `messages`, using `file_editor` rather than `str_replace_editor`. The Nebius tool list is also in `tools.json` next to the parquet: `execute_bash`, `str_replace_editor`, `think`, `finish`, `task_tracker`.

## Not staged

### Agent sets

| Dataset | What was opened | Why it stayed off the SSD |
| --- | --- | --- |
| [Solaris99/AgentBank](https://huggingface.co/datasets/Solaris99/AgentBank) | Apache-2.0 card. Hub size API: 53,205 train rows across the listed configs. `intercode_bash` row 0 is a 10-turn dialog whose actions are fenced bash blocks. | Multi-turn, and a 200-row bash config exists. The rest of the card is ALFRED, ALFWorld, web (Mind2Web, WebArena, WebShop), math, and code-contest dialogs. |
| [AgentGym/AgentTraj-L](https://huggingface.co/datasets/AgentGym/AgentTraj-L) | Hub size API: 14,485 train rows, about 108 MB of JSON. Card has no license field. The [AgentGym LICENSE](https://github.com/WooooDyy/AgentGym/blob/main/LICENSE) is MIT for the code. Files are ALFWorld, BabyAI, maze, Wordle, ScienceWorld, SQL, TextCraft, WebShop, and three API tools. | Train trajectories in those environments. [AgentGym/AgentEval](https://huggingface.co/datasets/AgentGym/AgentEval) is the matching test JSON. |
| [xlangai/AgentTrek](https://huggingface.co/datasets/xlangai/AgentTrek) | README: 52,594 web-browsing dialogue turns. No license field. Hub size API agrees on the row count. | Web navigation dialogs. |
| AgentOhana | [Paper page](https://huggingface.co/papers/2402.15506) lists zero linked datasets. The release point on the paper is [SalesforceAIResearch/xLAM](https://github.com/SalesforceAIResearch/xLAM). | No public trajectory dump on the Hub. `Salesforce/xlam-function-calling-60k` is gated single-shot function calling. |
| [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) | Apache-2.0. G2 and G3 are multi-tool solution paths. The README points at Google Drive and Tsinghua Cloud. | Multi-step RapidAPI calls. Hub mirrors exist (`Adorg/ToolBench` is 6.79 GB on the Hub page) and were not pinned. |
| [osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | CC-BY-4.0. Hub size API: 1,009 rows in `default/train`. The opened row has an `actions` list. Operation on that row: `SELECT`, then the action strings `CLICK` and `TYPE`. | A train split of web trajectories exists. Actions are browser operations. |
| [web-arena-x/webarena](https://github.com/web-arena-x/webarena) | README: evaluate 812 examples, then reset the sites. Human demonstration recordings cover about 170 tasks. | Eval environment. The human recordings are demonstrations on a subset of the tasks. |

### Coding sets that are not trajectories

| Dataset | What was opened | Why it stayed off the SSD |
| --- | --- | --- |
| [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | MIT. Hub size API: 2,438 train rows. Opened row fields are `problem_statement`, gold `patch`, `test_patch`, `FAIL_TO_PASS`. | Task instances. The trajectories are the OpenHands datasets above. |
| [SWE-bench/SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith) | MIT. Hub size API: 59,136 train rows. | Synthetic bug instances. Trajectories are the separate dataset, of which the `tool` split is staged. |
| [internlm/SWE-Fixer-Train-110K](https://huggingface.co/datasets/internlm/SWE-Fixer-Train-110K) | MIT. Hub size API: 115,406 train rows. Opened row is `problem_statement` plus a gold `patch`. | Single-shot issue-to-patch pairs. |
| [ByteDance-Seed/Multi-SWE-RL](https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-RL) | Card license `other`. README: unannotated RL instances, plus discarded Multi-SWE-bench ids. | Issue dumps, not agent traces. |
| [KodCode/KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1) | CC-BY-NC-4.0. Hub size API: 484,097 train rows. Opened row is `question`, `solution`, and `test`. | Coding problems with unit tests. No agent trace. Non-commercial license. |
| [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) | Hub size API: 6,055 rows in `train.raw`. Opened tools: `execute_bash`, `finish`, `str_replace_editor`. The card YAML has no `license` field. | Same action family as the staged SFT set. The 491-row SFT card is the MIT release. This raw card does not state a license, so it was not copied. |
| [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | CC-BY-4.0. Card: 80,036 trajectories, including the dev split of `princeton-nlp/SWE-bench`. The same card says Llama 3.1 output terms apply. Opened row is role/text from `swe-agent-llama-70b`, with a `generated_patch`. | Multi-step SWE-agent logs. The staged OpenHands sets already have shell and editor calls as function names. |
| [R2E-Gym/R2E-Gym-Lite](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Lite) | Hub size API: 4,578 `train` rows plus several `dev_*` splits. | Executable environments, not traces. The SFT trajectories are staged separately. |

## Eval-only

Keep these for scoring. They were not copied into the training directories.

| Set | What was opened |
| --- | --- |
| [SWE-bench/SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-bench) | The Hub page describes 2,294 issue-PR pairs and shows gold `patch` / `test_patch` fields. The splits API lists `test` and `dev` only. |
| [princeton-nlp/SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | Hub size API: `train` 19,008, `dev` 225, `test` 2,294. The opened train row is a gold patch and a problem statement, not a trajectory. `test` is the benchmark. |
| [SWE-bench/SWE-bench_Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | Splits API: `test` only. |
| [SWE-bench/SWE-bench_Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) | Splits API: `test` and `dev`. |
| WebArena | 812-task eval, above. |
| AgentGym/AgentEval | Test JSON paired with AgentTraj-L. |

## Failures

The first run of `scripts/stage_agent_coding_data.py` finished the Nebius download and then exited while scanning that parquet, with no traceback (`exit_code` unknown). A second run completed and wrote `datasets/coding/manifest.json`. The action counts in that manifest match a separate full scan of the same file (67,074 rows, 106 seconds).

Hub requests were unauthenticated. The downloads still finished. No SWE-bench test parquet was fetched.
