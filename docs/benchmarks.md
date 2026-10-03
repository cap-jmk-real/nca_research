# Benchmarks that can run beside the text NCA

**Date:** 2026-10-03

This note lists scored benchmarks. Public trace datasets are in [agent-trace-datasets.md](agent-trace-datasets.md). The Hermes run in [hermes-first-run.md](hermes-first-run.md) predicts the next tool name, or a single `respond` class, on 35 local sessions. The suite below starts after that comparison has a number. It still uses a text policy. It does not use the vision channels in [research-design.md](research-design.md).

A benchmark is listed only when an official page states a scorer and a held-out set. Sizes and licenses are taken from those pages. Preprints are marked.

## How to run them together

The shared machine is engine: one RTX 3060, with `qwen3.5:9b` already served by Ollama. The NCA readout is the policy under test. `qwen3.5:9b` is the local generator when a harness cannot call the NCA, and the user simulator when a benchmark needs one.

Run one generator at a time. CPU scorers (AST match, local Python, database unit tests) can overlap that generator. A desktop VM, a browser farm, or a full SWE-bench image pull does not fit in the same pass.

The tool-name head from the 35 sessions is a different action from the ones these benchmarks score. Each entry below needs the text grid to emit that benchmark's action: a function call with arguments, a nested call sequence, a short program, or a patch.

## Recommended parallel suite

Score these four on the same text policy, with no screenshots. Start each on a small slice, then report the held-out split.

1. **BFCL, non-live AST and irrelevance.** Function-call accuracy, no browser.
2. **NESTFUL `data_v2`.** Nested API sequences, executed locally.
3. **τ2-bench, text, `airline` and `retail`, one trial.** Multi-turn tools and a user, scored on the database end state.
4. **AppWorld `dev`, then `test_normal`.** Local apps, short programs over APIs, database unit tests. `test_normal` is the held-out number.

**SWE-bench Lite** is the fifth, and only as a handful of instance ids, once the action is a unified diff. The full Lite set of 300 is a repo-sandbox job.

## Each benchmark

### Berkeley Function Calling Leaderboard (BFCL)

- **Scores:** whether the emitted function call matches the expected call. Non-live categories use AST matching; other categories execute the call. The leaderboard also scores irrelevance (no call when no function fits), multi-turn, and agentic settings. [Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), [v1 writeup](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html), [code](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).
- **Fits** once the readout emits a function name and arguments. `simple_python`, parallel, multiple, and irrelevance need no browser and no paid API. The README documents an OpenAI-compatible endpoint, so Ollama can generate.
- **Beside the others:** yes, on those categories. The `web_search` category needs a SerpAPI key, so leave it out. Agentic slices are the same harness later, not extra machines.
- **License:** the README states that the leaderboard statistics and the data used to train the models are Apache 2.0.
- **Held-out:** `bfcl evaluate` scores a generated file against the fixed eval set. `--partial-eval` scores a subset and will not match the leaderboard.

### NESTFUL

- **Scores:** nested API sequences, where one call's output feeds the next call. The paper's metrics are function and parameter F1, partial sequence accuracy, full sequence accuracy, and win rate (executed calls match the gold answer). Preprint, [arXiv:2409.03797](https://arxiv.org/abs/2409.03797). Code: [github.com/IBM/NESTFUL](https://github.com/IBM/NESTFUL). The README's `data_v2/nestful_data.jsonl` has 1,861 rows, with Python under `data_v2/executable_functions`.
- **Fits** once the readout can emit a short sequence of calls. No browser. Use `data_v2`. The older `data_v1` executable split goes through RapidAPI.
- **Beside the others:** yes, if generation stays on the local model. The published `run.sh` loads the model with vLLM. Generate with Ollama and score with `src/scorer.py`.
- **License:** Apache 2.0, in the repo `LICENSE`. The paper states the same.
- **Held-out:** that JSONL file is the evaluation set. The paper builds it from MathQA's test split and a filtered coding split.

### τ2-bench

- **Scores:** a customer-service episode. The agent has tools and a written policy; a second model plays the user. The default reward multiplies a database end-state check by a check that required information was said. The original τ-bench metric is pass^k over repeated trials. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) covers retail and airline. τ2 adds telecom, where the user also has tools: preprint [arXiv:2506.07982](https://arxiv.org/abs/2506.07982). Code: [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench). Site: [taubench.com](https://www.taubench.com). Text domains in the README: `mock`, `airline`, `retail`, `telecom`. `banking_knowledge` is a separate install.
- **Fits** as a multi-turn tool loop that also has a reply turn. It does not fit a single tool-name label.
- **Beside the others:** yes for `airline` and `retail`, text mode, one trial, a small `--num-tasks` first. The README uses LiteLLM and says any supported provider works; the documented command uses `gpt-4.1`. Point both roles at one Ollama model before paying for an API. Skip voice mode: it uses a realtime provider, and the getting-started note requires ElevenLabs voices. Leave `telecom` until airline and retail have a number.
- **License:** MIT.
- **Held-out:** the CLI selects task splits. Report a split that was not used to tune the prompt.

### AppWorld

- **Scores:** database-state unit tests, including unintended changes. The report fields are `tgc` and `sgc`; `sgc` is defined over a group of tasks. [appworld.dev](https://appworld.dev/) describes 9 apps, 457 APIs, 106 simulated people, and 750 tasks. Code: [github.com/StonyBrookNLP/appworld](https://github.com/StonyBrookNLP/appworld). ACL 2024. Splits: `train`, `dev`, `test_normal`, `test_challenge`.
- **Fits** once the action is a short program that calls APIs, the Hermes `execute_code` shape.
- **Beside the others:** yes on `dev`. The engine is local. No browser and no paid API. Report `test_normal` as the held-out number. Their rule is no per-task error analysis on the test splits.
- **License:** the public portion of the repo is Apache 2.0. Task, app, and API material ships in encrypted bundles under Apache 2.0 plus a requirement that public redistribution stay encrypted.
- **Held-out:** `test_normal` and `test_challenge`. `train` and `dev` include solutions.

### SWE-bench Lite

- **Scores:** whether a patch resolves a GitHub issue, by applying it in Docker and running the repo's tests. [swebench.com](https://www.swebench.com/), [Lite](https://www.swebench.com/lite.html), [evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/). The FAQ lists 2,294 full test instances, 300 in Lite, and 500 in Verified. Lite's selection removed instances that contain images. The repo is MIT. The SWE-bench paper is listed there as ICLR 2024.
- **Fits** once the action is a patch, the Hermes `patch` / `write_file` shape.
- **Beside the others:** a few instance ids can be scored in the same pass. The full Lite set, Verified, and the full set are container jobs. Docker is required. No paid model API if the patch comes from the local policy.
- **Held-out:** the `test` split. Lite also publishes 23 development instances.

## Later text benchmarks

These have scorers and do not need the vision channels. They are too heavy, or too far from the tool loop, for the concurrent pass.

- **Terminal-Bench.** A task counts as resolved when its verifier passes. Current release: [tbench.ai](https://www.tbench.ai/). The [Harbor Hub task list](https://www.tbench.ai/tasks), fetched the same day, showed 66 tasks and says the suite includes GPU and multi-container tasks, with Modal or Daytona as the recommended sandbox. Code: [harbor-framework/terminal-bench](https://github.com/harbor-framework/terminal-bench), Apache 2.0. This is the benchmark that matches the Hermes `terminal` mass. It waits on a sandbox. The older [laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench) README describes a beta of about 100 tasks, a local Docker CLI, and dataset `terminal-bench-core` v0.1.1. That is a separate release.
- **WebArena.** Functional correctness on 812 tasks in self-hosted sites (shopping, a forum, GitLab, a CMS, plus utilities). [webarena.dev](https://webarena.dev/), [github.com/web-arena-x/webarena](https://github.com/web-arena-x/webarena), Apache 2.0, ICLR 2024. Needs a browser and several Docker sites. That is the `browser_exec` shape, and it is its own setup. The README says to reset the sites after a full run, and not to score against the demo sites.
- **GAIA.** Quasi-exact match on a short answer. [Dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA), [leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard), [arXiv:2311.12983](https://arxiv.org/abs/2311.12983). The leaderboard text says more than 450 questions, three levels, a public dev set, and a test set whose answers are private. Some questions attach a file, and the same text says the tasks need tooling and search. The test set cannot be scored locally.
- **LiveCodeBench.** Programs scored by tests. [livecodebench.github.io](https://livecodebench.github.io/), [repo](https://github.com/LiveCodeBench/LiveCodeBench), MIT on the evaluation code. The problems are from LeetCode, AtCoder, and Codeforces, so that MIT license does not cover the problem text. This checks a short program, not a tool choice.
- **AppWorld-UL.** Preprint, [arXiv:2607.20536](https://arxiv.org/abs/2607.20536), returned by the gateway. It adds clarification, confirmation, and infeasible requests on top of AppWorld. Run it after AppWorld has a number.

## Skip until vision channels exist

- **OSWorld.** Execution-based computer tasks. The project page states 369 tasks, or 361 if eight Google Drive tasks are dropped, and it publishes screenshot, accessibility-tree, and combined settings. [os-world.github.io](https://os-world.github.io/), [arXiv:2404.07972](https://arxiv.org/abs/2404.07972). The accessibility tree is text, and the environment is still a desktop VM. Screenshot runs wait for the vision channels. **OSWorld 2.0**, preprint [arXiv:2606.29537](https://arxiv.org/abs/2606.29537), is the same wait: the gateway abstract describes 108 long-horizon computer-use workflows.
- **VisualWebArena.** Visually grounded web tasks on the WebArena stack, observed as screenshots. [arXiv:2401.13649](https://arxiv.org/abs/2401.13649), [github.com/web-arena-x/visualwebarena](https://github.com/web-arena-x/visualwebarena).
- **SWE-bench Multimodal.** Issue resolution where the issue contains images. [swebench.com/multimodal](https://www.swebench.com/multimodal). That page says the original release had 517 issues with visual elements, and that multimodal v2 keeps 480 for reproducible evaluation. Same MIT repo as SWE-bench. The SWE-bench README lists the paper as ICLR 2025 ([arXiv:2410.03859](https://arxiv.org/abs/2410.03859)). Use text-only Lite first.

## A score is not ground truth

A gateway search on 2026-10-03 returned the preprint [Benchmarking the Benchmarks: A Validity Audit of Tool-Calling Evaluation](https://arxiv.org/abs/2607.02577). The abstract audits BFCL v4, τ2-bench, LiveMCPBench, and MCP-Atlas and reports evaluator-human disagreements, including a wide range across repeated LiveMCPBench runs. Keep the raw traces from the suite above. If the same policy gets two different numbers, the scorer is part of the result.

The gateway at `http://192.168.10.29:30487` answered. Searches returned NESTFUL, GAIA, OSWorld, AppWorld, AppWorld-UL, and OSWorld 2.0. The BFCL, τ-bench, SWE-bench, and WebArena papers were not in the top hits. Those four are cited from the official pages fetched the same day.
