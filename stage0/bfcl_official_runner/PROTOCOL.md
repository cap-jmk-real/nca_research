# Official BFCL v4 end-to-end check: protocol as run

This folder holds the files that produced Section "End-to-end check on the official BFCL v4 harness" of the paper (`paper/generated/numbers_bfcl_official.tex`, `paper/generated/tab_bfcl_official.tex`). They are copies of the engine run directory `official_bfcl_v4_20261111` (started 2026-11-11, final official evaluate 2026-11-14) and of the router hook that was implemented in that worktree but never committed before. Paths inside the shell scripts are the engine paths of that run. Change them through the environment variables the scripts read.

## Two different BFCL evaluations in the paper

| Evaluation | Data | Task | Comparable to the public leaderboard |
|---|---|---|---|
| Routing split `bfcl_e1v2` (Sections 4 to 6) | public BFCL v3 single-turn AST, live and irrelevance files (`stage0/e1v2_prepare_splits.py`), 692 test turns | next Hermes tool label | no |
| Official check (this folder) | `bfcl-eval==2026.3.23` with its bundled BFCL v4 data, all 22 scored categories, 5,106 entries per condition | end-to-end function calling by an actuator LLM, scored by the unmodified official checker | per category and sub-score in protocol, not as Overall Acc (local Q4 model, no-think mode, Ollama tool parser, SearXNG web search) |

## Conditions

- `baseline_no_gnca` (`local/qwen3.5-9b-FC`): actuator alone.
- `gnca_gate` (`local/qwen3.5-9b-FC-gnca`): before every actuator query the GNCA routes the trace so far. If its top-1 label is `respond`, the tool schema is withheld for that query. Otherwise the actuator sees the full official tool list. This is a respond gate, not schema pruning: the 13-label Hermes vocabulary cannot select BFCL functions. Implementation: `stage0/bfcl_gnca_router_hook.py`, router service `gnca_router_server.py`.

Actuator: Ollama `qwen3.5:9b` (Q4_K_M) with `num_ctx 32768` (`Modelfile.qwen35_9b_bfcl32k`), native function calling, no thinking, temperature 0.001, presence penalty 0. Gate checkpoint: Hermes tree GNCA, K = 8, 200k steps, `best_graph_nca_grokking.pt`, sha256 `6a5dd813af7c1cb83b0b4c1d419ebfa87e9428040afb50e7221c184aae8f5acc` (not one of the tuned five-seed models, as the paper states). Web search: SearXNG backend through `searxng_patch.py` (engines bing, mwmbl, wikipedia, wiby), no SerpAPI.

Harness fix (reported in the paper): the first multi-turn pass inherited presence penalty 1.5 from the Ollama model file, which produced tool-call XML that Ollama's parser rejected with HTTP 500. Reruns use presence penalty 0 for both conditions (`run_multiturn_rerun.sh`, `run_nonlive_live_rerun.sh`) and turn a parser failure into an empty assistant reply, logged separately. The checker is unchanged. `replay_errdump.py` replays one captured failing request.

## Files

| File | Role |
|---|---|
| `bfcl_nca.py` | registers the two model names with the stock `bfcl` CLI (OpenAI-compatible handler pointed at Ollama), parse fallback, optional SearXNG hook |
| `gnca_router_server.py` | HTTP router service that loads the GNCA checkpoint in the paper environment |
| `run_full.sh`, `run_multiturn_rerun.sh`, `run_nonlive_live_rerun.sh`, `run_web_search.sh`, `await_then_web_search.sh` | generate and evaluate per condition |
| `requirements.txt`, `requirements.lock.txt` | harness environment (Python 3.11.18) |
| `../stage0_results/bfcl_official_1116/summary.json` | compact score summary used by `scripts/build_bfcl_official_1116.py` |

Pins: `bfcl-eval==2026.3.23`, Gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Per-entry generations and scores stay on the engine and in the encrypted backup. They are not in this repository.

The `OPENAI_API_KEY=ollama` value in the scripts is the dummy string the OpenAI client needs for a local Ollama server. It is not a credential.
