# Official BFCL v4 evaluation (GNCA + actuator)

This document separates two evaluations that both use public BFCL v4 **corpus** data but **different protocols**:

| Evaluation | Split / size | Task | Comparable to public leaderboard? |
| --- | --- | --- | --- |
| **Routing transfer (`bfcl_e1v2`)** | 692 test turns (session-stratified) | Next **Hermes tool label** on mapped BFCL prompts | **No** — not `bfcl evaluate` Overall Acc |
| **Official protocol (`bfcl-eval`)** | ~4.7k scoring entries in `bfcl-eval==2026.3.23` (see manifest) | End-to-end **function calling** on a named **actuator LLM** | **Yes**, when run with full `all_scoring` and no `--partial-eval` |

GNCA is a **System-1 router** (tool/schema pruning). It is not an BFCL actuator. Official numbers require:

1. Pinned `bfcl-eval` (`config/bfcl_official_protocol.json`).
2. A registered local actuator on **engine** (default: FrogNano-4B via Ollama; same family as exploratory pilots).
3. **Baseline:** actuator + full registry, no GNCA.
4. **GNCA condition:** router narrows the tool list / schema before the same actuator generates the FC response.

## Commands (engine — Julian)

Single entry point for the full `all_scoring` run (Ollama on `127.0.0.1:11434`, default actuator `qwen3.5:9b`):

```bash
cd ~/nca_research   # or your checkout path
export BFCL_PROJECT_ROOT="/Volumes/Extreme SSD/nca_research/results/bfcl_official"
export NCA_DATA_ROOT="/Volumes/Extreme SSD/nca_research/results"
pip install 'bfcl-eval==2026.3.23'
bash scripts/run_bfcl_official_engine.sh
```

**Pins:** `bfcl-eval==2026.3.23`, Gorilla git `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.

**Expected artifacts after a full run:**

| Path | Contents |
| --- | --- |
| `$BFCL_PROJECT_ROOT/manifest_*.json` | Official entry counts (`n_scoring_entries`: 4696 at this pin) |
| `$BFCL_PROJECT_ROOT/result/<MODEL>/` | Raw model generations per category |
| `$BFCL_PROJECT_ROOT/score/data_overall.csv` | Leaderboard-style Overall Acc (fill Table `tab:bfcl-official`) |
| `$BFCL_PROJECT_ROOT/logs/<RUN_TAG>/` | Per-condition `baseline_no_gnca` vs `gnca_pruned` logs |

**Cloud VM / CI:** manifest + smoke only:

```bash
python3 scripts/bfcl_official_manifest.py
bash scripts/run_bfcl_official_eval.sh --smoke
```

Do not run full `bfcl generate` on the cloud agent without GPU and Ollama checkpoints.

## Cloud VM blockers

This repository records manifests and scripts on the cloud agent VM. A full official run is **blocked** here without engine GPU, Ollama checkpoints, and `BFCL_PROJECT_ROOT` on the external SSD. Result paths after a full run:

- `$BFCL_PROJECT_ROOT/result/<MODEL>/`
- `$BFCL_PROJECT_ROOT/score/data_overall.csv`

Fill paper Table `tab:bfcl-official` from those CSVs; do not substitute routing top-1 from the 692 split.

## Pinning

- **Gorilla git:** `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- **PyPI:** `bfcl-eval==2026.3.23`
- **Manifest artifact:** `stage0/stage0_results/bfcl_official_manifest.json`
