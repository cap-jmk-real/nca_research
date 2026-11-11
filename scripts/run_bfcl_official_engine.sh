#!/usr/bin/env bash
# Run on Julian's engine (RTX 3060, Ollama). Full official BFCL v4 all_scoring protocol.
# GNCA is NOT the scored model; the Ollama actuator is. Compare baseline vs GNCA-pruned prompts.
set -euo pipefail

REPO_ROOT="${NCA_REPO:-$HOME/nca_research}"
# Default SSD layout from docs/data-root.md
export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-/Volumes/Extreme SSD/nca_research/results/bfcl_official}"
export NCA_DATA_ROOT="${NCA_DATA_ROOT:-/Volumes/Extreme SSD/nca_research/results}"

BFCL_EVAL_PIN="2026.3.23"
GORILLA_COMMIT="6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"

# Ollama on engine — no cloud API keys
export LOCAL_SERVER_ENDPOINT="${LOCAL_SERVER_ENDPOINT:-127.0.0.1}"
export LOCAL_SERVER_PORT="${LOCAL_SERVER_PORT:-11434}"
ACTUATOR_OLLAMA="${BFCL_ACTUATOR_OLLAMA:-qwen3.5:9b}"
# bfcl registry name for local OpenAI-compatible handler (edit constants/model_config.py copy if needed)
BFCL_MODEL_NAME="${BFCL_LEADERBOARD_MODEL_NAME:-local/qwen3.5-9b-FC}"
TEST_CATEGORY="${BFCL_TEST_CATEGORY:-all_scoring}"

mkdir -p "$BFCL_PROJECT_ROOT"
cd "$REPO_ROOT"

if ! command -v bfcl >/dev/null 2>&1; then
  pip install -q "bfcl-eval==${BFCL_EVAL_PIN}"
fi

python3 scripts/bfcl_official_manifest.py | tee "$BFCL_PROJECT_ROOT/manifest_$(date -u +%Y%m%dT%H%M%SZ).json"

# Copy example .env if missing (no secrets committed)
if [[ ! -f "$BFCL_PROJECT_ROOT/.env" ]]; then
  ENV_EX="$(python3 -c "import bfcl_eval, pathlib; print(pathlib.Path(bfcl_eval.__path__[0]) / '.env.example')")"
  cp "$ENV_EX" "$BFCL_PROJECT_ROOT/.env"
  echo "Edit $BFCL_PROJECT_ROOT/.env: set LOCAL_SERVER_* for Ollama; omit SERPAPI unless running web_search."
fi

RUN_TAG="${BFCL_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$BFCL_PROJECT_ROOT/logs/$RUN_TAG"
mkdir -p "$LOG_DIR"

run_condition() {
  local label="$1"
  local gnca_enabled="$2"
  export GNCA_ROUTER_ENABLED="$gnca_enabled"
  export GNCA_CHECKPOINT="${GNCA_CHECKPOINT:-$NCA_DATA_ROOT/checkpoints/text_nca_graph_tree_best.pt}"
  echo "=== Condition: $label (GNCA_ROUTER_ENABLED=$gnca_enabled) ===" | tee "$LOG_DIR/${label}.log"
  bfcl generate \
    --model "$BFCL_MODEL_NAME" \
    --test-category $TEST_CATEGORY \
    --num-threads 1 \
    --skip-server-setup \
    2>&1 | tee -a "$LOG_DIR/${label}_generate.log"
  bfcl evaluate \
    --model "$BFCL_MODEL_NAME" \
    --test-category $TEST_CATEGORY \
    2>&1 | tee -a "$LOG_DIR/${label}_evaluate.log"
}

# Baseline actuator, full tool schema
run_condition "baseline_no_gnca" "0"

# GNCA schema pruning hook (implement in stage0/bfcl_gnca_router_hook.py when wiring prompts)
run_condition "gnca_pruned" "1"

echo "Artifacts:"
echo "  manifest: $BFCL_PROJECT_ROOT/manifest_*.json"
echo "  generations: $BFCL_PROJECT_ROOT/result/$BFCL_MODEL_NAME/"
echo "  scores: $BFCL_PROJECT_ROOT/score/data_overall.csv"
echo "  logs: $LOG_DIR/"
echo "Pinned: bfcl-eval==$BFCL_EVAL_PIN, gorilla@$GORILLA_COMMIT"
