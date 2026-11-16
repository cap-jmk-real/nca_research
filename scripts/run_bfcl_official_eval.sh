#!/usr/bin/env bash
# Official BFCL v4 protocol on engine: GNCA as router + local actuator vs. baseline.
# GNCA does not produce leaderboard AST scores by itself; this script runs bfcl-eval
# on the actuator with optional schema pruning from the GNCA checkpoint.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$REPO_ROOT/config/bfcl_official_protocol.json"
BFCL_EVAL_PIN="$(python3 -c "import json; print(json.load(open('$CFG'))['bfcl_eval_version_pin'])")"

export BFCL_PROJECT_ROOT="${BFCL_PROJECT_ROOT:-$REPO_ROOT/results/bfcl_official}"
mkdir -p "$BFCL_PROJECT_ROOT"

# Actuator: Ollama on engine (override for ablations)
ACTUATOR_MODEL="${BFCL_ACTUATOR_MODEL:-frog-nano:4b}"
BFCL_MODEL_NAME="${BFCL_LEADERBOARD_MODEL_NAME:-local/FrogNano-4B-FC}"
TEST_CATEGORY="${BFCL_TEST_CATEGORY:-all_scoring}"
MODE="${1:---help}"

if ! command -v bfcl >/dev/null 2>&1; then
  echo "Installing bfcl-eval==${BFCL_EVAL_PIN}"
  pip install -q "bfcl-eval==${BFCL_EVAL_PIN}"
fi

python3 "$REPO_ROOT/scripts/bfcl_official_manifest.py" >"$BFCL_PROJECT_ROOT/manifest_latest.json"

case "$MODE" in
  --smoke)
    echo "[smoke] manifest + 5-ID subset (simple_python) — no LLM required for manifest"
  python3 - <<'PY'
import json
from pathlib import Path
import bfcl_eval
from pathlib import Path as P
root = P(bfcl_eval.__path__[0]) / "data" / "BFCL_v4_simple_python.json"
ids = []
for i, line in enumerate(root.read_text().splitlines()):
    if line.strip():
        row = json.loads(line)
        ids.append(row["id"])
        if len(ids) >= 5:
            break
out = {"simple_python": ids}
proj = P(__import__("os").environ["BFCL_PROJECT_ROOT"])
proj.mkdir(parents=True, exist_ok=True)
(proj / "test_case_ids_to_generate.json").write_text(json.dumps(out, indent=2) + "\n")
print("Wrote", proj / "test_case_ids_to_generate.json")
PY
    echo "On engine, run:"
    echo "  bfcl generate --model $BFCL_MODEL_NAME --run-ids --num-threads 1"
    echo "  bfcl evaluate --model $BFCL_MODEL_NAME --partial-eval"
    ;;
  --manifest-only)
    python3 "$REPO_ROOT/scripts/bfcl_official_manifest.py"
    ;;
  --full)
    echo "[full] Official all_scoring — requires GPU/Ollama on engine"
    echo "Step 1 (baseline, no GNCA):"
    echo "  export GNCA_ROUTER_ENABLED=0"
    echo "  bfcl generate --model $BFCL_MODEL_NAME --test-category $TEST_CATEGORY --num-threads 1"
    echo "  bfcl evaluate --model $BFCL_MODEL_NAME --test-category $TEST_CATEGORY"
    echo "Step 2 (GNCA schema prune — implement hook in stage0/bfcl_gnca_router_hook.py):"
    echo "  export GNCA_ROUTER_ENABLED=1"
    echo "  export GNCA_CHECKPOINT=/Volumes/Extreme SSD/nca_research/results/checkpoints/..."
    echo "  bfcl generate --model $BFCL_MODEL_NAME --test-category $TEST_CATEGORY --num-threads 1"
    echo "  bfcl evaluate --model $BFCL_MODEL_NAME --test-category $TEST_CATEGORY"
    echo "Results: \$BFCL_PROJECT_ROOT/result and score/data_overall.csv"
    ;;
  --help|*)
    cat <<EOF
Usage: $0 [--smoke|--manifest-only|--full]

Environment:
  BFCL_PROJECT_ROOT     Output dir (default: repo/results/bfcl_official)
  BFCL_ACTUATOR_MODEL   Ollama model tag on engine (default: frog-nano:4b)
  BFCL_LEADERBOARD_MODEL_NAME  bfcl model registry name (default: local/FrogNano-4B-FC)
  BFCL_TEST_CATEGORY      default: all_scoring

Pinned bfcl-eval: $BFCL_EVAL_PIN (see config/bfcl_official_protocol.json)
EOF
    ;;
esac
