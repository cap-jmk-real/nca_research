#!/usr/bin/env bash
# E2: router ablation (random / nca1d / none / learned) on BFCL holdout; requires E1 checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -z "${NCA_DATA_ROOT:-}" ]]; then
  echo "NCA_DATA_ROOT must be set (no default). Example: export NCA_DATA_ROOT=\"\$PWD/.nca_data\"" >&2
  exit 1
fi
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "$PYTHON" stage0/run_e2_router_ablation.py --dry-run "$@"
fi

exec "$PYTHON" stage0/run_e2_router_ablation.py \
  --seeds 0,1,2 \
  --routers random,nca1d,none,learned \
  --out stage0/stage0_results/e2_router_ablation.json \
  "$@"
