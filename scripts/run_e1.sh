#!/usr/bin/env bash
# E1: public BFCL classification benchmark → stage0/stage0_results/e1_*.json
# Trains GraphNCA meta (3 seeds), saves alpha/adjacency, then runs E3 export.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export NCA_DATA_ROOT="${NCA_DATA_ROOT:-$ROOT/.nca_data}"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "$PYTHON" stage0/run_e1_benchmark.py --dry-run "$@"
fi

exec "$PYTHON" stage0/run_e1_benchmark.py \
  --seeds 0,1,2 \
  --steps 150000 \
  --topologies tree,meta,star,1d \
  --out stage0/stage0_results/e1_benchmark.json \
  "$@"
