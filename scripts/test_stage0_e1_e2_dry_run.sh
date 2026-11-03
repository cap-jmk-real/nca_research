#!/usr/bin/env bash
# CPU smoke: E1 dry-run must emit star checkpoints; E2 dry-run must finish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${NCA_DATA_ROOT:?Set NCA_DATA_ROOT to a writable directory before running this script}"
mkdir -p "$NCA_DATA_ROOT"
PYTHON="${PYTHON:-python3}"

"$ROOT/scripts/run_e1.sh" --dry-run
test -f stage0/stage0_results/e1_checkpoints/seed0_star.pt
test -f stage0/stage0_results/e1_checkpoints/seed0_meta.pt
test -f stage0/stage0_results/e1_checkpoints/seed0_1d.pt

"$ROOT/scripts/run_e2.sh" --dry-run
test -f stage0/stage0_results/e2_router_ablation.json

echo "stage0 E1→E2 dry-run chain OK"
