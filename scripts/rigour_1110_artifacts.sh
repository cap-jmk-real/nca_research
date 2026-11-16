#!/bin/bash
# Resume-only: missing Hermes artifacts (CPU and GPU in parallel) -> latency -> backup. Marker: $RIG_OUT/ART_DONE.
set -uo pipefail
cd "$(dirname "$0")/.."
export RIG_OUT=${RIG_OUT:-$HOME/nca_e_logs/rigour_1110} HPO_OUT=${HPO_OUT:-$HOME/nca_e_logs/hpo_1110}
export NCA_DATA_ROOT=${NCA_DATA_ROOT:-$HOME/nca_data} CUBLAS_WORKSPACE_CONFIG=:4096:8
P=${PY:-$HOME/nca_venv/bin/python}
fail() { echo "$(date -Is) FAILED $1" | tee "$RIG_OUT/ART_FAILED"; exit 1; }
[ -x ~/bin/research-backup-now ] || fail "backup tool missing"
bk() { ~/bin/research-backup-now "$1" || fail "backup $1"; echo "$(date -Is) backup ok $1"; }
rm -f "$RIG_OUT/ART_FAILED"
OMP_NUM_THREADS=8 $P stage0/run_artifacts_1110.py --worker cpu >>"$RIG_OUT/artifacts_cpu.log" 2>&1 & C=$!
OMP_NUM_THREADS=4 $P stage0/run_artifacts_1110.py --worker gpu >>"$RIG_OUT/artifacts_gpu.log" 2>&1 & G=$!
wait $G; rg=$?; wait $C; rc=$?
[ $rg -eq 0 ] && [ $rc -eq 0 ] || fail "artifacts gpu=$rg cpu=$rc"
bk rigour_1110_artifacts
[ -f "$RIG_OUT/latency.json" ] && mv "$RIG_OUT/latency.json" "$RIG_OUT/latency_empty_1413.json"
$P stage0/run_latency_1110.py >"$RIG_OUT/latency.log" 2>&1 || fail latency
bk rigour_1110_latency2
date -Is > "$RIG_OUT/ART_DONE"
