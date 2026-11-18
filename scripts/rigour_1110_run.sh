#!/bin/bash
# Engine launcher for stage0/run_rigour_1110.py + stage0/run_latency_1110.py (Nov 10 rigour pass).
# Phase base: multi-seed selection + 5-seed finals for the 10 models of hpo_1110 (CPU and GPU workers
# in parallel). Phase new: Transformer / untied GNN (GPU) and kNN (CPU) under the same protocol.
# Phase latency: single-thread end-to-end CPU latency on Hermes (run alone, after both workers).
# Every step ends with a restic backup (~/bin/research-backup-now); a failed or missing backup stops
# the job. Resumable: rerun the script, finished fits are skipped. Marker: $RIG_OUT/DONE or FAILED.
set -uo pipefail
cd "$(dirname "$0")/.."
export RIG_OUT=${RIG_OUT:-$HOME/nca_e_logs/rigour_1110} HPO_OUT=${HPO_OUT:-$HOME/nca_e_logs/hpo_1110}
export NCA_DATA_ROOT=${NCA_DATA_ROOT:-$HOME/nca_data} CUBLAS_WORKSPACE_CONFIG=:4096:8
P=${PY:-$HOME/nca_venv/bin/python}; NJ=${NJOBS:-8}
mkdir -p "$RIG_OUT"; rm -f "$RIG_OUT/FAILED"
fail() { echo "$(date -Is) FAILED $1" | tee "$RIG_OUT/FAILED"; exit 1; }
[ -x ~/bin/research-backup-now ] || fail "backup tool missing"
bk() { ~/bin/research-backup-now "$1" || fail "backup $1"; echo "$(date -Is) backup ok $1"; }
pair() {  # $1 = phase
  OMP_NUM_THREADS=$NJ $P stage0/run_rigour_1110.py --worker cpu --phase "$1" --n-jobs "$NJ" >>"$RIG_OUT/cpu_$1.log" 2>&1 &
  local C=$!
  OMP_NUM_THREADS=4 $P stage0/run_rigour_1110.py --worker gpu --phase "$1" >>"$RIG_OUT/gpu_$1.log" 2>&1 &
  local G=$!
  echo "$(date -Is) phase $1 cpu=$C gpu=$G"
  wait $G; local rg=$?; wait $C; local rc=$?
  echo "$(date -Is) phase $1 exit gpu=$rg cpu=$rc"
  [ $rg -eq 0 ] && [ $rc -eq 0 ] || fail "phase $1 gpu=$rg cpu=$rc"
}
echo "== rigour_1110 start $(date -Is) sha=$(cat COMMIT 2>/dev/null || echo n/a)"
bk rigour_1110_start
[ -f "$RIG_OUT/base.done" ] || { pair base; touch "$RIG_OUT/base.done"; bk rigour_1110_base; }
[ -f "$RIG_OUT/new.done" ] || { pair new; touch "$RIG_OUT/new.done"; bk rigour_1110_new; }
[ -f "$RIG_OUT/latency.json" ] || { $P stage0/run_latency_1110.py >"$RIG_OUT/latency.log" 2>&1 || fail latency; bk rigour_1110_latency; }
date -Is > "$RIG_OUT/DONE"; echo "== rigour_1110 done $(date -Is)"
