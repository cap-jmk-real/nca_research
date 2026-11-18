#!/bin/bash
# Engine launcher for stage0/run_hpo_1110.py (per-benchmark HPO, all models, equal trial budget).
# CPU worker (logreg/RF/HGB, tab + cell features) and GPU worker (MLP + 4 GNCA topologies) run in
# parallel; each is resumable from its per-trial JSONL logs. After each step a restic backup runs
# (~/bin/research-backup-now); a failed backup stops the job. Completion marker: $HPO_OUT/DONE
# (or $HPO_OUT/FAILED with the failing step).
set -uo pipefail
cd "$(dirname "$0")/.."
export HPO_OUT=${HPO_OUT:-$HOME/nca_e_logs/hpo_1110}
export NCA_DATA_ROOT=${NCA_DATA_ROOT:-$HOME/nca_data} CUBLAS_WORKSPACE_CONFIG=:4096:8
P=${PY:-$HOME/nca_venv/bin/python}; TRIALS=${TRIALS:-20}; NJ=${NJOBS:-12}
mkdir -p "$HPO_OUT"; rm -f "$HPO_OUT/FAILED"
fail() { echo "$(date -Is) FAILED $1" | tee "$HPO_OUT/FAILED"; exit 1; }
bk() { ~/bin/research-backup-now "$1" || fail "backup $1"; echo "$(date -Is) backup ok $1"; }
echo "== hpo_1110 start $(date -Is) trials=$TRIALS sha=$(cat COMMIT 2>/dev/null || echo n/a)"
bk hpo_1110_start
OMP_NUM_THREADS=$NJ $P stage0/run_hpo_1110.py --worker cpu --trials "$TRIALS" --n-jobs "$NJ" >"$HPO_OUT/cpu.log" 2>&1 &
CPU=$!
OMP_NUM_THREADS=4 $P stage0/run_hpo_1110.py --worker gpu --trials "$TRIALS" >"$HPO_OUT/gpu.log" 2>&1 &
GPU=$!
echo "cpu worker pid=$CPU gpu worker pid=$GPU"
wait $GPU; rg=$?; echo "$(date -Is) gpu worker exit $rg"; bk hpo_1110_gpu
wait $CPU; rc=$?; echo "$(date -Is) cpu worker exit $rc"; bk hpo_1110_cpu
[ $rg -eq 0 ] && [ $rc -eq 0 ] || fail "worker exit gpu=$rg cpu=$rc"
$P stage0/run_hpo_1110.py --collect --trials "$TRIALS" || fail collect
bk hpo_1110_collect
date -Is > "$HPO_OUT/DONE"; echo "== hpo_1110 done $(date -Is)"
