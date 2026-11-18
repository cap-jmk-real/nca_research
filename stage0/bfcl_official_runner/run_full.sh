#!/usr/bin/env bash
# Full official BFCL v4 run on engine for one condition.
# usage: run_full.sh <baseline_no_gnca|gnca_gate>
set -uo pipefail
R=/home/s4il0r/nca_e_logs/official_bfcl_v4_20261111
COND="$1"
export BFCL_PROJECT_ROOT=$R NCA_REPO=/home/s4il0r/nca_bfcl_official_1111 NCA_DATA_ROOT=/home/s4il0r/nca_data
export GNCA_CHECKPOINT=/home/s4il0r/nca_data/stage0_results/checkpoints/best_graph_nca_grokking.pt
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama
export BFCL_ACTUATOR_OLLAMA=qwen3.5-9b-bfcl32k BFCL_REASONING_EFFORT=none
case "$COND" in
  baseline_no_gnca) MODEL=local/qwen3.5-9b-FC; export GNCA_ROUTER_ENABLED=0 ;;
  gnca_gate) MODEL=local/qwen3.5-9b-FC-gnca; export GNCA_ROUTER_ENABLED=1 GNCA_GATE_LOG=$R/logs/gnca_gate_decisions.jsonl ;;
  *) echo "unknown condition $COND"; exit 2 ;;
esac
# Official all_scoring minus web_search_base / web_search_no_snippet (no SERPAPI key on engine).
CATS="simple_python simple_java simple_javascript multiple parallel parallel_multiple irrelevance live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context memory_kv memory_vector memory_rec_sum"
mkdir -p $R/logs
L=$R/logs/${COND}.log
echo "[$(date '+%F %T %Z')] start $COND model=$MODEL" >> $L
for c in $CATS; do
  echo "[$(date '+%F %T %Z')] generate $c" >> $L
  $R/venv/bin/python $R/runner/bfcl_nca.py generate --model $MODEL --test-category $c \
      --num-threads 1 --skip-server-setup --result-dir result >> $R/logs/${COND}_generate.log 2>&1 \
      || echo "[$(date '+%F %T %Z')] generate $c FAILED rc=$?" >> $L
done
echo "[$(date '+%F %T %Z')] evaluate" >> $L
CSV=$(echo $CATS | tr ' ' ',')
$R/venv/bin/python $R/runner/bfcl_nca.py evaluate --model $MODEL --test-category $CSV \
    --result-dir result --score-dir score_$COND >> $R/logs/${COND}_evaluate.log 2>&1 \
  || $R/venv/bin/python $R/runner/bfcl_nca.py evaluate --model $MODEL --test-category $CSV \
    --result-dir result --score-dir score_$COND --partial-eval >> $R/logs/${COND}_evaluate.log 2>&1
echo "[$(date '+%F %T %Z')] done $COND rc=$?" >> $L
