#!/usr/bin/env bash
# Multi-turn rerun (2026-11-12) after the decode/inference-error fix, one condition.
# usage: run_multiturn_rerun.sh <baseline_no_gnca|gnca_gate>
# Fix (same for both conditions, see STATUS.md "Multi-turn fix 2026-11-12"):
#   BFCL_PRESENCE_PENALTY=0       override Ollama qwen3.5 Modelfile default presence_penalty 1.5
#   BFCL_OLLAMA_PARSE_FALLBACK=1  Ollama tool-call XML parse 500 -> empty no-call reply (vLLM-like)
# Old multi-turn results: backup_multiturn_presencepen15_20261112/ (kept, not deleted).
# Non-live / live results are kept as generated on 2026-11-11.
# Afterwards: memory_* (never generated before), interim evaluate, then web_search (run_web_search.sh).
set -uo pipefail
R=/home/s4il0r/nca_e_logs/official_bfcl_v4_20261111
cd $R
COND="$1"
export BFCL_PROJECT_ROOT=$R NCA_REPO=/home/s4il0r/nca_bfcl_official_1111 NCA_DATA_ROOT=/home/s4il0r/nca_data
export GNCA_CHECKPOINT=/home/s4il0r/nca_data/stage0_results/checkpoints/best_graph_nca_grokking.pt
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama
export BFCL_ACTUATOR_OLLAMA=qwen3.5-9b-bfcl32k BFCL_REASONING_EFFORT=none
export BFCL_PRESENCE_PENALTY=0 BFCL_OLLAMA_PARSE_FALLBACK=1
case "$COND" in
  baseline_no_gnca) MODEL=local/qwen3.5-9b-FC; export GNCA_ROUTER_ENABLED=0 ;;
  gnca_gate) MODEL=local/qwen3.5-9b-FC-gnca; export GNCA_ROUTER_ENABLED=1 GNCA_GATE_LOG=$R/logs/gnca_gate_decisions_mtrerun_20261112.jsonl ;;
  *) echo "unknown condition $COND"; exit 2 ;;
esac
export BFCL_PARSE_FALLBACK_LOG=$R/logs/${COND}_parse_fallback_20261112.jsonl
CATS="multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context memory_kv memory_vector memory_rec_sum"
L=$R/logs/${COND}_mtrerun_20261112.log
echo "[$(date '+%F %T %Z')] start multi-turn rerun $COND model=$MODEL pp=$BFCL_PRESENCE_PENALTY fallback=$BFCL_OLLAMA_PARSE_FALLBACK" >> $L
for c in $CATS; do
  echo "[$(date '+%F %T %Z')] generate $c" >> $L
  $R/venv/bin/python $R/runner/bfcl_nca.py generate --model $MODEL --test-category $c \
      --num-threads 1 --skip-server-setup --result-dir result >> $R/logs/${COND}_mtrerun_20261112_generate.log 2>&1 \
      || echo "[$(date '+%F %T %Z')] generate $c FAILED rc=$?" >> $L
done
CSV="simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum"
echo "[$(date '+%F %T %Z')] interim evaluate (20 categories)" >> $L
$R/venv/bin/python $R/runner/bfcl_nca.py evaluate --model $MODEL --test-category $CSV \
    --result-dir result --score-dir score_$COND >> $R/logs/${COND}_mtrerun_20261112_evaluate.log 2>&1 \
  || $R/venv/bin/python $R/runner/bfcl_nca.py evaluate --model $MODEL --test-category $CSV \
    --result-dir result --score-dir score_$COND --partial-eval >> $R/logs/${COND}_mtrerun_20261112_evaluate.log 2>&1
echo "[$(date '+%F %T %Z')] multi-turn rerun done; backup + web_search next" >> $L
/home/s4il0r/bin/research-backup-now bfcl-mtrerun-$COND-done >> $L 2>&1 || echo "[$(date '+%F %T %Z')] backup FAILED" >> $L
# web_search follow-on (re-armed hook; replaces runner/await_then_web_search.sh, stopped 2026-11-12 06:53)
if [ -e $R/runner/NO_WEB_SEARCH ]; then echo "[$(date '+%F %T %Z')] NO_WEB_SEARCH present; skipping" >> $L; exit 0; fi
if [ "$COND" = gnca_gate ] && ! pgrep -f '[g]nca_router_server.py' >/dev/null; then
  echo "[$(date '+%F %T %Z')] router down; restarting" >> $L
  nohup setsid /home/s4il0r/nca_venv/bin/python runner/gnca_router_server.py >> $R/gnca_router_server.log 2>&1 &
  sleep 5
fi
echo "[$(date '+%F %T %Z')] exec run_web_search.sh $COND" >> $L
exec bash $R/runner/run_web_search.sh $COND >> $R/logs_web_search_${COND}.nohup 2>&1
