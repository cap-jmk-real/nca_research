#!/usr/bin/env bash
# Official web_search_base + web_search_no_snippet via SearXNG (no SerpAPI key).
# usage: run_web_search.sh <baseline_no_gnca|gnca_gate>
# Resume-safe: bfcl generate skips already-written ids in the same result dir.
# Starts after the 20-category run_full.sh jobs, so it does not share the GPU with them.
set -uo pipefail
R=/home/s4il0r/nca_e_logs/official_bfcl_v4_20261111
COND="$1"
export BFCL_PROJECT_ROOT=$R NCA_REPO=/home/s4il0r/nca_bfcl_official_1111 NCA_DATA_ROOT=/home/s4il0r/nca_data
export GNCA_CHECKPOINT=/home/s4il0r/nca_data/stage0_results/checkpoints/best_graph_nca_grokking.pt
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama
export BFCL_ACTUATOR_OLLAMA=qwen3.5-9b-bfcl32k BFCL_REASONING_EFFORT=none
export BFCL_PRESENCE_PENALTY="${BFCL_PRESENCE_PENALTY-0}" BFCL_OLLAMA_PARSE_FALLBACK="${BFCL_OLLAMA_PARSE_FALLBACK-1}"  # multi-turn fix 2026-11-12, same as rerun
export BFCL_SEARCH_BACKEND=searxng
export SEARXNG_URL="${SEARXNG_URL:-http://127.0.0.1:30479}"
export SEARXNG_ENGINES="${SEARXNG_ENGINES:-bing,mwmbl,wikipedia,wiby}"
case "$COND" in
  baseline_no_gnca) MODEL=local/qwen3.5-9b-FC; export GNCA_ROUTER_ENABLED=0 ;;
  gnca_gate) MODEL=local/qwen3.5-9b-FC-gnca; export GNCA_ROUTER_ENABLED=1 GNCA_GATE_LOG=$R/logs/gnca_gate_decisions.jsonl ;;
  *) echo "unknown condition $COND"; exit 2 ;;
esac
CATS="web_search_base web_search_no_snippet"
# Final official evaluate covers the earlier 20 categories plus these two (5106 scoring entries).
ALL_CATS="simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum,web_search_base,web_search_no_snippet"
mkdir -p $R/logs
L=$R/logs/${COND}_web_search.log
echo "[$(date '+%F %T %Z')] start web_search $COND model=$MODEL backend=searxng url=$SEARXNG_URL engines=$SEARXNG_ENGINES" >> $L
for c in $CATS; do
  echo "[$(date '+%F %T %Z')] generate $c" >> $L
  $R/venv/bin/python $R/runner/bfcl_nca.py generate --model $MODEL --test-category $c \
      --num-threads 1 --skip-server-setup --result-dir result >> $R/logs/${COND}_web_search_generate.log 2>&1 \
      || echo "[$(date '+%F %T %Z')] generate $c FAILED rc=$?" >> $L
done
echo "[$(date '+%F %T %Z')] evaluate all official scoring categories" >> $L
$R/venv/bin/python $R/runner/bfcl_nca.py evaluate --model $MODEL --test-category $ALL_CATS \
    --result-dir result --score-dir score_$COND >> $R/logs/${COND}_web_search_evaluate.log 2>&1 \
  || $R/venv/bin/python $R/runner/bfcl_nca.py evaluate --model $MODEL --test-category $ALL_CATS \
    --result-dir result --score-dir score_$COND --partial-eval >> $R/logs/${COND}_web_search_evaluate.log 2>&1
echo "[$(date '+%F %T %Z')] done web_search $COND rc=$?" >> $L
