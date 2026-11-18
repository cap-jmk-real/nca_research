#!/usr/bin/env bash
# Queued non_live/live rerun (Julian approved 2026-11-12 07:49) with the same fix as the multi-turn rerun,
# so the whole protocol uses presence_penalty 0 + Ollama parse fallback.
# usage: run_nonlive_live_rerun.sh <baseline_no_gnca|gnca_gate>
# Order per condition: wait until unit bfcl-mtrerun-<cond> (multi-turn, memory, evaluate, backup) is finished,
# then move old non_live/live result dirs aside (copies already in backup_nonlive_live_presencepen15_20261112/),
# regenerate the 13 categories, back up, then exec run_web_search.sh (web_search last; its final evaluate covers all 22).
# At most one generator per condition at any time (= 2 concurrent, as before).
set -uo pipefail
R=/home/s4il0r/nca_e_logs/official_bfcl_v4_20261111
cd $R
COND="$1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
L=$R/logs/${COND}_nonlive_live_rerun_20261112.log
echo "[$(date '+%F %T %Z')] queued; waiting for bfcl-mtrerun-$COND to finish" >> $L
while systemctl --user is-active --quiet bfcl-mtrerun-$COND.service \
   || pgrep -f "[r]un_multiturn_rerun.sh $COND" >/dev/null; do
  sleep 120
done
echo "[$(date '+%F %T %Z')] bfcl-mtrerun-$COND finished" >> $L
export BFCL_PROJECT_ROOT=$R NCA_REPO=/home/s4il0r/nca_bfcl_official_1111 NCA_DATA_ROOT=/home/s4il0r/nca_data
export GNCA_CHECKPOINT=/home/s4il0r/nca_data/stage0_results/checkpoints/best_graph_nca_grokking.pt
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=ollama
export BFCL_ACTUATOR_OLLAMA=qwen3.5-9b-bfcl32k BFCL_REASONING_EFFORT=none
export BFCL_PRESENCE_PENALTY=0 BFCL_OLLAMA_PARSE_FALLBACK=1
case "$COND" in
  baseline_no_gnca) MODEL=local/qwen3.5-9b-FC; MDIR=local_qwen3.5-9b-FC; export GNCA_ROUTER_ENABLED=0 ;;
  gnca_gate) MODEL=local/qwen3.5-9b-FC-gnca; MDIR=local_qwen3.5-9b-FC-gnca; export GNCA_ROUTER_ENABLED=1 GNCA_GATE_LOG=$R/logs/gnca_gate_decisions_nonlive_live_rerun_20261112.jsonl ;;
  *) echo "unknown condition $COND"; exit 2 ;;
esac
export BFCL_PARSE_FALLBACK_LOG=$R/logs/${COND}_parse_fallback_20261112.jsonl
B=$R/backup_nonlive_live_presencepen15_20261112/$MDIR
if [ ! -d "$B/non_live" ] || [ ! -d "$B/live" ]; then
  echo "[$(date '+%F %T %Z')] backup copy missing in $B; STOP" >> $L; exit 3
fi
mkdir -p $B/moved_from_result
for d in non_live live; do
  if [ -d result/$MDIR/$d ]; then mv result/$MDIR/$d $B/moved_from_result/$d; echo "[$(date '+%F %T %Z')] moved result/$MDIR/$d aside" >> $L; fi
done
if [ "$COND" = gnca_gate ] && ! pgrep -f '[g]nca_router_server.py' >/dev/null; then
  echo "[$(date '+%F %T %Z')] router down; restarting" >> $L
  nohup setsid /home/s4il0r/nca_venv/bin/python runner/gnca_router_server.py >> $R/gnca_router_server.log 2>&1 &
  sleep 5
fi
CATS="simple_python simple_java simple_javascript multiple parallel parallel_multiple irrelevance live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance"
echo "[$(date '+%F %T %Z')] start non_live/live rerun $COND model=$MODEL pp=$BFCL_PRESENCE_PENALTY fallback=$BFCL_OLLAMA_PARSE_FALLBACK" >> $L
for c in $CATS; do
  echo "[$(date '+%F %T %Z')] generate $c" >> $L
  $R/venv/bin/python $R/runner/bfcl_nca.py generate --model $MODEL --test-category $c \
      --num-threads 1 --skip-server-setup --result-dir result >> $R/logs/${COND}_nonlive_live_rerun_20261112_generate.log 2>&1 \
      || echo "[$(date '+%F %T %Z')] generate $c FAILED rc=$?" >> $L
done
echo "[$(date '+%F %T %Z')] non_live/live rerun done; backup" >> $L
if ! /home/s4il0r/bin/research-backup-now bfcl-nonlive-live-rerun-$COND-done >> $L 2>&1; then
  echo "[$(date '+%F %T %Z')] backup FAILED; not starting web_search" >> $L; exit 4
fi
echo "[$(date '+%F %T %Z')] exec run_web_search.sh $COND" >> $L
exec bash $R/runner/run_web_search.sh $COND >> $R/logs_web_search_${COND}.nohup 2>&1
