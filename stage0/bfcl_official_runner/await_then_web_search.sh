#!/usr/bin/env bash
# Wait for the in-flight 20-category official run, then start the SearXNG web-search pass.
set -uo pipefail
R=/home/s4il0r/nca_e_logs/official_bfcl_v4_20261111
cd "$R"
L=$R/logs/await_web_search.log
mkdir -p $R/logs
echo "[$(date '+%F %T %Z')] waiter start; will launch web_search when run_full.sh drivers exit" >> $L
while pgrep -f '[b]ash runner/run_full.sh baseline_no_gnca' >/dev/null \
   || pgrep -f '[b]ash runner/run_full.sh gnca_gate' >/dev/null; do
  sleep 60
done
echo "[$(date '+%F %T %Z')] run_full.sh drivers gone" >> $L
# Keep / start the GNCA router for the gnca_gate condition.
if ! pgrep -f '[g]nca_router_server.py' >/dev/null; then
  echo "[$(date '+%F %T %Z')] router down; restarting" >> $L
  NCA_DATA_ROOT=/home/s4il0r/nca_data \
  GNCA_CHECKPOINT=/home/s4il0r/nca_data/stage0_results/checkpoints/best_graph_nca_grokking.pt \
    nohup setsid /home/s4il0r/nca_venv/bin/python runner/gnca_router_server.py \
      >> $R/gnca_router_server.log 2>&1 &
  sleep 2
fi
# Do not start a second copy if a previous waiter already launched these.
if pgrep -f '[b]ash runner/run_web_search.sh' >/dev/null; then
  echo "[$(date '+%F %T %Z')] run_web_search.sh already running; not starting another" >> $L
  exit 0
fi
nohup setsid bash runner/run_web_search.sh baseline_no_gnca >> $R/logs_web_search_baseline.nohup 2>&1 &
echo "[$(date '+%F %T %Z')] started run_web_search.sh baseline_no_gnca pid=$!" >> $L
nohup setsid bash runner/run_web_search.sh gnca_gate >> $R/logs_web_search_gnca.nohup 2>&1 &
echo "[$(date '+%F %T %Z')] started run_web_search.sh gnca_gate pid=$!" >> $L
