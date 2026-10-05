#!/usr/bin/env zsh
# Start / resume the 3-day auto-research daemon + optional compile + caffeinate.
# Usage: ./scripts/start_auto_research.sh [--with-compile] [--days N]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

WITH_COMPILE=0
DAYS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-compile) WITH_COMPILE=1; shift ;;
    --days) DAYS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DAYS" ]]; then
  DAYS="$("$NCA_PYTHON" - <<'PY'
import json, time
from pathlib import Path
p = Path("/Volumes/Extreme SSD/nca_research/results/auto_research/daemon/state.json")
if p.is_file():
    s = json.loads(p.read_text())
    left = float(s.get("deadline_unix") or 0) - time.time()
    print(max(0.05, left / 86400) if left > 0 else 3.0)
else:
    print(3.0)
PY
)"
fi

command -v tmux >/dev/null || { echo "tmux required"; exit 1; }
command -v caffeinate >/dev/null || true

mkdir -p "$NCA_DATA_ROOT/auto_research/daemon"
rm -f "$NCA_DATA_ROOT/auto_research/daemon/STOP"

start_sess() {
  local name="$1" cmd="$2"
  if tmux has-session -t "=$name" 2>/dev/null; then
    echo "tmux session exists: $name (reuse)"
  else
    tmux new-session -d -s "$name" -c "$ROOT" -- "${SHELL:-zsh}" -l
  fi
  tmux send-keys -t "$name:0.0" "$cmd" C-m
}

# Caffeinate
if command -v caffeinate >/dev/null; then
  start_sess "nca-caffeine" "caffeinate -dims"
fi

# Daemon (preserves deadline when state.json still valid)
DAEMON_CMD="source '$ROOT/scripts/env.sh' --quiet && $NCA_PYTHON -u scripts/run_auto_research_daemon.py --days $DAYS --sleep-between 180 2>&1 | tee -a '$NCA_DATA_ROOT/auto_research/daemon/console.log'"
start_sess "nca-daemon" "$DAEMON_CMD"

if [[ "$WITH_COMPILE" -eq 1 ]]; then
  COMP_CMD="source '$ROOT/scripts/env.sh' --quiet && $NCA_PYTHON -u scripts/compile_specialist_datasets.py --langs python typescript shell --max-rows-per-source 20000 2>&1 | tee -a '$NCA_ROOT/datasets/specialists/compile.log'"
  mkdir -p "$NCA_ROOT/datasets/specialists"
  start_sess "nca-compile" "$COMP_CMD"
fi

sleep 2
"$NCA_PYTHON" "$ROOT/scripts/status_lean.py" || true
echo
echo "tmux ls:"
tmux ls 2>/dev/null || true
echo "Attach: tmux attach -t nca-daemon"
