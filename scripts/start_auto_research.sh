#!/usr/bin/env zsh
# Start / resume the auto-research daemon + optional compile + caffeinate.
# Usage: ./scripts/start_auto_research.sh [--with-compile] [--days N] [--force]
#
# On 16GB M1 Air: do NOT stack daemon training (150k steps) with compile or
# other 100% CPU jobs (e.g. gmx mdrun). That combination previously drove an
# ~11GB Python RSS + watchdog panic (2026-10-05).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

WITH_COMPILE=0
DAYS=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-compile) WITH_COMPILE=1; shift ;;
    --days) DAYS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mem_guard() {
  local reason="$1"
  # Phys free+inactive is soft; also refuse known heavy roommates.
  local others
  others="$(pgrep -fl 'gmx|mdrun|compile_specialist|nca_text.hypotheses' 2>/dev/null || true)"
  if [[ -n "$others" && "$FORCE" -ne 1 ]]; then
    echo "REFUSING ($reason): other heavy CPU/RAM jobs already running:" >&2
    echo "$others" | head -20 >&2
    echo "Stop them, or pass --force if you accept a freeze/watchdog risk." >&2
    return 1
  fi
  local free_mb
  free_mb="$(
    memory_pressure 2>/dev/null | awk '
      /Pages free/ { gsub(/[^0-9]/,"",$3); free=$3 }
      /Pages inactive/ { gsub(/[^0-9]/,"",$3); inactive=$3 }
      END { if (free+inactive>0) print int((free+inactive)*16/1024); else print 0 }
    '
  )"
  # Fallback via vm_stat if memory_pressure parse failed
  if [[ -z "$free_mb" || "$free_mb" -eq 0 ]]; then
    free_mb="$(
      vm_stat 2>/dev/null | awk '
        /Pages free/ { gsub(/\./,"",$3); free=$3 }
        /Pages inactive/ { gsub(/\./,"",$3); inactive=$3 }
        END { print int((free+inactive)*16/1024) }
      '
    )"
  fi
  echo "Memory headroom (free+inactive, approx): ${free_mb:-?} MiB"
  # Need ~4 GiB headroom before starting MPS training on 16GB.
  if [[ -n "$free_mb" && "$free_mb" -lt 4096 && "$FORCE" -ne 1 ]]; then
    echo "REFUSING ($reason): only ~${free_mb} MiB reclaimable; need ≥4096 MiB." >&2
    echo "Quit browsers/MD jobs, or pass --force." >&2
    return 1
  fi
  local avail_kb
  avail_kb="$(df -k / | awk 'NR==2{print $4}')"
  if [[ -n "$avail_kb" && "$avail_kb" -lt 5242880 && "$FORCE" -ne 1 ]]; then
    echo "REFUSING ($reason): only ~$((avail_kb/1024/1024)) GiB free on /." >&2
    return 1
  fi
  return 0
}

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

mem_guard "start_auto_research" || exit 3

mkdir -p "$NCA_DATA_ROOT/auto_research/daemon"
rm -f "$NCA_DATA_ROOT/auto_research/daemon/STOP"

start_sess() {
  local name="$1" cmd="$2"
  if tmux has-session -t "=$name" 2>/dev/null; then
    echo "tmux session exists: $name (reuse; not re-sending keys)"
    return 0
  fi
  tmux new-session -d -s "$name" -c "$ROOT" -- "${SHELL:-zsh}" -l
  tmux send-keys -t "$name:0.0" "$cmd" C-m
}

# Cap BLAS/OMP threads so training cannot pin all 8 cores forever alongside UI.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"

# Caffeinate: prevent idle sleep only. Prefer this over pmset disablesleep.
if command -v caffeinate >/dev/null; then
  # -i idle, -m disk idle; skip -d/-s to avoid fighting display/system power mgmt hard.
  start_sess "nca-caffeine" "caffeinate -im"
fi

# Daemon (preserves deadline when state.json still valid)
DAEMON_CMD="source '$ROOT/scripts/env.sh' --quiet && export OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS && $NCA_PYTHON -u scripts/run_auto_research_daemon.py --days $DAYS --sleep-between 180 2>&1 | tee -a '$NCA_DATA_ROOT/auto_research/daemon/console.log'"
start_sess "nca-daemon" "$DAEMON_CMD"

if [[ "$WITH_COMPILE" -eq 1 ]]; then
  if tmux has-session -t "=nca-daemon" 2>/dev/null && [[ "$FORCE" -ne 1 ]]; then
    echo "WARN: --with-compile refused while nca-daemon is running (16GB RAM)." >&2
    echo "Run compile later alone, or pass --force to override." >&2
  else
    COMP_CMD="source '$ROOT/scripts/env.sh' --quiet && export OMP_NUM_THREADS=$OMP_NUM_THREADS && $NCA_PYTHON -u scripts/compile_specialist_datasets.py --langs python typescript shell --max-rows-per-source 20000 2>&1 | tee -a '$NCA_ROOT/datasets/specialists/compile.log'"
    mkdir -p "$NCA_ROOT/datasets/specialists"
    start_sess "nca-compile" "$COMP_CMD"
  fi
fi

sleep 2
"$NCA_PYTHON" "$ROOT/scripts/status_lean.py" || true
echo
echo "tmux ls:"
tmux ls 2>/dev/null || true
echo "Attach: tmux attach -t nca-daemon"
echo "Note: long train uses NCA_HYP_STEPS=${NCA_HYP_STEPS:-150000}; watch RSS with scripts/status_lean.py"
