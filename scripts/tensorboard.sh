#!/bin/bash
# Launch TensorBoard against the external-SSD log dir. Nothing is written
# under the repo or the home directory. Caches and the package tree stay on
# the Extreme SSD.
set -euo pipefail

SSD="/Volumes/Extreme SSD"
if [[ ! -d "$SSD" ]]; then
  echo "Extreme SSD is not mounted. Refusing to start TensorBoard." >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$SSD/nca_research/results"
LOGDIR="$RESULTS/tb"
PACKAGES="$SSD/nca_research/tb-packages"
TMPDIR="$RESULTS/tmp"
PIP_CACHE_DIR="$RESULTS/pip-cache"
PYCACHE="$RESULTS/pycache"
mkdir -p "$LOGDIR/train" "$LOGDIR/train_grid" "$LOGDIR/benchmarks" "$TMPDIR" "$PIP_CACHE_DIR" "$PYCACHE" "$PACKAGES"

export TMPDIR PIP_CACHE_DIR
export PYTHONPYCACHEPREFIX="$PYCACHE"
# Repo first. tb_log appends the SSD package dir itself so system numpy/torch
# stay ahead of the TensorBoard dependency copies.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

PY="${PYTHON:-python3}"

if ! "$PY" -c "import sys; sys.path.append('$PACKAGES'); import tensorboard" >/dev/null 2>&1; then
  echo "Installing tensorboard into $PACKAGES"
  "$PY" -m pip install --target "$PACKAGES" "tensorboard>=2.18"
fi

"$PY" -m nca_text.tb_log --backfill

FOLLOW_PID="$LOGDIR/follow.pid"
if [[ -f "$FOLLOW_PID" ]]; then
  old="$(cat "$FOLLOW_PID" 2>/dev/null || true)"
  if [[ -n "${old}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "train follower already running (pid $old)"
  else
    rm -f "$FOLLOW_PID"
  fi
fi
if [[ ! -f "$FOLLOW_PID" ]]; then
  nohup "$PY" -m nca_text.tb_log --backfill --follow \
    >>"$RESULTS/logs/tensorboard_follow.log" 2>&1 &
  echo $! >"$FOLLOW_PID"
  echo "following train jsonl (pid $(cat "$FOLLOW_PID"))"
fi

HOST="127.0.0.1"
PORT="${TB_PORT:-6006}"
echo "TensorBoard logdir: $LOGDIR"
echo "URL: http://${HOST}:${PORT}/"
exec env PYTHONPATH="$PACKAGES:$REPO" "$PY" -m tensorboard.main --logdir "$LOGDIR" --host "$HOST" --port "$PORT"
