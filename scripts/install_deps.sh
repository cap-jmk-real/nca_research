#!/usr/bin/env zsh
# Install / refresh Python deps with all caches on Extreme SSD.
# Usage: ./scripts/install_deps.sh [--user]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" --quiet

PY="${NCA_PYTHON:-python3}"
USER_FLAG=()
if [[ "${1:-}" == "--user" ]]; then
  USER_FLAG=(--user)
fi

echo "=== install_deps ==="
echo "python: $PY ($($PY -c 'import sys; print(sys.version.split()[0])'))"
echo "pip cache: $PIP_CACHE_DIR"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$NCA_DATA_ROOT"

# Keep wheels/caches off Macintosh HD.
$PY -m pip install "${USER_FLAG[@]}" --upgrade pip
$PY -m pip install "${USER_FLAG[@]}" -r "$ROOT/requirements.txt"

# Optional specialist / HF tooling (skip quietly if network blocked).
$PY -m pip install "${USER_FLAG[@]}" huggingface_hub 2>/dev/null || \
  echo "note: huggingface_hub skipped (install later for publish_hf.py)"

$PY - <<'PY'
import torch
print("torch", torch.__version__, "mps", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
PY

echo "OK — caches under $NCA_ROOT"
echo "Next: ./scripts/check_setup.sh"
