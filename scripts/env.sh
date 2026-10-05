#!/bin/zsh
# Sourceable environment for nca_research (M1 + Extreme SSD).
# Usage: source scripts/env.sh
#
# Does not install packages. Prefer: scripts/install_deps.sh

setopt err_return 2>/dev/null || set -e

NCA_REPO="${NCA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
NCA_VOLUME="${NCA_VOLUME:-/Volumes/Extreme SSD}"
NCA_ROOT="${NCA_ROOT:-$NCA_VOLUME/nca_research}"

if [[ ! -d "$NCA_VOLUME" ]]; then
  echo "env.sh: Extreme SSD not mounted at $NCA_VOLUME" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p \
  "$NCA_ROOT/results" \
  "$NCA_ROOT/results/tmp" \
  "$NCA_ROOT/results/pycache" \
  "$NCA_ROOT/results/torch-home" \
  "$NCA_ROOT/results/inductor-cache" \
  "$NCA_ROOT/results/hf-home" \
  "$NCA_ROOT/datasets" \
  "$NCA_ROOT/tmp" \
  2>/dev/null || true

export NCA_REPO
export NCA_VOLUME
export NCA_ROOT
export NCA_DATA_ROOT="${NCA_DATA_ROOT:-$NCA_ROOT/results}"
export TMPDIR="${TMPDIR:-$NCA_ROOT/tmp}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$NCA_DATA_ROOT/pycache}"
export TORCH_HOME="${TORCH_HOME:-$NCA_DATA_ROOT/torch-home}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$NCA_DATA_ROOT/inductor-cache}"
export HF_HOME="${HF_HOME:-$NCA_DATA_ROOT/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$NCA_ROOT/pip-cache}"
export NCA_ENGINE_HOST="${NCA_ENGINE_HOST:-192.168.10.29}"
export NCA_OLLAMA="${NCA_OLLAMA:-http://$NCA_ENGINE_HOST:11434}"
export NCA_PYTHON="${NCA_PYTHON:-/opt/homebrew/bin/python3.14}"

# Prefer repo checkout under ~/code (TCC-safe) when present.
if [[ -d "$HOME/code/nca_research/.git" && "$NCA_REPO" == *"/Documents/"* ]]; then
  echo "env.sh: note — prefer worker-dir ~/code/nca_research (TCC); current=$NCA_REPO" >&2
fi

if [[ "${1:-}" == "-q" || "${1:-}" == "--quiet" ]]; then
  :
else
  echo "NCA_REPO=$NCA_REPO"
  echo "NCA_DATA_ROOT=$NCA_DATA_ROOT"
  echo "NCA_PYTHON=$NCA_PYTHON"
fi
