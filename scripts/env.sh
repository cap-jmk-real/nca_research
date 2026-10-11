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

# A leftover /Volumes/Extreme SSD directory on the boot disk is not the SSD.
_nca_internal_dev="$(stat -f '%d' /)"
_nca_vol_dev="$(stat -f '%d' "$NCA_VOLUME")"
if [[ "$_nca_vol_dev" == "$_nca_internal_dev" ]]; then
  echo "env.sh: $NCA_VOLUME is on the internal disk; refusing to use it" >&2
  return 1 2>/dev/null || exit 1
fi

export NCA_REPO
export NCA_VOLUME
export NCA_ROOT
export NCA_DATA_ROOT="${NCA_DATA_ROOT:-$NCA_ROOT/results}"
case "$NCA_DATA_ROOT" in
  /Volumes/*) ;;
  *)
    echo "env.sh: refusing NCA_DATA_ROOT outside /Volumes: $NCA_DATA_ROOT" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
case "$NCA_DATA_ROOT" in
  *"Macintosh HD"*)
    echo "env.sh: refusing internal NCA_DATA_ROOT: $NCA_DATA_ROOT" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# macOS sets TMPDIR under /var/folders. Override any cache var that is not
# already on an external /Volumes path. Do not keep an internal default.
_nca_pin() {
  local key="$1" dest="$2" cur
  cur="${(P)key-}"
  if [[ -z "$cur" || "$cur" != /Volumes/* || "$cur" == *"Macintosh HD"* ]]; then
    export "$key=$dest"
  fi
  mkdir -p "$dest" 2>/dev/null || true
}

mkdir -p \
  "$NCA_DATA_ROOT" \
  "$NCA_ROOT/datasets" \
  2>/dev/null || true

_nca_pin TMPDIR "$NCA_DATA_ROOT/tmp"
_nca_pin PYTHONPYCACHEPREFIX "$NCA_DATA_ROOT/pycache"
_nca_pin TORCH_HOME "$NCA_DATA_ROOT/torch-home"
_nca_pin TORCHINDUCTOR_CACHE_DIR "$NCA_DATA_ROOT/inductor-cache"
_nca_pin HF_HOME "$NCA_DATA_ROOT/hf-home"
_nca_pin HF_HUB_CACHE "$HF_HOME/hub"
_nca_pin HUGGINGFACE_HUB_CACHE "$HF_HOME/hub"
_nca_pin HF_DATASETS_CACHE "$HF_HOME/datasets"
_nca_pin HF_XET_CACHE "$HF_HOME/xet"
_nca_pin XDG_CACHE_HOME "$NCA_DATA_ROOT/xdg-cache"
_nca_pin PIP_CACHE_DIR "$NCA_DATA_ROOT/pip-cache"
export HF_HUB_DISABLE_TELEMETRY=1
unset _nca_internal_dev _nca_vol_dev
export NCA_ENGINE_HOST="${NCA_ENGINE_HOST:-192.168.10.29}"
export NCA_OLLAMA="${NCA_OLLAMA:-http://$NCA_ENGINE_HOST:11434}"
export NCA_PYTHON="${NCA_PYTHON:-/opt/homebrew/bin/python3.14}"

# Prefer repo checkout under ~/code (TCC-safe) when present.
if [[ -d "$HOME/code/nca_research/.git" && "$NCA_REPO" == *"/Documents/"* ]]; then
  if [[ "${1:-}" != "-q" && "${1:-}" != "--quiet" ]]; then
    echo "env.sh: note — prefer worker-dir ~/code/nca_research (TCC); current=$NCA_REPO" >&2
  fi
fi

if [[ "${1:-}" == "-q" || "${1:-}" == "--quiet" ]]; then
  :
else
  echo "NCA_REPO=$NCA_REPO"
  echo "NCA_DATA_ROOT=$NCA_DATA_ROOT"
  echo "NCA_PYTHON=$NCA_PYTHON"
fi
