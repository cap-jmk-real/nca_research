#!/usr/bin/env zsh
# Top-level install entrypoint.
# Usage: ./install.sh [--worker-only] [--deps]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKER_ONLY=0
DEPS=0
for arg in "$@"; do
  case "$arg" in
    --worker-only) WORKER_ONLY=1 ;;
    --deps) DEPS=1 ;;
  esac
done

chmod +x "$ROOT"/scripts/*.sh 2>/dev/null || true

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Running Mac worker prep (sudo password may be required)..."
  "$ROOT/scripts/setup_mac_worker.sh" ${WORKER_ONLY:+--no-sync}
else
  echo "Non-Darwin: sourcing env only"
  # shellcheck disable=SC1091
  source "$ROOT/scripts/env.sh"
fi

if [[ "$DEPS" -eq 1 ]]; then
  "$ROOT/scripts/install_deps.sh"
fi

"$ROOT/scripts/check_setup.sh"
echo
echo "Install entry done. Start mission with: ./scripts/start_auto_research.sh --with-compile"
