#!/usr/bin/env zsh
# One-shot Mac worker prep: DNS, sleep, optional ~/code sync, FDA UI, verify.
# Needs your Mac password (sudo). Run in Terminal.app, not the agent sandbox.
#
# Usage: ./scripts/setup_mac_worker.sh [--no-sync] [--no-ui]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DO_SYNC=1
DO_UI=1
for arg in "$@"; do
  case "$arg" in
    --no-sync) DO_SYNC=0 ;;
    --no-ui) DO_UI=0 ;;
  esac
done

echo "=== NCA setup_mac_worker ==="
echo "Repo: $ROOT"
echo "Will set DNS (8.8.8.8 / 1.1.1.1), reduce AC sleep, verify Extreme SSD."
echo

sudo -v

# DNS on every hardware service
networksetup -listallnetworkservices | while IFS= read -r svc; do
  [[ "$svc" == *"*"* ]] && continue
  [[ -z "$svc" ]] && continue
  echo "DNS -> $svc"
  sudo networksetup -setdnsservers "$svc" 8.8.8.8 1.1.1.1 || true
done
sudo dscacheutil -flushcache || true
sudo killall -HUP mDNSResponder || true

# Stay awake on power (display may sleep)
sudo pmset -c sleep 0 disksleep 0 >/dev/null || true
sudo pmset -c displaysleep 10 >/dev/null || true

echo
echo "Wi-Fi DNS:"
networksetup -getdnsservers Wi-Fi || true
echo "pmset custom:"
pmset -g custom 2>/dev/null | head -20 || pmset -g | head -20

# Sync Documents -> ~/code (TCC-safe worker path)
if [[ "$DO_SYNC" -eq 1 ]]; then
  mkdir -p "$HOME/code"
  DEST="$HOME/code/nca_research"
  echo
  echo "Syncing $ROOT -> $DEST ..."
  rsync -a \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude 'results' \
    "$ROOT/" "$DEST/"
  echo "Synced. Point Cursor private worker --worker-dir at: $DEST"
fi

if [[ "$DO_UI" -eq 1 ]]; then
  echo
  echo "Opening Full Disk Access — enable Cursor + cursor-agent, then restart worker."
  open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles" 2>/dev/null || \
    open "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles" 2>/dev/null || true
  AGENT_BIN="$HOME/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/agent-cli/.local/bin/cursor-agent"
  if [[ -e "$AGENT_BIN" ]]; then
    echo "cursor-agent: $AGENT_BIN"
    open -R "$AGENT_BIN" 2>/dev/null || true
  fi
  open "https://github.com/settings/installations" 2>/dev/null || true
  open "https://cursor.com/dashboard?tab=integrations" 2>/dev/null || true
fi

echo
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" --quiet || true
"$ROOT/scripts/check_setup.sh" || true

echo
echo "Optional next:"
echo "  source scripts/env.sh && ./scripts/install_deps.sh"
echo "  gh auth login -h github.com   # choose HTTPS"
echo "  ./scripts/start_auto_research.sh"
