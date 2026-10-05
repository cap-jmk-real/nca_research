#!/usr/bin/env zsh
# One-shot Mac worker prep: DNS (Wi-Fi only), optional sleep/sync/UI, verify.
# Needs your Mac password (sudo) for DNS/sleep. Run in Terminal.app, not the agent sandbox.
#
# Usage:
#   ./scripts/setup_mac_worker.sh              # safe defaults
#   ./scripts/setup_mac_worker.sh --sync       # also rsync repo -> ~/code (no --delete)
#   ./scripts/setup_mac_worker.sh --disable-sleep  # pmset -c sleep 0 (opt-in; prefer caffeinate)
#   ./scripts/setup_mac_worker.sh --all-dns    # DNS on every network service (avoid unless needed)
#   ./scripts/setup_mac_worker.sh --flush-dns  # also HUP mDNSResponder (opt-in)
#   ./scripts/setup_mac_worker.sh --no-ui
#
# NEVER run this concurrently with training/compile/gmx on a 16GB M1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DO_SYNC=0
DO_UI=1
DO_SLEEP=0
ALL_DNS=0
FLUSH_DNS=0
for arg in "$@"; do
  case "$arg" in
    --sync) DO_SYNC=1 ;;
    --no-sync) DO_SYNC=0 ;;
    --no-ui) DO_UI=0 ;;
    --disable-sleep) DO_SLEEP=1 ;;
    --all-dns) ALL_DNS=1 ;;
    --flush-dns) FLUSH_DNS=1 ;;
    --help|-h)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

echo "=== NCA setup_mac_worker (safe mode) ==="
echo "Repo: $ROOT"
echo "Defaults: DNS on Wi-Fi only; no pmset disablesleep; no rsync; no mDNSResponder kill."
echo

# Refuse to thrash the machine if it is already under heavy load.
HEAVY="$(pgrep -fl 'nca_text.hypotheses|compile_specialist|run_auto_research|gmx|mdrun' 2>/dev/null || true)"
if [[ -n "$HEAVY" ]]; then
  echo "REFUSING: heavy jobs already running (training/compile/MD). Stop them first:" >&2
  echo "$HEAVY" | head -20 >&2
  exit 3
fi

# Free-disk guard (Macintosh HD below ~5 GiB has caused jetsam / UI stalls here).
AVAIL_KB="$(df -k / | awk 'NR==2{print $4}')"
if [[ -n "$AVAIL_KB" && "$AVAIL_KB" -lt 5242880 ]]; then
  echo "WARN: only ~$((AVAIL_KB/1024/1024)) GiB free on /. Free space before heavy research." >&2
fi

LOAD="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}' || echo 0)"
# Rough: refuse if 1-min load > 2x CPU count
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
if awk -v l="$LOAD" -v n="$NCPU" 'BEGIN{exit !(l+0 > n*2)}'; then
  echo "REFUSING: load average $LOAD is high (cpus=$NCPU). Let the machine settle, then retry." >&2
  exit 3
fi

echo "Will ask for sudo once (DNS${DO_SLEEP:+ + pmset})."
sudo -v

# DNS: Wi-Fi only by default. Touching every service + HUP mDNSResponder can stall
# networking and WindowServer under load (correlated with a 2026-10-05 watchdog panic).
set_dns() {
  local svc="$1"
  echo "DNS -> $svc"
  sudo networksetup -setdnsservers "$svc" 8.8.8.8 1.1.1.1 || true
}

if [[ "$ALL_DNS" -eq 1 ]]; then
  echo "Aggressive: setting DNS on ALL network services (--all-dns)."
  networksetup -listallnetworkservices | while IFS= read -r svc; do
    [[ "$svc" == *"*"* ]] && continue
    [[ -z "$svc" ]] && continue
    set_dns "$svc"
  done
else
  for svc in "Wi-Fi" "Ethernet" "USB 10/100/1000 LAN"; do
    if networksetup -listallnetworkservices 2>/dev/null | grep -Fxq "$svc"; then
      set_dns "$svc"
    fi
  done
fi

sudo dscacheutil -flushcache || true
if [[ "$FLUSH_DNS" -eq 1 ]]; then
  echo "Sending HUP to mDNSResponder (--flush-dns)."
  sudo killall -HUP mDNSResponder || true
else
  echo "Skipping mDNSResponder HUP (pass --flush-dns if DNS still stale)."
fi

# Sleep: prefer caffeinate from start_auto_research.sh. pmset sleep 0 is opt-in.
if [[ "$DO_SLEEP" -eq 1 ]]; then
  echo "Opt-in: pmset -c sleep 0 disksleep 0 (--disable-sleep)."
  sudo pmset -c sleep 0 disksleep 0 >/dev/null || true
  sudo pmset -c displaysleep 10 >/dev/null || true
else
  echo "Leaving pmset sleep alone (use caffeinate, or pass --disable-sleep)."
fi

echo
echo "Wi-Fi DNS:"
networksetup -getdnsservers Wi-Fi || true
echo "pmset custom:"
pmset -g custom 2>/dev/null | head -20 || pmset -g | head -20

# Sync Documents -> ~/code (TCC-safe worker path). Opt-in; never --delete.
if [[ "$DO_SYNC" -eq 1 ]]; then
  mkdir -p "$HOME/code"
  DEST="$HOME/code/nca_research"
  echo
  echo "Syncing $ROOT -> $DEST (rsync -a, NO --delete) ..."
  rsync -a \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude 'results' \
    --exclude '.git/objects' \
    "$ROOT/" "$DEST/"
  echo "Synced. Point Cursor private worker --worker-dir at: $DEST"
else
  echo
  echo "Skipping ~/code sync (pass --sync if the worker should use ~/code/nca_research)."
fi

if [[ "$DO_UI" -eq 1 ]]; then
  echo
  echo "Opening Full Disk Access (one pane). Enable Cursor + cursor-agent, then restart worker."
  open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles" 2>/dev/null || \
    open "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles" 2>/dev/null || true
  sleep 1
  AGENT_BIN="$HOME/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/agent-cli/.local/bin/cursor-agent"
  if [[ -e "$AGENT_BIN" ]]; then
    echo "cursor-agent: $AGENT_BIN"
    open -R "$AGENT_BIN" 2>/dev/null || true
  fi
  # Do NOT blast multiple browser tabs by default (WindowServer hangs under load).
  echo "Manual links (open one at a time if needed):"
  echo "  https://github.com/settings/installations"
  echo "  https://cursor.com/dashboard?tab=integrations"
fi

echo
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" --quiet || true
"$ROOT/scripts/check_setup.sh" || true

echo
echo "Optional next (do NOT stack with other 100% CPU jobs on 16GB):"
echo "  source scripts/env.sh && ./scripts/install_deps.sh"
echo "  gh auth login -h github.com   # choose HTTPS"
echo "  ./scripts/start_auto_research.sh            # daemon only"
echo "  ./scripts/start_auto_research.sh --with-compile  # only if RAM headroom"
