#!/usr/bin/env zsh
# Resume an in-progress auto-research mission after login/reboot.
#
# Waits for the Extreme SSD (SanDisk unlock can lag login), then starts the
# existing daemon via start_auto_research.sh (caffeinate, SSD paths, memory
# guard). Does not start a new mission, and does not clear a STOP file.
#
# Usage:
#   ./scripts/resume_after_reboot.sh            # one attempt
#   ./scripts/resume_after_reboot.sh install    # LaunchAgent: login + every 10 min
#   ./scripts/resume_after_reboot.sh uninstall
#
# Launchd cannot read ~/Documents (TCC). `install` mirrors the repo to
# ~/code/nca_research and points the agent there. Re-run install after
# pulling so that copy stays current.
#
# The agent is idempotent: if the daemon is already running, it exits.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOLUME="/Volumes/Extreme SSD"
DATA="$VOLUME/nca_research/results"
STATE="$DATA/auto_research/daemon/state.json"
STOP="$DATA/auto_research/daemon/STOP"
LABEL="com.nca.autoresearch.resume"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
FALLBACK_LOG="$HOME/Library/Logs/nca-auto-research-resume.log"
LOCKDIR="$HOME/Library/Logs/nca-auto-research-resume.lockdir"
WAIT_SECS="${NCA_RESUME_WAIT_SECS:-180}"
UID_NUM="$(id -u)"

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

log() {
  local line
  line="$(date -Iseconds) $*"
  printf '%s\n' "$line"
  mkdir -p "$(dirname "$FALLBACK_LOG")"
  printf '%s\n' "$line" >> "$FALLBACK_LOG"
  if [[ -d "$DATA/auto_research/daemon" ]]; then
    # Launchd often cannot write the external volume; don't fail the resume on that.
    { printf '%s\n' "$line" >> "$DATA/auto_research/daemon/reboot_resume.log" } 2>/dev/null || true
  fi
}

install_agent() {
  local dest="$HOME/code/nca_research"
  local user_name
  user_name="$(id -un)"
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs" "$dest"
  # LaunchAgents cannot open scripts under ~/Documents. Mirror to ~/code.
  rsync -a \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude 'results' \
    --exclude '.git/objects' \
    "$ROOT/" "$dest/"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/login</string>
    <string>-qf</string>
    <string>${user_name}</string>
    <string>${dest}/scripts/resume_after_reboot.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>StartInterval</key>
  <integer>600</integer>
  <key>StandardOutPath</key>
  <string>${FALLBACK_LOG}</string>
  <key>StandardErrorPath</key>
  <string>${FALLBACK_LOG}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF
  launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
  launchctl bootstrap "gui/${UID_NUM}" "$PLIST"
  launchctl enable "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
  echo "installed ${PLIST}"
  echo "loaded gui/${UID_NUM}/${LABEL} from ${dest} (at login + every 10 min; no-op if already running)"
}

uninstall_agent() {
  launchctl bootout "gui/${UID_NUM}/${LABEL}" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed ${LABEL}"
}

case "${1:-}" in
  install) install_agent; exit 0 ;;
  uninstall) uninstall_agent; exit 0 ;;
  "") ;;
  *) echo "usage: $0 [install|uninstall]" >&2; exit 2 ;;
esac

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  log "resume already in progress"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

if pgrep -f "run_auto_research_daemon.py" >/dev/null 2>&1; then
  log "daemon already running"
  exit 0
fi

elapsed=0
until [[ -d "$DATA" && -w "$DATA" ]]; do
  if (( elapsed >= WAIT_SECS )); then
    log "SSD not writable at ${DATA} after ${WAIT_SECS}s (unlock the Extreme SSD)"
    exit 0
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

if [[ -f "$STOP" ]]; then
  log "STOP file present; not resuming"
  exit 0
fi

if [[ ! -f "$STATE" ]]; then
  log "no daemon state; not starting a new mission"
  exit 0
fi

# deadline_unix is display-only. A passed deadline must not block resume.
if ! python3 - "$STATE" <<'PY'
import json, sys
from pathlib import Path
json.loads(Path(sys.argv[1]).read_text())
PY
then
  log "could not read daemon state"
  exit 0
fi

# A tmux session left behind without the daemon will not accept a new command.
if tmux has-session -t "=nca-daemon" 2>/dev/null; then
  log "removing stale nca-daemon tmux session"
  tmux kill-session -t "=nca-daemon" 2>/dev/null || true
fi

log "resuming auto-research from ${STATE}"
set +e
"$ROOT/scripts/start_auto_research.sh"
code=$?
set -e
if [[ "$code" -ne 0 ]]; then
  log "start_auto_research.sh exited ${code} (will retry on the next interval)"
fi
exit 0
