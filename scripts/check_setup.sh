#!/usr/bin/env zsh
# Health check for Mac self-hosted worker + SSD + DNS + training readiness.
# Usage: ./scripts/check_setup.sh
# Exit 0 if critical checks pass; 1 if SSD/DNS/repo broken.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh" --quiet || {
  echo "FAIL: cannot source env.sh (SSD missing?)"
  exit 1
}

FAIL=0
ok() { echo "OK  $*"; }
bad() { echo "FAIL $*"; FAIL=1; }
warn() { echo "WARN $*"; }

echo "=== check_setup ==="

# Repo
if [[ -d "$ROOT/.git" ]]; then
  ok "repo $ROOT ($(git -C "$ROOT" branch --show-current 2>/dev/null || echo '?'))"
else
  bad "not a git repo: $ROOT"
fi
if [[ "$ROOT" == *"/Documents/"* ]]; then
  warn "repo under Documents (TCC risk) — prefer ~/code/nca_research"
fi

# SSD mount + write
if [[ -d "$NCA_VOLUME" ]]; then
  ok "volume $NCA_VOLUME"
else
  bad "volume missing: $NCA_VOLUME (plug in / unlock Extreme SSD)"
fi

PROBE="$NCA_DATA_ROOT/_setup_probe_$$"
if mkdir -p "$NCA_DATA_ROOT" 2>/dev/null && echo ok >"$PROBE" 2>/dev/null; then
  rm -f "$PROBE"
  ok "SSD write $NCA_DATA_ROOT"
else
  bad "SSD write denied — Full Disk Access for Cursor/cursor-agent; unlock volume"
fi

# Torch save (the failure mode that killed H6 150k)
"$NCA_PYTHON" - <<PY 2>/dev/null && ok "torch SSD checkpoint save" || bad "torch cannot save to SSD"
from pathlib import Path
import torch
p = Path("$NCA_DATA_ROOT") / "hypothesis_tests" / "checkpoints" / "_setup_probe.pt"
p.parent.mkdir(parents=True, exist_ok=True)
torch.save({"ok": True}, p)
assert p.is_file()
p.unlink()
print("torch_ok")
PY

# DNS
if curl -sS --max-time 8 -o /dev/null -w "" https://api.github.com 2>/dev/null; then
  ok "DNS api.github.com"
else
  bad "DNS — run ./scripts/setup_mac_worker.sh (sets 8.8.8.8 / 1.1.1.1)"
fi

# gh
if command -v gh >/dev/null 2>&1; then
  if gh auth status -h github.com >/dev/null 2>&1; then
    ok "gh auth"
  else
    warn "gh not logged in — gh auth login -h github.com (HTTPS)"
  fi
else
  warn "gh CLI missing"
fi

# Engine (optional)
if curl -sS --connect-timeout 2 --max-time 3 -o /dev/null "http://${NCA_ENGINE_HOST}:11434/api/tags" 2>/dev/null; then
  ok "engine Ollama ${NCA_ENGINE_HOST}:11434"
else
  warn "engine Ollama down (${NCA_ENGINE_HOST}) — distill/Klaut will wait"
fi

# Sleep hint (prefer caffeinate over pmset disablesleep on 16GB Air)
if pgrep -x caffeinate >/dev/null 2>&1; then
  ok "caffeinate running"
elif pmset -g custom 2>/dev/null | grep -q "sleep.*0" || pmset -g 2>/dev/null | grep -E "^\s*sleep\s+0" >/dev/null; then
  ok "sleep settings look awake-friendly (pmset)"
else
  warn "system may sleep — prefer: caffeinate -im  (avoid pmset disablesleep unless needed)"
fi

# Disk / roommate warnings for 16GB M1
AVAIL_KB="$(df -k / | awk 'NR==2{print $4}')"
if [[ -n "$AVAIL_KB" && "$AVAIL_KB" -lt 5242880 ]]; then
  warn "low free space on / (~$((AVAIL_KB/1024/1024)) GiB) — free disk before train/compile"
fi
if pgrep -fl 'gmx|mdrun' >/dev/null 2>&1; then
  warn "gmx/mdrun running — do not start NCA train/compile on 16GB until it finishes"
fi
if [[ -f "$HOME/Library/LaunchAgents/com.nca.autoresearch.resume.plist" ]]; then
  ok "reboot resume agent installed"
else
  warn "reboot resume agent not installed — ./scripts/resume_after_reboot.sh install"
fi

# Lean status if daemon exists
if [[ -f "$ROOT/scripts/status_lean.py" ]]; then
  echo "--- status_lean ---"
  "$NCA_PYTHON" "$ROOT/scripts/status_lean.py" || true
fi

echo "=== done (fail=$FAIL) ==="
exit "$FAIL"
