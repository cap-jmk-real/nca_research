# Mac self-hosted worker prep (nca_research)

**Date:** 2026-10-05

Use this when agents “lose the remote”, SSD writes fail, or DNS dies.

## Scripts (preferred)

Run in **Terminal.app** (agent sandbox cannot elevate / set DNS):

```bash
# One-shot: DNS + sleep + ~/code sync + FDA UI + healthcheck
./scripts/setup_mac_worker.sh

# Or full entrypoint
./install.sh --deps

# Health only
./scripts/check_setup.sh

# Source env (SSD caches)
source scripts/env.sh

# Start / resume 3-day daemon
./scripts/start_auto_research.sh --with-compile

# Push branch after gh auth login (HTTPS)
./scripts/push_branch.sh
```

Desktop aliases (same thing): `NCA-Fix-Mac-Worker.command`, `NCA-Push-Now.command`.

## Must work

| Check | Healthy | Broken symptom |
| --- | --- | --- |
| Extreme SSD mounted writable | `/Volumes/Extreme SSD/nca_research` | `Operation not permitted` / missing volume |
| Repo readable by worker | Prefer `~/code/nca_research` not `~/Documents/...` | TCC blocks Documents |
| Full Disk Access | Cursor.app + `cursor-agent` | Random PermissionError mid-train / failed `.pt` save |
| DNS | `curl https://api.github.com` | `Could not resolve host` |
| Stay awake | Prevent sleep on power | Worker + training die overnight |
| Engine (optional) | `192.168.10.29:11434` | Distill/Klaut skipped |

## Manual fallbacks

### Full Disk Access
**System Settings → Privacy & Security → Full Disk Access** → enable Cursor + cursor-agent  
(`~/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/agent-cli/.local/bin/cursor-agent`).

### DNS
```bash
sudo networksetup -setdnsservers Wi-Fi 8.8.8.8 1.1.1.1
sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder
```

### Sleep
```bash
sudo pmset -c sleep 0 disksleep 0
caffeinate -dims &
```

### GitHub
```bash
gh auth login -h github.com   # HTTPS
# Cursor GitHub App must include private nca_research:
open https://github.com/settings/installations
```

## After a crash / TCC event

1. Unlock + remount SSD  
2. `./scripts/check_setup.sh` (must pass torch SSD save)  
3. `./scripts/start_auto_research.sh` (preserves deadline from `daemon/state.json`)
