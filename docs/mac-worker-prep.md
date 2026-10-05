# Mac self-hosted worker prep (nca_research)

**Date:** 2026-10-05

Use this when agents “lose the remote”, SSD writes fail, or DNS dies.

## Scripts (preferred)

Run in **Terminal.app** (agent sandbox cannot elevate / set DNS):

```bash
# Safe one-shot: Wi-Fi DNS only + FDA UI + healthcheck
# (no pmset disablesleep, no rsync, no mDNSResponder kill, no browser blast)
./scripts/setup_mac_worker.sh

# Optional flags when needed:
#   --sync            rsync repo → ~/code (never --delete)
#   --disable-sleep   pmset -c sleep 0 (prefer caffeinate instead)
#   --all-dns         DNS on every network service (avoid)
#   --flush-dns       HUP mDNSResponder (avoid under load)

# Or full entrypoint
./install.sh --deps

# Health only
./scripts/check_setup.sh

# Source env (SSD caches)
source scripts/env.sh

# Start / resume daemon ONLY (memory-guarded on 16GB)
./scripts/start_auto_research.sh

# Compile datasets separately when RAM is free — never stack with 150k train:
# ./scripts/start_auto_research.sh --with-compile

# Push branch after gh auth login (HTTPS)
./scripts/push_branch.sh
```

Desktop aliases (same thing): `NCA-Fix-Mac-Worker.command`, `NCA-Push-Now.command`.

### Crash note (2026-10-05)

A kernel **watchdog timeout** panic (`no checkins from watchdogd in 94s`) coincided with
`NCA-Fix-Mac-Worker.command` plus two Python jobs (~11 GiB + ~0.6 GiB RSS) and heavy
`UVFSService` SSD I/O on a 16 GB M1 Air. Scripts above are now refuse-heavy /
non-stacking by default.

## Must work

| Check | Healthy | Broken symptom |
| --- | --- | --- |
| Extreme SSD mounted writable | `/Volumes/Extreme SSD/nca_research` | `Operation not permitted` / missing volume |
| Repo readable by worker | Prefer `~/code/nca_research` not `~/Documents/...` | TCC blocks Documents |
| Full Disk Access | Cursor.app + `cursor-agent` | Random PermissionError mid-train / failed `.pt` save |
| DNS | `curl https://api.github.com` | `Could not resolve host` |
| Stay awake | Prevent sleep on power | Worker + training die overnight |
| Engine (optional) | `192.168.10.29:11434` | Distill/Klaut skipped |
| RAM headroom | ≥4 GiB free+inactive before train | Watchdog / freeze on 16 GB |

## Manual fallbacks

### Full Disk Access
**System Settings → Privacy & Security → Full Disk Access** → enable Cursor + cursor-agent  
(`~/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/agent-cli/.local/bin/cursor-agent`).

### DNS
```bash
# Prefer Wi-Fi only. Avoid looping every networksetup service under load.
sudo networksetup -setdnsservers Wi-Fi 8.8.8.8 1.1.1.1
sudo dscacheutil -flushcache
# Only if still stale: sudo killall -HUP mDNSResponder
```

### Sleep
```bash
# Prefer caffeinate over pmset disablesleep:
caffeinate -im &
# Opt-in only: sudo pmset -c sleep 0 disksleep 0
```

### GitHub
```bash
gh auth login -h github.com   # HTTPS
# Cursor GitHub App must include private nca_research:
open https://github.com/settings/installations
```

## After a crash / TCC event

1. Unlock + remount SSD; free Macintosh HD space if &lt;5 GiB
2. Stop other 100% CPU jobs (`gmx`, stacked compiles)
3. `./scripts/check_setup.sh` (must pass torch SSD save)
4. `./scripts/start_auto_research.sh` alone (preserves deadline from `daemon/state.json`)
