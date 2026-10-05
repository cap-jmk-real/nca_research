# Mac self-hosted worker prep (nca_research)

Use this when agents “lose the remote”, SSD writes fail, or DNS dies.

## Must work

| Check | Healthy | Broken symptom |
| --- | --- | --- |
| Extreme SSD mounted writable | `/Volumes/Extreme SSD/nca_research` | `Operation not permitted` / missing volume |
| Repo readable by worker | Prefer `~/code/nca_research` not `~/Documents/...` | TCC blocks Documents |
| Full Disk Access | Cursor.app + `cursor-agent` | Random PermissionError mid-train |
| DNS | `ping github.com` / `curl https://api.github.com` | `Could not resolve host` |
| Stay awake | Prevent sleep on power | Worker + training die overnight |
| Engine (optional) | `192.168.10.29:11434` | Distill/Klaut skipped |

## One-time setup (at the computer)

### 1. SSD
1. Plug in Extreme SSD; unlock if encrypted (Finder password prompt).
2. Confirm Finder shows **Extreme SSD** and you can create a file in  
   `/Volumes/Extreme SSD/nca_research/`.
3. Keep it plugged in while the 3-day mission runs.

### 2. Full Disk Access
**System Settings → Privacy & Security → Full Disk Access** → enable:
- **Cursor**
- **cursor-agent** (add via `+` → go to  
  `~/Library/Application Support/Cursor/User/globalStorage/anysphere.cursor-agent-worker/agent-cli/.local/bin/cursor-agent`  
  or the versioned binary it symlinks to)

Quit and reopen Cursor / restart the private worker after changing this.

### 3. Move worker out of Documents (recommended)
macOS TCC is harsh on `~/Documents`:

```bash
mkdir -p ~/code
# if not already moved:
# rsync -a ~/Documents/coding-projects/nca_research/ ~/code/nca_research/
```

Restart the Cursor private worker with `--worker-dir ~/code/nca_research`.

### 4. DNS
**System Settings → Network → Wi‑Fi → Details → DNS** → add:
- `8.8.8.8`
- `1.1.1.1`

Apply, then:

```bash
dscacheutil -flushcache; sudo killall -HUP mDNSResponder
curl -I https://api.github.com
```

### 5. Sleep
**System Settings → Energy** (or Battery → Options):
- Prevent automatic sleeping on power adapter when display is off
- Or leave lid open on adapter

Optional during long trains:

```bash
caffeinate -dims &
```

### 6. GitHub App (PR creation)
[Installations](https://github.com/settings/installations) → Cursor → include **nca_research**.  
[Cursor Integrations](https://cursor.com/dashboard?tab=integrations) → GitHub connected.

### 7. Hugging Face (later publish)
Create `HF_TOKEN` with write access; do not commit it.  
See `docs/huggingface-distribution.md`.

## Quick health script

```bash
python3 scripts/status_lean.py
ls "/Volumes/Extreme SSD/nca_research/results/auto_research/daemon/heartbeat.json"
curl -sS --max-time 5 -o /dev/null -w "%{http_code}\n" https://api.github.com
curl -sS --max-time 3 -o /dev/null -w "%{http_code}\n" http://192.168.10.29:11434/api/tags || true
```

## After a crash / TCC event

1. Unlock + remount SSD if needed  
2. Confirm torch can save to SSD (agent will smoke-test)  
3. Restart daemon:

```bash
python scripts/run_auto_research_daemon.py --days <remaining> --sleep-between 180
```

Deadline is preserved in `daemon/state.json` when still in the future.
