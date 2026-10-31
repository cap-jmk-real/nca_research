"""Build hermes_ood eval rows from Julian local Hermes state.db with the curate_traces.py
row logic (goal=last user msg[:500], last 4 steps, result[:500], respond=no-tool turn).
Tools outside the 13 labels (browser_*, web_search, others) are dropped like unmapped calls."""
import sqlite3, json, hashlib, collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import curate_traces as ct
from common import DATA, dist
LABELS = json.load(open(Path(__file__).with_name("labels.json")))
SRC = DATA / "hermes_raw/db/state.db"; OUT = DATA / "stage0_splits/hermes_ood"; OUT.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
def res(t):
    t = (t or "").strip()
    try: p = json.loads(t)
    except Exception: return ct.truncate(t, ct.RESULT_CHARS)
    return ct.truncate(ct.stringify(p["content"] if isinstance(p, dict) and "content" in p else p), ct.RESULT_CHARS)
raw = collections.Counter(); kept = collections.Counter(); dropped = collections.Counter(); rows = []; per_sess = collections.Counter()
for (sid,) in c.execute("select id from sessions order by started_at"):
    goal = ""; hist = []; pend = {}; idx = 0
    for role, content, tcs, tcid in c.execute("select role,content,tool_calls,tool_call_id from messages where session_id=? and (active=1 or compacted=1) order by timestamp,id", (sid,)):
        if role == "user": goal = ct.truncate((content or "").strip(), ct.GOAL_CHARS); continue
        if role == "tool":
            if tcid in pend: hist[pend.pop(tcid)]["result"] = res(content)
            continue
        if role != "assistant": continue
        calls = json.loads(tcs) if tcs else []
        if not calls:
            rows.append({"session_id": sid, "step_index": idx, "goal": goal, "recent_steps": ct.snapshot(hist), "target_tool": "respond"}); idx += 1; per_sess[sid] += 1; kept["respond"] += 1; continue
        for x in calls:
            n = (x.get("function") or {}).get("name"); raw[n] += 1
            if n not in LABELS: dropped[n] += 1; continue
            rows.append({"session_id": sid, "step_index": idx, "goal": goal, "recent_steps": ct.snapshot(hist), "target_tool": n}); idx += 1; per_sess[sid] += 1; kept[n] += 1
            hist.append({"tool": n, "result": ct.PENDING}); pend[x.get("id") or x.get("call_id")] = len(hist) - 1
p = OUT / "test.jsonl"
with open(p, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
tot = sum(raw.values())
summ = {"source": "Hermes state.db (Julian local), active+compacted messages", "rows": len(rows), "sessions_with_rows": len(per_sess),
        "sessions_total": c.execute("select count(*) from sessions").fetchone()[0], "raw_tool_calls": tot,
        "mapped_tool_calls": tot - sum(dropped.values()), "unmapped_share_pct": round(100 * sum(dropped.values()) / tot, 2),
        "dropped_calls": dict(dropped.most_common()), "labels": dist([r["target_tool"] for r in rows]),
        "rows_per_session": sorted(per_sess.values(), reverse=True)}
(OUT / "summary.json").write_text(json.dumps(summ, indent=1))
import subprocess; subprocess.run("cd %s && sha256sum test.jsonl summary.json > SHA256SUMS.txt" % OUT, shell=True, check=True)
s = dict(summ); print(json.dumps(s, indent=0))
