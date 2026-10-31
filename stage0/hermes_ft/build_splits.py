"""Hermes fine-tune data: two label variants on the same sessions + 5 grouped, size-balanced folds.
drop : = stage0_splits/hermes_ood/test.jsonl (tools outside 13 labels dropped, as in curate_traces.py)
other: same row logic, but every call outside the 13 labels becomes target other and stays in history under its raw name."""
import sqlite3, json, sys, random, collections, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import curate_traces as ct
from common import DATA
LABELS = json.load(open(Path(__file__).resolve().parents[1] / "labels.json"))
OUT = DATA / "stage0_splits/hermes_ft"; OUT.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(f"file:{DATA}/hermes_raw/db/state.db?mode=ro", uri=True)
def res(t):
    t = (t or "").strip()
    try: p = json.loads(t)
    except Exception: return ct.truncate(t, ct.RESULT_CHARS)
    return ct.truncate(ct.stringify(p["content"] if isinstance(p, dict) and "content" in p else p), ct.RESULT_CHARS)
rows = []
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
            rows.append({"session_id": sid, "step_index": idx, "goal": goal, "recent_steps": ct.snapshot(hist), "target_tool": "respond"}); idx += 1; continue
        for x in calls:
            n = (x.get("function") or {}).get("name"); tgt = n if n in LABELS else "other"
            rows.append({"session_id": sid, "step_index": idx, "goal": goal, "recent_steps": ct.snapshot(hist), "target_tool": tgt, "raw_tool": n}); idx += 1
            hist.append({"tool": n, "result": ct.PENDING}); pend[x.get("id") or x.get("call_id")] = len(hist) - 1
with open(OUT / "other_all.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
drop = [json.loads(l) for l in open(DATA / "stage0_splits/hermes_ood/test.jsonl")]
# folds: greedy size balance on other row counts (superset of sessions), largest first, ties shuffled with fixed seed
cnt = collections.Counter(r["session_id"] for r in rows); dcnt = collections.Counter(r["session_id"] for r in drop)
rnd = random.Random(20261026); sess = list(cnt); rnd.shuffle(sess); sess.sort(key=lambda s: -cnt[s])
K = 5; load = [0] * K; fold = {}
for s in sess:
    k = min(range(K), key=lambda i: (load[i], i)); fold[s] = k; load[k] += cnt[s]
json.dump(fold, open(OUT / "folds.json", "w"), indent=0)
stats = {"K": K, "seed": 20261026, "rule": "fold k = test, fold (k+1)%5 = val, other 3 folds = train",
  "folds": {k: {"sessions": sum(1 for s in fold if fold[s] == k), "rows_other": load[k], "rows_drop": sum(dcnt[s] for s in fold if fold[s] == k),
                "largest_session_rows_drop": max([dcnt[s] for s in fold if fold[s] == k] or [0]),
                "terminal_pct_drop": round(100 * sum(1 for r in drop if fold[r["session_id"]] == k and r["target_tool"] == "terminal") / max(1, sum(dcnt[s] for s in fold if fold[s] == k)), 1)} for k in range(K)},
  "rows_other": len(rows), "rows_drop": len(drop), "sessions_other": len(cnt), "sessions_drop": len(dcnt),
  "other_labels": dict(collections.Counter(r["target_tool"] for r in rows).most_common())}
(OUT / "split_stats.json").write_text(json.dumps(stats, indent=1)); print(json.dumps(stats, indent=1))
import subprocess; subprocess.run(f"cd {OUT} && sha256sum other_all.jsonl folds.json split_stats.json ../hermes_ood/test.jsonl > SHA256SUMS.txt", shell=True, check=True)
