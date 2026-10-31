"""Fixed, seeded session-level splits. Curated: keep train; split original val
sessions 50/50 into val/test. Hermes: 20/6/7 sessions (run only on engine)."""
from __future__ import annotations
import json, random, sys
from collections import defaultdict
from common import DATA, SPLITS, SPLIT_SEED, dist

def write(name, parts):
    d = SPLITS / name; d.mkdir(parents=True, exist_ok=False)  # never overwrite
    summary = {"seed": SPLIT_SEED}
    for split, rows in parts.items():
        with open(d / f"{split}.jsonl", "w") as fh:
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = {"rows": len(rows), "sessions": len({r["session_id"] for r in rows}),
                          "labels": dist([r["target_tool"] for r in rows])}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (v if k == "seed" else {"rows": v["rows"], "sessions": v["sessions"]}) for k, v in summary.items()}))

def by_session(path):
    s = defaultdict(list)
    for line in open(path):
        r = json.loads(line); s[r["session_id"]].append(r)
    return s

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "curated":
        src = DATA / "datasets" / "curated"
        train = [r for rows in by_session(src / "train.jsonl").values() for r in rows]
        val = by_session(src / "val.jsonl"); ids = sorted(val); random.Random(SPLIT_SEED).shuffle(ids)
        h = len(ids) // 2
        write("curated", {"train": train, "val": [r for i in ids[:h] for r in val[i]],
                          "test": [r for i in ids[h:] for r in val[i]]})
    elif which == "hermes":
        ses = by_session(sys.argv[2]); ids = sorted(ses); random.Random(SPLIT_SEED).shuffle(ids)
        write("hermes", {"train": [r for i in ids[:20] for r in ses[i]],
                         "val": [r for i in ids[20:26] for r in ses[i]],
                         "test": [r for i in ids[26:33] for r in ses[i]]})
