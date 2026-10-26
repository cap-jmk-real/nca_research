"""Stage 0 per-row predictions for saved best checkpoints + paired bootstrap vs LLM teachers.
Read-only on run dirs; writes ~/nca_data/stage0_results/rowpreds_<date>/. CPU only."""
from __future__ import annotations
import glob, json, sys, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, load_split
from train_eval import pack, build, LABELS
torch.set_num_threads(2)
OUT = DATA / "stage0_results" / "rowpreds_20261026"; OUT.mkdir(parents=True, exist_ok=True)
ex = load_split("curated", "test"); B = 10000; rng = np.random.default_rng(20261026)
def teacher_rows(path):
    P = {}
    for l in open(path):
        r = json.loads(l); pr = r.get("probs") or {}
        P[r["row"]] = (max(pr, key=pr.get) if pr else "terminal", r["gold"])
    return P
teachers = {"sol": os.path.expanduser("~/nca_data/teacher_rows/sol_20261026_0950/preds_sol_test.jsonl")}
for name, pat in (("deepseek", "~/nca_data/teacher_rows/*deepseek*/preds_*test*.jsonl"), ("maverick", "~/nca_data/teacher_rows/*maverick*/preds_*test*.jsonl")):
    g = glob.glob(os.path.expanduser(pat)); 
    if g: teachers[name] = g[0]
T = {k: teacher_rows(v) for k, v in teachers.items()}
sol_rows = sorted(T["sol"])
cache = {}; preds = {}
for d in sorted(glob.glob(str(DATA / "stage0_runs/*"))):
    tag = Path(d).name.split("_", 1)[1].rsplit("_curated", 1)[0]
    if tag.startswith("G1p") or not os.path.exists(d + "/result.json"): continue
    ck = torch.load(d + "/best.pt", map_location="cpu"); a = ck["args"]
    if a["arch"] not in cache:
        X, y = pack(a["arch"], ex); cache[a["arch"]] = (torch.from_numpy(X), y)
    X, y = cache[a["arch"]]
    m = build(a["arch"], X, a["hidden"], a["nca_steps"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad(): p = torch.cat([m(X[i:i+2048]).argmax(1) for i in range(0, len(X), 2048)]).numpy()
    ref = json.load(open(d + "/result.json"))["test"]["top1"]
    acc = round(100 * float((p == y).mean()), 2)
    assert abs(acc - ref) < 0.05, (d, acc, ref)
    preds.setdefault(tag, []).append((a["seed"], p, y))
    with open(OUT / f"{Path(d).name}.jsonl", "w") as f:
        for i in range(len(p)): f.write(json.dumps({"row": i, "pred": LABELS[p[i]], "gold": LABELS[y[i]]}) + "\n")
y = cache["nca1d"][1]
for t, P in T.items():
    for r, (_, gold) in P.items(): assert LABELS[y[r]] == gold, (t, r)
summary = {"n_test": len(y), "bootstrap": B, "teacher_files": teachers, "models": {}}
for tag, runs in preds.items():
    C = np.stack([(p == yy).astype(float) for _, p, yy in runs])  # seeds x rows
    s = {"seeds": len(runs), "full_test_top1_per_seed": [round(100*c.mean(), 2) for c in C], "vs": {}}
    for t, P in T.items():
        rows = np.array(sorted(P)); tc = np.array([LABELS.index(P[r][0]) == y[r] for r in rows], float)
        mc = C[:, rows].mean(0); diff = mc - tc
        bs = [diff[rng.integers(0, len(rows), len(rows))].mean() for _ in range(B)]
        s["vs"][t] = {"rows": len(rows), "model_top1_per_seed": [round(100*C[i, rows].mean(), 2) for i in range(len(runs))],
                      "model_top1_mean": round(100*mc.mean(), 2), "teacher_top1": round(100*tc.mean(), 2),
                      "diff_pp": round(100*diff.mean(), 2), "ci95_pp": [round(100*np.percentile(bs, 2.5), 2), round(100*np.percentile(bs, 97.5), 2)]}
    summary["models"][tag] = s
(OUT / "paired_summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary["models"], indent=0)[:4000])
