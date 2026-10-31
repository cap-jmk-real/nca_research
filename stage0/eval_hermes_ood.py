"""Score saved best.pt of all stage0 runs (excl. G1p grokking) on hermes_ood; CPU only, read-only on runs."""
import glob, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, load_split, metrics
from train_eval import pack, build, LABELS
from obs_pack import fit_or_load_tfidf
torch.set_num_threads(2)
ex = load_split("hermes_ood", "test"); sess = np.array([e.session_id if hasattr(e,"session_id") else e[0] for e in ex])
OUT = DATA / "stage0_results/hermes_ood_20261026"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261026); B = 10000
us = np.unique(sess); members = [np.where(sess == s)[0] for s in us]
def cluster_ci(v):
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(us), len(us)); ix = np.concatenate([members[i] for i in pick]); bs.append(v[ix].mean())
    return [round(100*np.percentile(bs, 2.5), 2), round(100*np.percentile(bs, 97.5), 2)]
def row_ci(v):
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(100*np.percentile(bs, 2.5), 2), round(100*np.percentile(bs, 97.5), 2)]
cache = {}; res = {}
for d in sorted(glob.glob(str(DATA / "stage0_runs/*"))):
    tag = Path(d).name.split("_", 1)[1].rsplit("_curated", 1)[0]
    if tag.startswith("G1p") or not os.path.exists(d + "/result.json"): continue
    ck = torch.load(d + "/best.pt", map_location="cpu"); a = ck["args"]
    if a.get("distill"): continue
    tf = fit_or_load_tfidf(load_split("curated", "train"), a["seed"]) if a.get("text_feat") == "tfidf_svd128" else None
    key = (a["arch"], a.get("text_feat", "hash48"))
    if key not in cache:
        X, y = pack(a["arch"], ex, a.get("text_feat", "hash48"), tf); cache[key] = (torch.from_numpy(X), y)
    X, y = cache[key]
    m = build(a["arch"], X, a["hidden"], a["nca_steps"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad(): lg = torch.cat([m(X[i:i+1024]) for i in range(0, len(X), 1024)])
    ranked = torch.argsort(lg, 1, descending=True).tolist(); mt = metrics(y.tolist(), ranked, len(LABELS))
    res.setdefault(tag, {"arch": a["arch"], "runs": []})["runs"].append({"seed": a["seed"], **mt, "correct": (lg.argmax(1).numpy() == y).astype(float)})
y = cache["nca1d"][1]; term = (y == LABELS.index("terminal")).astype(float)
out = {"n_rows": int(len(y)), "n_sessions": int(len(us)), "bootstrap": B, "floor_terminal": {"top1": round(100*term.mean(), 2), "ci_row": row_ci(term), "ci_session": cluster_ci(term)}, "models": {}}
for tag, r in res.items():
    C = np.stack([x.pop("correct") for x in r["runs"]]); mc = C.mean(0)
    t1 = [x["top1"] for x in r["runs"]]; t3 = [x["top3"] for x in r["runs"]]; f1 = [x["macro_f1"] for x in r["runs"]]
    sd = lambda v: round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else None
    out["models"][tag] = {"arch": r["arch"], "seeds": len(t1), "top1_mean": round(float(np.mean(t1)), 2), "top1_sd": sd(t1),
        "top3_mean": round(float(np.mean(t3)), 2), "top3_sd": sd(t3), "macro_f1_mean": round(float(np.mean(f1)), 2), "macro_f1_sd": sd(f1),
        "ci_row_seedmean": row_ci(mc), "ci_session_seedmean": cluster_ci(mc), "diff_vs_floor_pp": round(100*(mc-term).mean(), 2),
        "diff_vs_floor_ci_session": cluster_ci(mc-term), "per_seed": r["runs"]}
(OUT / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "models"}))
for t, v in out["models"].items(): print(t, {k: v[k] for k in v if k != "per_seed"})
