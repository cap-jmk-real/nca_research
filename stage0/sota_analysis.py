"""Paired bootstraps for SOTA follow-up. Curated test: rows; Hermes: sessions. 10k resamples, per-row correctness averaged over 3 seeds."""
import json, glob, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_split, metrics
from train_eval import pack, build, LABELS
O = DATA / "stage0_results" / "sota_baselines_20261026"; RP = DATA / "stage0_results" / "rowpreds_20261026"
B = 10000; rng = np.random.default_rng(20261026)
P = [json.load(open(O / f"rowpreds_v2_s{s}.json")) for s in range(3)]
gold = {k: np.array(P[0]["gold"][k]) for k in ("test", "hermes")}
def acc(name, split): return np.stack([(np.array(p[name][split]) == gold[split]).astype(float) for p in P])
def ms(C): t = 100 * C.mean(1); return round(float(t.mean()), 2), round(float(t.std(ddof=1)), 2)
def nn_rows(tag):
    fs = sorted(glob.glob(str(RP / f"*_{tag}_curated_s*.jsonl"))); assert len(fs) == 3, tag
    return np.stack([[float(json.loads(l)["pred"] == json.loads(l)["gold"]) for l in open(f)] for f in fs])
def row_boot(d):
    n = len(d); bs = [d[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)]
sess = np.array(P[0]["hermes_session"]); us = np.unique(sess); mem = [np.where(sess == u)[0] for u in us]
def sess_boot(d):
    bs = []
    for _ in range(B):
        ix = np.concatenate([mem[i] for i in rng.integers(0, len(us), len(us))]); bs.append(d[ix].mean())
    return [round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)]
R = {"bootstrap": B, "curated_test": {}, "hermes": {}}
models = {m: acc(m, "test") for m in ("hgb", "hgb_noresult", "logreg")}
nn = {"H1_nocw": nn_rows("H1_nocw"), "B_tx": nn_rows("B_tx")}
for m, C in {**models, **nn}.items(): R["curated_test"][m] = dict(zip(("top1_mean", "top1_sd"), ms(C)))
for a in ("hgb", "hgb_noresult", "logreg"):
    for b in ("H1_nocw", "B_tx"):
        d = models[a].mean(0) - nn[b].mean(0); R["curated_test"][f"{a}_minus_{b}"] = {"diff": round(100 * d.mean(), 2), "ci95_row": row_boot(d)}
d = models["hgb"].mean(0) - models["hgb_noresult"].mean(0); R["curated_test"]["hgb_minus_hgb_noresult"] = {"diff": round(100 * d.mean(), 2), "ci95_row": row_boot(d)}
# Hermes: flat per-row via saved checkpoints (same procedure as eval_hermes_ood.py)
ex = load_split("hermes_ood", "test"); torch.set_num_threads(2); flat = []
for dd in sorted(glob.glob(str(DATA / "stage0_runs/*_B_flat_curated_s*"))):
    ck = torch.load(dd + "/best.pt", map_location="cpu"); a = ck["args"]; X, y = pack(a["arch"], ex); X = torch.from_numpy(X)
    m = build(a["arch"], X, a["hidden"], a["nca_steps"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad(): lg = torch.cat([m(X[i:i + 1024]) for i in range(0, len(X), 1024)])
    flat.append((lg.argmax(1).numpy() == y).astype(float))
flat = np.stack(flat); assert (np.array(y) == gold["hermes"]).all(), "label order mismatch"
term = (gold["hermes"] == LABELS.index("terminal")).astype(float)
R["hermes"]["n_rows"] = int(len(term)); R["hermes"]["n_sessions"] = int(len(us)); R["hermes"]["floor"] = round(100 * term.mean(), 2)
R["hermes"]["B_flat"] = dict(zip(("top1_mean", "top1_sd"), ms(flat)))
for a in ("hgb", "hgb_noresult", "logreg"):
    C = acc(a, "hermes"); mc = C.mean(0); R["hermes"][a] = dict(zip(("top1_mean", "top1_sd"), ms(C)))
    R["hermes"][a]["minus_floor"] = {"diff": round(100 * (mc - term).mean(), 2), "ci95_session": sess_boot(mc - term)}
    R["hermes"][a]["minus_flat"] = {"diff": round(100 * (mc - flat.mean(0)).mean(), 2), "ci95_session": sess_boot(mc - flat.mean(0))}
(O / "sota_followup_summary.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
