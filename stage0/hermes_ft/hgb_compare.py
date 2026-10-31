"""Paired session-bootstrap: Hermes fine-tune/scratch models vs HistGB (zero-shot, curated-trained) on the drop variant."""
import json, sys, glob, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_ft import OUT, load
import train_eval as te
base = json.load(open(Path(te.__file__).with_name("labels.json")))
R = load("drop"); y = np.array([base.index(r["target_tool"]) for r in R]); sess = np.array([r["session_id"] for r in R])
S = OUT / "summary.json"; summ = json.load(open(S)); sel = summ["selection"]
rp = [json.load(open(p)) for p in sorted(glob.glob(str(OUT.parent / "sota_baselines_20261026/rowpreds_v2_s*.json")))]
assert list(rp[0]["hermes_session"]) == list(sess), "row order mismatch"
def corr(v):
    v = np.array(v)
    return v.astype(float) if set(np.unique(v)) <= {0, 1} and v.dtype != object and len(base) > 2 and not (v > 1).any() and np.array_equal(v, v.astype(bool)) and False else (v == y).astype(float)
gold = np.array(rp[0]["gold"]["hermes"]); 
if gold.dtype.kind in "U": gold = np.array([base.index(g) for g in gold])
assert np.array_equal(gold, y), "gold mismatch"
H = np.stack([(np.array([base.index(p) if isinstance(p, str) else p for p in r["hgb"]["hermes"]]) == y).astype(float) for r in rp])
runs = {json.load(open(p))["name"]: json.load(open(p)) for p in glob.glob(str(OUT / "runs/*_drop_*.json"))}
us, inv = np.unique(sess, return_inverse=True); mem = [np.where(inv == i)[0] for i in range(len(us))]; rng = np.random.default_rng(20261026)
idx = [rng.integers(0, len(us), len(us)) for _ in range(10000)]
def ci(x): bs = [np.concatenate([x[mem[i]] for i in b]).mean() for b in idx]; return [round(100*np.percentile(bs, 2.5), 2), round(100*np.percentile(bs, 97.5), 2)]
out = {"hgb_hermes_top1_mean": round(100*H.mean(1).mean(), 2), "hgb_hermes_top1_sd": round(100*H.mean(1).std(ddof=1), 2), "n_seeds_hgb": len(rp), "models": {}}
for key, s in sel.items():
    m, st, v = key.split("/")
    if v != "drop": continue
    lr = s["chosen_lr"]; A = []
    for sd in range(3):
        a = np.full(len(y), np.nan)
        for k in range(5):
            r = runs[f"{m}_{st}_drop_lr{lr}_s{sd}_f{k}"]; a[r["test_rows"]] = np.array(r["test_pred"]) == y[r["test_rows"]]
        A.append(a)
    A = np.stack(A); d = A.mean(0) - H.mean(0)
    out["models"][f"{m}/{st}"] = {"lr": lr, "minus_hgb": round(100*d.mean(), 2), "minus_hgb_ci": ci(d)}
summ["hgb_paired_drop"] = out; S.write_text(json.dumps(summ, indent=1)); print(json.dumps(out, indent=0))
