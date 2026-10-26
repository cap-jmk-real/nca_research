"""Classical baselines on the frozen curated split: HistGradientBoosting and
logistic regression on TF-IDF(goal)->SVD + positional one-hot of the last 4 tools.
Model selection on val only; test scored once. Writes result json per seed."""
import json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, metrics, DATA
LABELS = json.load(open(Path(__file__).resolve().parent / "labels.json")); L = {t: i for i, t in enumerate(LABELS)}
seed = int(sys.argv[1]); out = DATA / "stage0_results" / "sota_baselines_20261026"; out.mkdir(parents=True, exist_ok=True)
S = {s: load_split("curated", s) for s in ("train", "val", "test")}
def struct(ex):
    v = np.zeros(4 * (len(LABELS) + 1) + 6, dtype=np.float32)
    st = list(ex.recent_steps)[-4:] if hasattr(ex, "recent_steps") else []
    return v, st
def tools_of(ex):
    return list(ex.steps)
    for a in ("recent_steps", "recent", "steps", "history"):
        if hasattr(ex, a): return list(getattr(ex, a))
    return list(ex[3])
def goal_of(ex): return getattr(ex, "goal", None) if hasattr(ex, "goal") else ex[2]
def label_of(ex): return L[ex.label]
def feats(ex):
    st = tools_of(ex)[-4:]; K = len(LABELS) + 1
    v = np.zeros(4 * K + 4, dtype=np.float32)
    for j in range(4):
        k = len(st) - 1 - j  # j=0 most recent
        v[j * K + (L.get(st[k][0], len(LABELS)) if k >= 0 else len(LABELS))] = 1 if k >= 0 else 0
        if k < 0: v[j * K + len(LABELS)] = 1
    v[4 * K] = len(st); v[4 * K + 1] = np.log1p(len(str(st[-1][1]))) if st else 0
    v[4 * K + 2] = float(any("error" in str(r).lower() for _, r in st[-1:])); v[4 * K + 3] = np.log1p(len(goal_of(ex) or ""))
    return v
tf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
G = {s: [ (goal_of(e) or "") + " || " + " ".join(str(r)[:200] for _, r in tools_of(e)[-1:]) for e in S[s]] for s in S}
Xt = tf.fit_transform(G["train"]); svd = TruncatedSVD(128, random_state=seed).fit(Xt)
X = {s: np.hstack([svd.transform(tf.transform(G[s])), np.stack([feats(e) for e in S[s]])]) for s in S}
Y = {s: np.array([label_of(e) for e in S[s]]) for s in S}
def rank(P): return [list(np.argsort(-p)) for p in P]
res = {"seed": seed, "features": "tfidf(goal+last result 200ch) 1-2gram -> SVD128 + one-hot last-4 tools + 4 scalars", "split": "curated stage0 (session-level, seed 20261026)"}
# HGB: pick max_iter by val top1
best = None
for lr, it in [(0.1, 200), (0.1, 400), (0.05, 800)]:
    t = time.time(); m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=31, l2_regularization=1.0, max_features=0.8, early_stopping=False, random_state=seed).fit(X["train"], Y["train"])
    v = metrics(list(Y["val"]), rank(m.predict_proba(X["val"])), len(LABELS))["top1"]
    print("hgb", lr, it, v, round(time.time() - t), flush=True)
    if best is None or v > best[0]: best = (v, lr, it, m)
res["hgb"] = {"lr": best[1], "max_iter": best[2], "val": metrics(list(Y["val"]), rank(best[3].predict_proba(X["val"])), len(LABELS)), "test": metrics(list(Y["test"]), rank(best[3].predict_proba(X["test"])), len(LABELS))}
# sparse logistic regression on full tfidf + struct
Xs = {s: sp.hstack([tf.transform(G[s]), sp.csr_matrix(np.stack([feats(e) for e in S[s]]))]).tocsr() for s in S}
bl = None
for C in (1.0, 4.0, 16.0):
    m = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Xs["train"], Y["train"])
    v = metrics(list(Y["val"]), rank(m.predict_proba(Xs["val"])), len(LABELS))["top1"]; print("lr", C, v, flush=True)
    if bl is None or v > bl[0]: bl = (v, C, m)
res["logreg"] = {"C": bl[1], "val": metrics(list(Y["val"]), rank(bl[2].predict_proba(Xs["val"])), len(LABELS)), "test": metrics(list(Y["test"]), rank(bl[2].predict_proba(Xs["test"])), len(LABELS))}
(out / f"gbdt_s{seed}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
