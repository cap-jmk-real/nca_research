"""SOTA follow-up (26 Oct): per seed refit HGB (val-selected config from gbdt_s{seed}.json),
HGB without last-result text, and logreg (val-selected C); save per-row top-1 preds on curated test and hermes_ood."""
import json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, DATA
LABELS = json.load(open(Path(__file__).resolve().parent / "labels.json")); L = {t: i for i, t in enumerate(LABELS)}
seed = int(sys.argv[1]); out = DATA / "stage0_results" / "sota_baselines_20261026"
cfg = json.load(open(out / f"gbdt_s{seed}.json"))
S = {s: load_split("curated", s) for s in ("train", "val", "test")}; S["hermes"] = load_split("hermes_ood", "test")
def feats(ex):
    st = list(ex.steps)[-4:]; K = len(LABELS) + 1; v = np.zeros(4 * K + 4, dtype=np.float32)
    for j in range(4):
        k = len(st) - 1 - j
        if k >= 0: v[j * K + L.get(st[k][0], len(LABELS))] = 1
        else: v[j * K + len(LABELS)] = 1
    v[4 * K] = len(st); v[4 * K + 1] = np.log1p(len(str(st[-1][1]))) if st else 0
    v[4 * K + 2] = float(any("error" in str(r).lower() for _, r in st[-1:])); v[4 * K + 3] = np.log1p(len(ex.goal or ""))
    return v
F = {s: np.stack([feats(e) for e in S[s]]) for s in S}
Y = {s: np.array([L.get(e.label, -1) for e in S[s]]) for s in S}
def text(e, res): return (e.goal or "") + ((" || " + " ".join(str(r)[:200] for _, r in list(e.steps)[-1:])) if res else "")
preds = {}
for name, res in (("hgb", True), ("hgb_noresult", False)):
    tf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    G = {s: [text(e, res) for e in S[s]] for s in S}; Xt = tf.fit_transform(G["train"]); svd = TruncatedSVD(128, random_state=seed).fit(Xt)
    X = {s: np.hstack([svd.transform(tf.transform(G[s])), F[s]]) for s in S}
    m = HistGradientBoostingClassifier(learning_rate=cfg["hgb"]["lr"], max_iter=cfg["hgb"]["max_iter"], max_leaf_nodes=31, l2_regularization=1.0, max_features=0.8, early_stopping=False, random_state=seed).fit(X["train"], Y["train"])
    preds[name] = {s: m.predict(X[s]).tolist() for s in ("val", "test", "hermes")}; print(name, flush=True)
    if res:
        Xs = {s: sp.hstack([tf.transform(G[s]), sp.csr_matrix(F[s])]).tocsr() for s in S}
        m = LogisticRegression(C=cfg["logreg"]["C"], max_iter=2000, random_state=seed).fit(Xs["train"], Y["train"])
        preds["logreg"] = {s: m.predict(Xs[s]).tolist() for s in ("val", "test", "hermes")}; print("logreg", flush=True)
preds["gold"] = {s: Y[s].tolist() for s in ("val", "test", "hermes")}
preds["hermes_session"] = [e.session_id for e in S["hermes"]]
(out / f"rowpreds_v2_s{seed}.json").write_text(json.dumps(preds)); print("done", seed)
