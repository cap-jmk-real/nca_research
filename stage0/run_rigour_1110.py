#!/usr/bin/env python3
"""Rigour pass for the tuned comparison (2026-11-10, second half).

What this adds to stage0/run_hpo_1110.py (whose 20-trial seed-0 search logs are reused unchanged):
  (a) Multi-seed selection. For every model x benchmark the TOPK=5 trials with the highest seed-0
      validation top-1 are refitted with seeds 1 and 2 on TRAIN and scored on VALIDATION. The
      configuration with the highest 3-seed mean validation top-1 (ties: lower trial id) is selected.
      Test labels are never read before selection.
  (b) Five-seed finals (seeds 0..4) of the selected configuration, evaluated once on TEST. Per test
      turn we save correctness, top-1 prediction, full class probabilities (float16) and, for the
      GNCA/GNN models, the escalation step-norm term. Models trained on Hermes (curated) are also
      scored on the out-of-distribution split hermes_ood (held-out private sessions, never used for
      training, selection or earlier tuning).
  (c) For Hermes seeds 0..2 the fitted models + featuriser are pickled to ART (engine only, not git)
      for stage0/run_latency_1110.py (end-to-end CPU latency incl. featurisation).
  (e) New baselines under the SAME protocol and budget (20 trials incl. default trial 0, seed-0 search,
      then (a)+(b)): 'transformer' (pre-LN Transformer encoder over the 5 cells), 'gnn_untied'
      (the same perception/gated update on the same tree graph but with separate weights per step,
      i.e. a message-passing GNN without the NCA weight tying), 'knn' (cosine/euclidean kNN retrieval
      of the most similar training turns, feature set chosen by the search).
Resumable: every fit writes one JSON line (selection) or one file pair (final); reruns skip them.

Usage: run_rigour_1110.py --worker cpu|gpu [--phase base|new] [--bench ...]
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, pickle, shutil, sys, time, zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run_hpo_1110 as H  # noqa: E402

BENCHES = H.BENCHES
SEEDS_SEL = (1, 2)
SEEDS_FINAL = (0, 1, 2, 3, 4)
ART_SEEDS = (0, 1, 2)
TOPK = 5
N_TRIALS = 20
SMOKE = os.environ.get("RIG_SMOKE") == "1"  # wiring test only: 2 new-model trials, TOPK 1, 1 seed each, 50 steps
if SMOKE:
    SEEDS_SEL, SEEDS_FINAL, ART_SEEDS, TOPK = (1,), (0,), (0,), 1
NEW_CPU = ("knn",)
NEW_GPU = ("transformer", "gnn_untied")
ALL_CPU = H.CPU_MODELS + NEW_CPU
ALL_GPU = H.GPU_MODELS + NEW_GPU
OUT = Path(os.environ.get("RIG_OUT", HERE / "stage0_results" / "rigour_1110"))
HPO = Path(os.environ.get("HPO_OUT", H.OUT))
ART = Path(os.environ.get("RIG_ART", OUT / "artifacts"))


# ----------------------------------------------------------------------------- new models
def new_default(model, bench):
    steps = 3000 if bench == "curated" else 800
    neural = {"lr": 2e-3, "weight_decay": 0.05, "label_smoothing": 0.05, "batch_size": 512, "train_steps": steps}
    if model == "knn":
        return {"n_neighbors": 10, "weights": "distance", "metric": "cosine", "feat": "tab"}
    if model == "transformer":
        return {**neural, "d_model": 64, "layers": 2, "heads": 4, "ff_mult": 2, "dropout": 0.1}
    return {**neural, "K": 4, "hidden_c": 16, "update_hidden": 96}  # gnn_untied: GNCA defaults


def new_sample(model, bench, rng):
    c = H.sample_config("gnca_tree", bench, rng)  # same neural ranges as every other neural model
    if model == "knn":
        return {"n_neighbors": H._choice(rng, [1, 3, 5, 10, 20, 50, 100]),
                "weights": H._choice(rng, ["uniform", "distance"]),
                "metric": H._choice(rng, ["cosine", "euclidean"]),
                "feat": H._choice(rng, ["tab", "cell"])}
    if model == "transformer":
        base = {k: c[k] for k in ("lr", "weight_decay", "label_smoothing", "batch_size", "train_steps")}
        return {**base, "d_model": H._choice(rng, [32, 64, 128]), "layers": H._choice(rng, [1, 2, 3]),
                "heads": H._choice(rng, [2, 4]), "ff_mult": H._choice(rng, [2, 4]),
                "dropout": H._choice(rng, [0.0, 0.1, 0.2])}
    return c


def trial_configs(model, bench):
    if model in NEW_CPU + NEW_GPU:
        rng = np.random.default_rng(H.SEARCH_SEED + zlib.crc32(f"{bench}/{model}".encode()))
        n = 2 if SMOKE else N_TRIALS
        return [new_default(model, bench)] + [new_sample(model, bench, rng) for _ in range(n - 1)]
    return H.trial_configs(model, bench, N_TRIALS)


def build_net(model, cfg, obs_c):
    import torch.nn as nn
    r = H.R()
    if model == "mlp":
        net = r.BaselineMLP(obs_c * r.SEQ_LEN, r.N_CLASSES, hidden_dim=cfg["hidden"]); net.net[3].p = cfg["dropout"]
        return net
    if model == "transformer":
        return TinyTransformer(obs_c, r.N_CLASSES, cfg["d_model"], cfg["layers"], cfg["heads"], cfg["ff_mult"], cfg["dropout"], r.SEQ_LEN)
    if model == "gnn_untied":
        return UntiedGNN(obs_c=obs_c, hidden_c=cfg["hidden_c"], n_classes=r.N_CLASSES, steps=cfg["K"], seq_len=r.SEQ_LEN,
                         topology="tree", update_hidden=cfg["update_hidden"])
    return r.GraphNCA(obs_c=obs_c, hidden_c=cfg["hidden_c"], n_classes=r.N_CLASSES, steps=cfg["K"], seq_len=r.SEQ_LEN,
                      topology=H.TOPO[model], update_hidden=cfg["update_hidden"])


def _torch_classes():
    import torch, torch.nn as nn
    from nca_text.model import GraphNCA

    class _TinyTransformer(nn.Module):
        """Pre-LN Transformer encoder over the 5 cells (tokens), learned positions, flattened readout."""
        def __init__(self, obs_c, n_classes, d, layers, heads, ff_mult, dropout, seq_len):
            super().__init__()
            self.inp = nn.Linear(obs_c, d)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d)); nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, heads, ff_mult * d, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d); self.head = nn.Linear(d * seq_len, n_classes)

        def forward(self, obs):
            x = self.inp(obs.permute(0, 2, 1)) + self.pos
            return self.head(self.norm(self.enc(x)).flatten(1))

    class _UntiedGNN(GraphNCA):
        """Same perception (x, Ax, Ax-x) and gated residual as the GNCA, same graph, but step t has its
        own (W1_t, W2_t): a K-layer message-passing GNN. Only the weight tying differs from the GNCA."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            w1s, w2s = [self.w1], [self.w2]
            for _ in range(self.steps - 1):
                w1 = nn.Conv1d(self.w1.in_channels, self.w1.out_channels, 1)
                w2 = nn.Conv1d(self.w2.in_channels, self.w2.out_channels, 1, bias=False); nn.init.zeros_(w2.weight)
                w1s.append(w1); w2s.append(w2)
            del self.w1, self.w2
            self.w1s, self.w2s = nn.ModuleList(w1s), nn.ModuleList(w2s)

        def evolve(self, obs, hidden=None, steps=None):
            b, _, length = obs.shape
            if hidden is None:
                hidden = obs.new_zeros(b, self.hidden_c, length)
            traj = [hidden]
            for t in range(self.steps):
                h = hidden + self.step_embed[t].view(1, -1, 1)
                feat = torch.relu(self.w1s[t](self._perceive(obs, h)))
                delta, gate = self.w2s[t](feat).chunk(2, dim=1)
                hidden = hidden + torch.sigmoid(gate) * delta
                traj.append(hidden)
            return hidden, traj
    return _TinyTransformer, _UntiedGNN


TinyTransformer = UntiedGNN = None


def _ensure_torch_classes():
    global TinyTransformer, UntiedGNN
    if TinyTransformer is None:
        TinyTransformer, UntiedGNN = _torch_classes()


# ----------------------------------------------------------------------------- data (adds hermes_ood)
class Data(H.Data):
    def __init__(self, bench, need_cell, need_tab, device=None):
        super().__init__(bench, need_cell, need_tab, device)
        r = H.R()
        self.ood = None
        if bench == "curated":
            ex = [e for e in r.load_split("hermes_ood", "test") if e.label in r.L2I]
            self.ood = ex
            self.y["ood"] = np.array([r.L2I[e.label] for e in ex])
        self._gt = {}

    def feats(self, seed):
        if seed not in self._cache:
            r = H.R()
            np.random.seed(seed)
            gt = r.fit_tfidf_svd(self.sp["train"], seed)
            sp = dict(self.sp)
            if self.ood is not None:
                sp["ood"] = self.ood
            f = {"tab": {s: r.tab_features(gt, sp[s]) for s in sp}}
            if self.need_cell:
                f["cell"] = {s: r.pack_aligned_trace(sp[s], r.L2I, gt)[0].astype(np.float32) for s in sp}
            self._cache = {seed: f}
            self._gt = {seed: gt}
        return self._cache[seed]


def full_proba(clf, P):
    r = H.R()
    out = np.zeros((P.shape[0], r.N_CLASSES), np.float32)
    out[:, clf.classes_] = P
    return out


def X_for(model, cfg, f, splits):
    if model in ("logreg", "rf", "hgb") or (model == "knn" and cfg["feat"] == "tab"):
        return {s: f["tab"][s] for s in splits}
    if model in ("rf_cell", "hgb_cell", "knn"):
        return {s: f["cell"][s].reshape(f["cell"][s].shape[0], -1) for s in splits}
    return {s: f["cell"][s] for s in splits}


def fit(model, cfg, data, seed, final, device, n_jobs):
    """Fit on train. final=False: return val_top1. final=True: also return test (and ood) outputs."""
    if SMOKE:
        cfg = {**cfg, **({"train_steps": 50} if "train_steps" in cfg else {}), **({"n_estimators": 5} if "n_estimators" in cfg else {}),
               **({"max_iter": 10} if model.startswith("hgb") else {})}
    f = data.feats(seed)
    splits = ["train", "val"] + (["test"] + (["ood"] if data.ood is not None else []) if final else [])
    X = X_for(model, cfg, f, splits)
    y = data.y
    t0 = time.time()
    out = {}
    if model in ALL_CPU:
        import warnings
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        if model == "knn":
            clf = KNeighborsClassifier(n_neighbors=cfg["n_neighbors"], weights=cfg["weights"], metric=cfg["metric"],
                                       algorithm="brute", n_jobs=n_jobs)
        elif model == "logreg":
            clf = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=cfg["max_iter"], random_state=seed)
        elif model.startswith("rf"):
            clf = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                                         max_features=cfg["max_features"], class_weight=cfg["class_weight"], random_state=seed, n_jobs=n_jobs)
        else:
            clf = HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"], max_leaf_nodes=cfg["max_leaf_nodes"],
                                                 min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=cfg["l2_regularization"],
                                                 class_weight=cfg["class_weight"], early_stopping=False, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X["train"], y["train"])
            out["fit_s"] = round(time.time() - t0, 1)
            P = {s: full_proba(clf, clf.predict_proba(X[s])) for s in splits if s != "train"}
        obj = clf
    else:
        import torch
        import torch.nn.functional as F
        _ensure_torch_classes()
        r = H.R()
        Xs = {s: torch.from_numpy(X[s]).to(device) for s in splits}
        torch.manual_seed(seed)
        net = build_net(model, cfg, Xs["train"].shape[1]).to(device)
        te = "test" if final else "val"
        res = r.train_eval_neural(net, Xs["train"], torch.from_numpy(y["train"]).to(device), Xs["val"], y["val"], Xs[te], y[te],
                                  data.labels_all, steps=cfg["train_steps"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"], label_smoothing=cfg["label_smoothing"], eval_every=100, seed=seed)
        out["fit_s"] = round(time.time() - t0, 1); out["best_step"] = res["best_step"]; out["n_params"] = int(res["n_params"])
        net.eval(); P = {}; SN = {}
        with torch.no_grad():
            for s in splits:
                if s == "train":
                    continue
                lg, sn = [], []
                for i in range(0, len(Xs[s]), 4096):
                    xb = Xs[s][i:i + 4096]
                    if hasattr(net, "evolve"):
                        st, traj = net.evolve(xb)
                        lg.append(net.tool_head(st.flatten(1)))
                        node = torch.linalg.vector_norm(traj[-1] - traj[-2], ord=2, dim=1)
                        sn.append((node.sum(-1) / (r.SEQ_LEN * math.sqrt(net.hidden_c))))
                    else:
                        lg.append(net(xb))
                P[s] = F.softmax(torch.cat(lg), -1).float().cpu().numpy()
                if sn:
                    SN[s] = torch.cat(sn).cpu().numpy()
        out["_sn"] = SN
        obj = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        del net, Xs; torch.cuda.empty_cache()
    out["val_top1"] = round(100.0 * float((P["val"].argmax(1) == y["val"]).mean()), 2)
    out["_P"] = P; out["_obj"] = obj
    return out


# ----------------------------------------------------------------------------- driver
def read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append(p, rec):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def trials_path(bench, model):
    return (HPO if model in H.CPU_MODELS + H.GPU_MODELS else OUT) / "trials" / bench / f"{model}.jsonl"


def search(models, bench, data, device, n_jobs):
    for model in models:
        if model in H.CPU_MODELS + H.GPU_MODELS:
            continue  # reuse run_hpo_1110 logs
        tp = trials_path(bench, model)
        done = {t["trial"] for t in read_jsonl(tp)}
        for i, cfg in enumerate(trial_configs(model, bench)):
            if i in done:
                continue
            t0 = time.time()
            try:
                res = fit(model, cfg, data, 0, False, device, n_jobs)
                rec = {"trial": i, "config": cfg, "val_top1": res["val_top1"], "fit_s": res["fit_s"],
                       **({"n_params": res["n_params"]} if "n_params" in res else {})}
            except Exception as e:
                rec = {"trial": i, "config": cfg, "val_top1": -1.0, "error": repr(e)[:300]}
            rec["wall_s"] = round(time.time() - t0, 1); rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            append(tp, rec); log(f"SEARCH {bench}/{model} trial {i} val={rec['val_top1']} ({rec['wall_s']}s)")


def topk(bench, model):
    n = 2 if (SMOKE and model in NEW_CPU + NEW_GPU) else N_TRIALS
    T = sorted(read_jsonl(trials_path(bench, model)), key=lambda t: t["trial"])[:n]
    assert len(T) == n, (bench, model, len(T))
    return sorted(T, key=lambda t: (-t["val_top1"], t["trial"]))[:TOPK]


def select(models, bench, data, device, n_jobs):
    for seed in SEEDS_SEL:
        for model in models:
            sp = OUT / "select" / bench / f"{model}.jsonl"
            done = {(r["trial"], r["seed"]) for r in read_jsonl(sp)}
            for t in topk(bench, model):
                if (t["trial"], seed) in done:
                    continue
                t0 = time.time()
                try:
                    v = fit(model, t["config"], data, seed, False, device, n_jobs)["val_top1"]
                except Exception as e:
                    v = -1.0; log(f"ERR {bench}/{model} t{t['trial']} s{seed}: {e!r}")
                append(sp, {"trial": t["trial"], "seed": seed, "val_top1": v, "wall_s": round(time.time() - t0, 1)})
                log(f"SELECT {bench}/{model} trial {t['trial']} seed {seed} val={v}")


def chosen(bench, model):
    sel = read_jsonl(OUT / "select" / bench / f"{model}.jsonl")
    rows = []
    for t in topk(bench, model):
        vs = [t["val_top1"]] + [r["val_top1"] for r in sel if r["trial"] == t["trial"]]
        assert len(vs) == 1 + len(SEEDS_SEL), (bench, model, t["trial"], vs)
        rows.append({"trial": t["trial"], "config": t["config"], "val_seed0": t["val_top1"], "val_mean3": float(np.mean(vs)), "vals": vs})
    best = max(rows, key=lambda x: (round(x["val_mean3"], 6), -x["trial"]))
    return best, rows


def sid_hash(s):
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def finals(models, bench, data, device, n_jobs):
    md = OUT / "meta"; md.mkdir(parents=True, exist_ok=True)
    mp = md / f"{bench}.json"
    if not mp.exists():
        m = {"test_session": [sid_hash(e.session_id) for e in data.sp["test"]], "y_test": data.y["test"].tolist()}
        if data.ood is not None:
            m["ood_session"] = [sid_hash(e.session_id) for e in data.ood]; m["y_ood"] = data.y["ood"].tolist()
        mp.write_text(json.dumps(m))
    for seed in SEEDS_FINAL:
        for model in models:
            fd = OUT / "final" / bench / model; fd.mkdir(parents=True, exist_ok=True)
            if (fd / f"seed{seed}.json").exists():
                continue
            best, rows = chosen(bench, model)
            t0 = time.time()
            res = fit(model, best["config"], data, seed, True, device, n_jobs)
            P, SN, obj = res.pop("_P"), res.pop("_sn", {}), res.pop("_obj")
            arrs = {}
            for s in [k for k in ("test", "ood") if k in P]:
                arrs[f"{s}_proba"] = P[s].astype(np.float16)
                arrs[f"{s}_pred"] = P[s].argmax(1).astype(np.int16)
                arrs[f"{s}_correct"] = (P[s].argmax(1) == data.y[s]).astype(np.int8)
                if s in SN:
                    arrs[f"{s}_stepnorm"] = SN[s].astype(np.float32)
            np.savez_compressed(fd / f"seed{seed}.npz", **arrs)
            rec = {"seed": seed, "selected_trial": best["trial"], "config": best["config"], "selection": rows, **res,
                   "test_top1": round(100.0 * float(arrs["test_correct"].mean()), 2), "n_test": int(arrs["test_correct"].size),
                   "wall_s": round(time.time() - t0, 1)}
            if "ood_correct" in arrs:
                rec["ood_top1"] = round(100.0 * float(arrs["ood_correct"].mean()), 2); rec["n_ood"] = int(arrs["ood_correct"].size)
            if bench == "curated" and seed in ART_SEEDS and shutil.disk_usage(str(Path.home())).free > 6e9:
                ad = ART / model; ad.mkdir(parents=True, exist_ok=True)
                with open(ad / f"seed{seed}.pkl", "wb") as fh:
                    pickle.dump({"model": model, "config": best["config"], "obj": obj, "featuriser": data._gt[seed],
                                 "obs_c": int(data.feats(seed)["cell"]["train"].shape[1]) if data.need_cell else None}, fh)
            (fd / f"seed{seed}.json").write_text(json.dumps(rec, indent=1, default=str))
            log(f"FINAL {bench}/{model} seed {seed} test={rec['test_top1']} ood={rec.get('ood_top1')} sel_trial={best['trial']}")


def worker(kind, phase, benches, n_jobs):
    import torch
    device = torch.device("cuda" if (kind == "gpu" and torch.cuda.is_available()) else "cpu")
    if kind == "gpu":
        torch.set_num_threads(4)
    if phase == "base":
        models = H.CPU_MODELS if kind == "cpu" else H.GPU_MODELS
    else:
        models = NEW_CPU if kind == "cpu" else NEW_GPU
    for bench in benches:
        data = Data(bench, need_cell=True, need_tab=True, device=device)
        log(f"== {kind}/{phase} {bench} models={models}")
        search(models, bench, data, device, n_jobs)
        select(models, bench, data, device, n_jobs)
        finals(models, bench, data, device, n_jobs)
        del data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--phase", choices=("base", "new"), default="base")
    ap.add_argument("--bench", default=",".join(BENCHES))
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    worker(a.worker, a.phase, tuple(a.bench.split(",")), a.n_jobs)


if __name__ == "__main__":
    main()
