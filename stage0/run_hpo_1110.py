#!/usr/bin/env python3
"""Per-benchmark hyperparameter search for every model (2026-11-10).

Protocol (identical for every model):
  * Random search with the SAME number of trials per model and benchmark (--trials, default 20).
    Trial 0 is always the untuned default configuration used in review_1110, so the search can
    only match or beat the default on validation. Trials 1..N-1 are drawn from the model's search
    space with numpy.random.default_rng(seed=SEARCH_SEED + crc32(benchmark/model)).
  * Every trial is fitted on the TRAIN split with seed 0 (features: TF-IDF/SVD fitted on train,
    seed 0) and scored by top-1 accuracy on the VALIDATION split. Neural models additionally keep
    their best-validation checkpoint inside a trial (same rule as before). Test labels are never
    read during the search.
  * The configuration with the highest validation top-1 (ties: lower trial id) is then refitted
    with seeds 0, 1, 2 and evaluated once on the TEST split. Per-example test correctness is saved
    in the same order as review_1110/correct_by_seed.npz for the seed-clustered bootstrap + Holm.

Models: logreg, rf, hgb (188 tabular features, as in review_1110), rf_cell, hgb_cell (730 cell
features), mlp, gnca_chain, gnca_star, gnca_tree, gnca_learned (5x146 cell tensor).

Resumability: each finished trial appends one JSON line to
  OUT/trials/<bench>/<model>.jsonl
and each finished final seed writes OUT/final/<bench>/<model>/seed<k>.json + .npy (correctness).
Re-running skips everything already on disk. No checkpoints are written (results only).

Usage:
  run_hpo_1110.py --worker cpu|gpu [--bench curated,bfcl_e1v2,toolbench,apibank] [--trials 20]
  run_hpo_1110.py --collect          # merge into OUT/hpo_summary.json + OUT/correct_by_seed_hpo.npz
  run_hpo_1110.py --smoke            # one tiny trial per model on apibank, into OUT/smoke/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BENCHES = ("toolbench", "apibank", "bfcl_e1v2", "curated")
SEEDS = (0, 1, 2)
SEARCH_SEED = 20261110
CPU_MODELS = ("logreg", "rf", "hgb", "rf_cell", "hgb_cell")
GPU_MODELS = ("mlp", "gnca_chain", "gnca_star", "gnca_tree", "gnca_learned")
TOPO = {"gnca_chain": "1d", "gnca_star": "star", "gnca_tree": "tree", "gnca_learned": "meta"}
OUT = Path(os.environ.get("HPO_OUT", HERE / "stage0_results" / "hpo_1110"))


# ----------------------------------------------------------------------------- search spaces
def _loguni(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _choice(rng, xs):
    return xs[int(rng.integers(len(xs)))]


def default_config(model: str, bench: str) -> dict:
    steps = 3000 if bench == "curated" else 800
    if model == "logreg":
        return {"C": 1.0, "class_weight": None, "max_iter": 300}
    if model in ("rf", "rf_cell"):
        return {"n_estimators": 50, "max_depth": 15, "min_samples_leaf": 1, "max_features": "sqrt",
                "class_weight": None}
    if model in ("hgb", "hgb_cell"):
        return {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                "l2_regularization": 0.0, "class_weight": None}
    neural = {"lr": 2e-3, "weight_decay": 0.05, "label_smoothing": 0.05, "batch_size": 512, "train_steps": steps}
    if model == "mlp":
        return {**neural, "hidden": 76, "dropout": 0.1}
    return {**neural, "K": 4, "hidden_c": 16, "update_hidden": 96}


def sample_config(model: str, bench: str, rng) -> dict:
    hermes = bench == "curated"
    if model == "logreg":
        return {"C": _loguni(rng, 1e-3, 1e2), "class_weight": _choice(rng, [None, "balanced"]), "max_iter": 1000}
    if model in ("rf", "rf_cell"):
        return {"n_estimators": _choice(rng, [50, 100, 200, 300]),
                "max_depth": _choice(rng, [10, 15, 20, 30, None]),
                "min_samples_leaf": _choice(rng, [1, 2, 4, 8]),
                "max_features": _choice(rng, ["sqrt", "log2", 0.1, 0.3]),
                "class_weight": _choice(rng, [None, "balanced_subsample"])}
    if model in ("hgb", "hgb_cell"):
        return {"learning_rate": _loguni(rng, 0.02, 0.3),
                "max_iter": _choice(rng, [100, 200, 400]),
                "max_leaf_nodes": _choice(rng, [15, 31, 63, 127]),
                "min_samples_leaf": _choice(rng, [5, 20, 50]),
                "l2_regularization": _choice(rng, [0.0, 1e-3, 1e-1, 1.0, 10.0]),
                "class_weight": _choice(rng, [None, "balanced"])}
    neural = {"lr": _loguni(rng, 3e-4, 6e-3),
              "weight_decay": _loguni(rng, 1e-3, 0.2),
              "label_smoothing": _choice(rng, [0.0, 0.05, 0.1]),
              "batch_size": _choice(rng, [256, 512, 1024]),
              "train_steps": _choice(rng, [1500, 3000, 6000] if hermes else [400, 800, 1600])}
    if model == "mlp":
        return {**neural, "hidden": _choice(rng, [38, 76, 152, 304]), "dropout": _choice(rng, [0.0, 0.1, 0.3])}
    return {**neural, "K": _choice(rng, [2, 4, 6, 8]), "hidden_c": _choice(rng, [8, 16, 32]),
            "update_hidden": _choice(rng, [48, 96, 192])}


def trial_configs(model: str, bench: str, n: int) -> list[dict]:
    rng = np.random.default_rng(SEARCH_SEED + zlib.crc32(f"{bench}/{model}".encode()))
    return [default_config(model, bench)] + [sample_config(model, bench, rng) for _ in range(n - 1)]


# ----------------------------------------------------------------------------- data
_R = None


def R():
    global _R
    if _R is None:
        import run_reviewer_experiments as mod
        _R = mod
    return _R


class Data:
    """Per-benchmark data with per-seed feature caches."""

    def __init__(self, bench: str, need_cell: bool, need_tab: bool, device=None):
        r = R()
        self.bench = bench
        self.sp = {s: r.load_split(bench, s) for s in ("train", "val", "test")}
        self.y = {s: np.array([r.L2I[e.label] for e in self.sp[s]]) for s in self.sp}
        self.labels_all = sorted(set(self.y["train"].tolist()) | set(self.y["test"].tolist()))
        self.need_cell, self.need_tab, self.device = need_cell, need_tab, device
        self._cache: dict[int, dict] = {}

    def feats(self, seed: int) -> dict:
        if seed not in self._cache:
            r = R()
            np.random.seed(seed)
            gt = r.fit_tfidf_svd(self.sp["train"], seed)
            f: dict = {}
            if self.need_tab:
                f["tab"] = {s: r.tab_features(gt, self.sp[s]) for s in self.sp}
            if self.need_cell:
                f["cell"] = {s: r.pack_aligned_trace(self.sp[s], r.L2I, gt)[0].astype(np.float32) for s in self.sp}
            self._cache = {seed: f}  # keep one seed in memory
        return self._cache[seed]


# ----------------------------------------------------------------------------- fitting
def fit_sklearn(model: str, cfg: dict, X: dict, y: dict, seed: int, n_jobs: int, eval_test: bool):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    r = R()
    if model == "logreg":
        clf = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=cfg["max_iter"], random_state=seed)
    elif model.startswith("rf"):
        clf = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                     min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                                     class_weight=cfg["class_weight"], random_state=seed, n_jobs=n_jobs)
    else:
        clf = HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"],
                                             max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=cfg["min_samples_leaf"],
                                             l2_regularization=cfg["l2_regularization"], class_weight=cfg["class_weight"],
                                             early_stopping=False, random_state=seed)
    t0 = time.time()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X["train"], y["train"])
    fit_s = time.time() - t0
    vr = r.eu.rank_from_proba(clf.predict_proba(X["val"]), clf.classes_)
    out = {"val_top1": round(100.0 * float((vr[:, 0] == y["val"]).mean()), 2), "fit_s": round(fit_s, 1)}
    if model.startswith("rf"):
        out["n_tree_nodes"] = int(sum(e.tree_.node_count for e in clf.estimators_))
        out["n_params"] = int(sum(e.tree_.node_count * (2 + clf.n_classes_) for e in clf.estimators_))
    elif model.startswith("hgb"):
        nodes = int(sum(t.nodes.shape[0] for itr in clf._predictors for t in itr))
        out["n_tree_nodes"], out["n_params"] = nodes, nodes * 5
    else:
        out["n_params"] = int(clf.coef_.size + clf.intercept_.size)
    if eval_test:
        tr = r.eu.rank_from_proba(clf.predict_proba(X["test"]), clf.classes_)
        out["ranked0"] = tr[:, 0]
    return out


def fit_neural(model: str, cfg: dict, X: dict, y: dict, seed: int, labels_all, device, eval_test: bool):
    import torch
    r = R()
    Xs = {s: torch.from_numpy(X[s]).to(device) for s in X}
    ytr = torch.from_numpy(y["train"]).to(device)
    obs_c = Xs["train"].shape[1]
    torch.manual_seed(seed)
    if model == "mlp":
        net = r.BaselineMLP(obs_c * r.SEQ_LEN, r.N_CLASSES, hidden_dim=cfg["hidden"])
        net.net[3].p = cfg["dropout"]
    else:
        net = r.GraphNCA(obs_c=obs_c, hidden_c=cfg["hidden_c"], n_classes=r.N_CLASSES, steps=cfg["K"],
                         seq_len=r.SEQ_LEN, topology=TOPO[model], update_hidden=cfg["update_hidden"])
    net = net.to(device)
    t0 = time.time()
    # The test tensor passed here is only used after training, to score the best-val checkpoint;
    # during the search we pass the validation split in its place so test is never touched.
    te_X, te_y = (Xs["test"], y["test"]) if eval_test else (Xs["val"], y["val"])
    res = r.train_eval_neural(net, Xs["train"], ytr, Xs["val"], y["val"], te_X, te_y, labels_all,
                              steps=cfg["train_steps"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                              weight_decay=cfg["weight_decay"], label_smoothing=cfg["label_smoothing"],
                              eval_every=100, seed=seed)
    out = {"val_top1": res["val_top1"], "best_step": res["best_step"], "fit_s": round(time.time() - t0, 1),
           "n_params": int(res["n_params"])}
    if eval_test:
        out["ranked0"] = res["ranked"][:, 0]
    del net, Xs
    torch.cuda.empty_cache()
    return out


def run_one(model, cfg, data: Data, seed, eval_test, device, n_jobs):
    f = data.feats(seed)
    if model in ("logreg", "rf", "hgb"):
        X = f["tab"]
    elif model in ("rf_cell", "hgb_cell"):
        X = {s: v.reshape(v.shape[0], -1) for s, v in f["cell"].items()}
    else:
        X = f["cell"]
    if not eval_test:
        X = {k: v for k, v in X.items() if k != "test"}
    if model in CPU_MODELS:
        return fit_sklearn(model, cfg, X, data.y, seed, n_jobs, eval_test)
    return fit_neural(model, cfg, X, data.y, seed, data.labels_all, device, eval_test)


# ----------------------------------------------------------------------------- driver
def read_trials(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def worker(models, benches, n_trials, n_jobs, out: Path, final: bool = True):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if any(m in GPU_MODELS for m in models):
        torch.set_num_threads(4)
    for bench in benches:
        data = Data(bench, need_cell=any(m != "logreg" and m not in ("rf", "hgb") for m in models),
                    need_tab=any(m in ("logreg", "rf", "hgb") for m in models), device=device)
        for model in models:
            tp = out / "trials" / bench / f"{model}.jsonl"
            tp.parent.mkdir(parents=True, exist_ok=True)
            done = {t["trial"] for t in read_trials(tp)}
            for i, cfg in enumerate(trial_configs(model, bench, n_trials)):
                if i in done:
                    continue
                t0 = time.time()
                try:
                    res = run_one(model, cfg, data, 0, False, device, n_jobs)
                    rec = {"trial": i, "config": cfg, **res}
                except Exception as e:  # a failed trial counts as a trial with no score
                    rec = {"trial": i, "config": cfg, "val_top1": -1.0, "error": repr(e)[:300]}
                rec["wall_s"] = round(time.time() - t0, 1)
                rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                with tp.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"[{time.strftime('%H:%M:%S')}] {bench}/{model} trial {i} val={rec['val_top1']} ({rec['wall_s']}s)", flush=True)
        if not final:
            continue
        best = {}
        for model in models:
            trials = sorted(read_trials(out / "trials" / bench / f"{model}.jsonl"), key=lambda t: t["trial"])[:n_trials]
            best[model] = max(trials, key=lambda t: (t["val_top1"], -t["trial"]))
        for seed in SEEDS:  # seed-outer so features are built once per seed
            for model in models:
                fd = out / "final" / bench / model
                fd.mkdir(parents=True, exist_ok=True)
                if (fd / f"seed{seed}.json").exists():
                    continue
                t0 = time.time()
                res = run_one(model, best[model]["config"], data, seed, True, device, n_jobs)
                pred = res.pop("ranked0")
                corr = (pred == data.y["test"]).astype(np.int8)
                np.save(fd / f"seed{seed}.npy", corr)
                rec = {"seed": seed, "best_trial": best[model]["trial"], "config": best[model]["config"], **res,
                       "test_top1": round(100.0 * float(corr.mean()), 2), "n_test": int(corr.size),
                       "wall_s": round(time.time() - t0, 1)}
                (fd / f"seed{seed}.json").write_text(json.dumps(rec, indent=1))
                print(f"[{time.strftime('%H:%M:%S')}] FINAL {bench}/{model} seed {seed} test={rec['test_top1']} val={rec['val_top1']}", flush=True)


def collect(out: Path, n_trials: int):
    summ = {"protocol": __doc__, "trials_per_model": n_trials, "seeds": list(SEEDS), "benchmarks": {}}
    corr = {}
    for bench in BENCHES:
        b = {}
        for model in CPU_MODELS + GPU_MODELS:
            trials = read_trials(out / "trials" / bench / f"{model}.jsonl")
            fd = out / "final" / bench / model
            finals = [json.loads((fd / f"seed{s}.json").read_text()) for s in SEEDS if (fd / f"seed{s}.json").exists()]
            if not trials:
                continue
            tops = [f["test_top1"] for f in finals]
            b[model] = {"n_trials": len(trials), "search_wall_s": round(sum(t.get("wall_s", 0) for t in trials), 1),
                        "default_val_top1": next((t["val_top1"] for t in trials if t["trial"] == 0), None),
                        "best_trial": finals[0]["best_trial"] if finals else None,
                        "best_config": finals[0]["config"] if finals else None,
                        "best_val_top1_seed0": max(t["val_top1"] for t in trials),
                        "final": finals,
                        "test_top1_mean": round(float(np.mean(tops)), 2) if tops else None,
                        "test_top1_sd": round(float(np.std(tops, ddof=1)), 2) if len(tops) > 1 else None}
            if len(finals) == len(SEEDS):
                corr[f"{bench}/{model}"] = np.stack([np.load(fd / f"seed{s}.npy") for s in SEEDS])
        summ["benchmarks"][bench] = b
    (out / "hpo_summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    np.savez_compressed(out / "correct_by_seed_hpo.npz", **corr)
    print("collected", len(corr), "model/benchmark pairs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=("cpu", "gpu"))
    ap.add_argument("--bench", default=",".join(BENCHES))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    benches = tuple(a.bench.split(","))
    if a.smoke:
        global sample_config
        so = OUT / "smoke"
        orig = sample_config

        def tiny(model, bench, rng):
            c = orig(model, bench, rng)
            if "train_steps" in c:
                c["train_steps"] = 100
            if "n_estimators" in c:
                c["n_estimators"] = 10
            if "max_iter" in c and model.startswith("hgb"):
                c["max_iter"] = 20
            return c
        sample_config = tiny
        t0 = time.time()
        worker(CPU_MODELS + GPU_MODELS, ("apibank",), 2, a.n_jobs, so, final=False)
        # drop trial 0 (full default) timing from smoke: trial 1 is the tiny one
        print(f"SMOKE OK {time.time() - t0:.1f}s", flush=True)
        return
    if a.collect:
        collect(OUT, a.trials)
        return
    worker(CPU_MODELS if a.worker == "cpu" else GPU_MODELS, benches, a.trials, a.n_jobs, OUT)


if __name__ == "__main__":
    main()
