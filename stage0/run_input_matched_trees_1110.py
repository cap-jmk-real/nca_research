"""Input-matched tree baselines (2026-11-10).

Fits Random Forest and HistGradientBoosting on exactly the packed cell features
(5 cells x 146 channels, flattened) that the MLP and the GNCA receive, with the same
fixed hyperparameters as run_reviewer_experiments.run_benchmark_split, seeds 0-2,
on all four benchmark splits. Test order equals the order in correct_by_seed.npz,
so per-example correctness can be paired with the GNCA results.

Outputs: stage0/stage0_results/review_1110/{input_matched_trees.json, correct_by_seed_trees.npz}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_reviewer_experiments as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "stage0_results" / "review_1110"
BENCHES = ("curated", "bfcl_e1v2", "toolbench", "apibank")
SEEDS = (0, 1, 2)


def flat(examples, gt):
    obs, _ = R.pack_aligned_trace(examples, R.L2I, gt)
    return obs.reshape(obs.shape[0], -1)


def main() -> None:
    res: dict = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seeds": list(SEEDS), "benchmarks": {}}
    corr: dict[str, np.ndarray] = {}
    t_all = time.time()
    for b in BENCHES:
        sp = {s: R.load_split(b, s) for s in ("train", "val", "test")}
        y = {s: np.array([R.L2I[e.label] for e in sp[s]]) for s in sp}
        per = {"RandomForest_packed": [], "HistGradientBoosting_packed": []}
        cs = {k: [] for k in per}
        for seed in SEEDS:
            np.random.seed(seed)
            gt = R.fit_tfidf_svd(sp["train"], seed)
            X = {s: flat(sp[s], gt) for s in sp}
            models = {
                "RandomForest_packed": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=seed, n_jobs=-1),
                "HistGradientBoosting_packed": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=seed),
            }
            for name, clf in models.items():
                t0 = time.time()
                clf.fit(X["train"], y["train"])
                pred = R.eu.rank_from_proba(clf.predict_proba(X["test"]), clf.classes_)[:, 0]
                pv = R.eu.rank_from_proba(clf.predict_proba(X["val"]), clf.classes_)[:, 0]
                c = pred == y["test"]
                per[name].append({"seed": seed, "test_top1": round(100 * float(c.mean()), 2),
                                  "val_top1": round(100 * float((pv == y["val"]).mean()), 2),
                                  "n_features": int(X["train"].shape[1]), "fit_s": round(time.time() - t0, 1)})
                cs[name].append(c)
                print(b, name, seed, per[name][-1], flush=True)
        res["benchmarks"][b] = {"n_test": int(len(y["test"])), "models": {
            k: {"runs": v, "test_top1_mean": round(float(np.mean([r["test_top1"] for r in v])), 2),
                "test_top1_sd": round(float(np.std([r["test_top1"] for r in v], ddof=1)), 2)} for k, v in per.items()}}
        for k, v in cs.items():
            corr[f"{b}/{k}"] = np.stack(v)
    res["elapsed_s"] = round(time.time() - t_all, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_matched_trees.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / "correct_by_seed_trees.npz", **corr)
    print("done", res["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()
