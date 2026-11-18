#!/usr/bin/env python3
"""Head-to-head efficiency of every model in the main comparison (2026-11-18).

Models: the 13 rows of Table tab:rig-main (logreg, rf, hgb, rf_cell, hgb_cell, knn, mlp, transformer,
gnn_untied, gnca_chain, gnca_star, gnca_tree, gnca_learned). Benchmarks: curated (Hermes), bfcl_e1v2,
toolbench, apibank. Configuration: the SELECTED tuned configuration of the rigour pass, read verbatim
from stage0/stage0_results/rigour_1110/rigour_summary.json. Seeds 0..4 (the same five seeds).

Everything runs on CPU (the GPU is not used; neural models are trained on CPU here, unlike the GPU
runs that produced the accuracy numbers, so the refit test top-1 is reported next to the published one
as a reproduction check).

Phase train  (--threads 1 or 8): per (benchmark, model, seed)
  featuriser_fit_s : fit of the shared TF-IDF/SVD featuriser on the training split (model independent)
  train_s          : wall-clock clf.fit (sklearn) or the full neural training loop of
                     run_reviewer_experiments.train_eval_neural (same steps, batch size, optimiser and
                     validation every 100 steps for best-checkpoint selection, as in the paper protocol)
  peak_rss_mb      : increase of the process resident set size over its pre-fit level during the fit,
                     sampled every 5 ms from /proc/self/status (approximate)
  n_params         : trainable parameters (neural, logistic regression); tree models: node and leaf
                     counts (not comparable to weights); kNN: number of stored training values
  size_bytes       : serialized model (pickle for sklearn, torch.save(state_dict) for torch)
  test_top1        : test accuracy of this refit (sanity check only; the paper's accuracy numbers stay
                     the rigour-pass ones)
  --threads 1: OMP/MKL/OPENBLAS=1, torch.set_num_threads(1), n_jobs=1 (the launcher pins the process
  to one physical core). --threads 8: the pipeline's multi-thread setting (n_jobs=8, OMP=8,
  torch.set_num_threads(8)), pinned to 8 physical cores. The --threads 1 fitted models are pickled
  to EFF_ART (engine only, not in git) for the latency phase.

Phase latency (always 1 thread, batch size 1): per (benchmark, model, seed) over every test turn of
  the benchmark (at most LAT_N, seeded permutation), after WARMUP untimed turns:
  featurise_ms : raw turn -> shared featuriser -> model input (188 tabular features or 5x146 cells)
  predict_ms   : model forward only (predict_proba / forward + argmax)
  total_ms     : end to end, featurise + predict
  median, p95, mean, n. time.perf_counter_ns.

Output: EFF_OUT/train/t<threads>/<bench>/<model>/seed<k>.json and EFF_OUT/latency/<bench>/<model>/seed<k>.json
(resumable: existing files are skipped). scripts/build_efficiency_1118.py aggregates them.
Usage: run_efficiency_1118.py --phase train|latency --threads 1|8 [--bench ...] [--models ...] [--seeds 0,1,2,3,4]
"""
from __future__ import annotations

import os
import sys


def _early_threads() -> int:
    a = sys.argv
    if "--threads" in a:
        return int(a[a.index("--threads") + 1])
    return 1


THREADS = _early_threads()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import argparse, io, json, pickle, platform, threading, time  # noqa: E402,E401
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STAGE0 = HERE.parent
sys.path.insert(0, str(STAGE0))
sys.path.insert(0, str(STAGE0.parent))
import run_hpo_1110 as H  # noqa: E402
import run_rigour_1110 as RG  # noqa: E402

BENCHES = ("curated", "bfcl_e1v2", "toolbench", "apibank")
MODELS = RG.ALL_CPU + RG.ALL_GPU
SEEDS = (0, 1, 2, 3, 4)
SUMMARY = STAGE0 / "stage0_results" / "rigour_1110" / "rigour_summary.json"
EFF_OUT = Path(os.environ.get("EFF_OUT", STAGE0 / "stage0_results" / "efficiency_1118"))
EFF_ART = Path(os.environ.get("EFF_ART", EFF_OUT / "artifacts"))
LAT_N, WARMUP = int(os.environ.get("LAT_N", 1000)), int(os.environ.get("LAT_WARMUP", 50))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def selected_config(bench, model):
    return json.loads(SUMMARY.read_text())["benchmarks"][bench][model]["config"]


def published_top1(bench, model, seed):
    for r in json.loads(SUMMARY.read_text())["benchmarks"][bench][model]["final"]:
        if r["seed"] == seed:
            return r["test_top1"]
    return None


def hardware():
    cpu = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
    import sklearn, torch
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
            "affinity": sorted(os.sched_getaffinity(0)), "threads": THREADS}


class RSSPeak:
    """Peak increase of VmRSS (MB) over the level at start, sampled every 5 ms."""

    def __enter__(self):
        self.base = self._rss(); self.peak = self.base; self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        return self

    @staticmethod
    def _rss():
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
        return 0.0

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self._rss()); time.sleep(0.005)

    def __exit__(self, *a):
        self._stop = True; self._t.join(); self.peak = max(self.peak, self._rss())
        self.delta_mb = round(self.peak - self.base, 1)


def count_params(model, obj, n_train_feat=None):
    """obj: fitted sklearn estimator or torch module."""
    if model == "logreg":
        return {"n_params": int(obj.coef_.size + obj.intercept_.size), "kind": "weights"}
    if model.startswith("rf"):
        nodes = int(sum(e.tree_.node_count for e in obj.estimators_))
        leaves = int(sum(e.tree_.n_leaves for e in obj.estimators_))
        return {"n_params": None, "n_tree_nodes": nodes, "n_leaves": leaves, "n_trees": len(obj.estimators_), "kind": "tree_nodes"}
    if model.startswith("hgb"):
        preds = [t for itr in obj._predictors for t in itr]
        nodes = int(sum(t.nodes.shape[0] for t in preds))
        leaves = int(sum(int(t.nodes["is_leaf"].sum()) for t in preds))
        return {"n_params": None, "n_tree_nodes": nodes, "n_leaves": leaves, "n_trees": len(preds), "kind": "tree_nodes"}
    if model == "knn":
        return {"n_params": None, "n_stored_values": int(n_train_feat), "kind": "stored_training_data"}
    n = int(sum(p.numel() for p in obj.parameters() if p.requires_grad))
    return {"n_params": n, "kind": "weights"}


def make_sklearn(model, cfg, seed, n_jobs):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    if model == "knn":
        return KNeighborsClassifier(n_neighbors=cfg["n_neighbors"], weights=cfg["weights"], metric=cfg["metric"],
                                    algorithm="brute", n_jobs=n_jobs)
    if model == "logreg":
        return LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=cfg["max_iter"], random_state=seed)
    if model.startswith("rf"):
        return RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                                      max_features=cfg["max_features"], class_weight=cfg["class_weight"], random_state=seed, n_jobs=n_jobs)
    return HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"], max_leaf_nodes=cfg["max_leaf_nodes"],
                                          min_samples_leaf=cfg["min_samples_leaf"], l2_regularization=cfg["l2_regularization"],
                                          class_weight=cfg["class_weight"], early_stopping=False, random_state=seed)


def train_phase(benches, models, seeds):
    import torch
    import warnings
    torch.set_num_threads(THREADS)
    RG._ensure_torch_classes()
    r = H.R()
    hw = hardware()
    for bench in benches:
        todo = [(m, s) for s in seeds for m in models if not (EFF_OUT / "train" / f"t{THREADS}" / bench / m / f"seed{s}.json").exists()]
        if not todo:
            continue
        data = RG.Data(bench, need_cell=True, need_tab=True, device=torch.device("cpu"))
        for seed in sorted({s for _, s in todo}):
            t0 = time.perf_counter(); gt_probe = r.fit_tfidf_svd(data.sp["train"], seed); feat_fit_s = time.perf_counter() - t0
            del gt_probe
            f = data.feats(seed); gt = data._gt[seed]
            for model in [m for m, s in todo if s == seed]:
                cfg = selected_config(bench, model)
                X = RG.X_for(model, cfg, f, ["train", "val", "test"])
                rec = {"bench": bench, "model": model, "seed": seed, "config": cfg, "threads": THREADS,
                       "featuriser_fit_s": round(feat_fit_s, 3), "n_train": int(len(data.y["train"])), "hardware": hw}
                if model in RG.ALL_CPU:
                    clf = make_sklearn(model, cfg, seed, THREADS)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with RSSPeak() as mem:
                            t0 = time.perf_counter(); clf.fit(X["train"], data.y["train"]); train_s = time.perf_counter() - t0
                        P = RG.full_proba(clf, clf.predict_proba(X["test"]))
                    if hasattr(clf, "n_jobs"):
                        clf.n_jobs = 1
                    size = len(pickle.dumps(clf))
                    rec.update(count_params(model, clf, n_train_feat=X["train"].size))
                    obj, obs_c = clf, None
                else:
                    import torch.nn.functional as F
                    Xs = {s: torch.from_numpy(X[s]) for s in X}
                    torch.manual_seed(seed)
                    net = RG.build_net(model, cfg, Xs["train"].shape[1])
                    with RSSPeak() as mem:
                        t0 = time.perf_counter()
                        res = r.train_eval_neural(net, Xs["train"], torch.from_numpy(data.y["train"]), Xs["val"], data.y["val"],
                                                  Xs["val"], data.y["val"], data.labels_all, steps=cfg["train_steps"],
                                                  batch_size=cfg["batch_size"], lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                                  label_smoothing=cfg["label_smoothing"], eval_every=100, seed=seed)
                        train_s = time.perf_counter() - t0
                    net.eval()
                    with torch.no_grad():
                        P = F.softmax(torch.cat([net(Xs["test"][i:i + 4096]) for i in range(0, len(Xs["test"]), 4096)]), -1).numpy()
                    buf = io.BytesIO(); torch.save(net.state_dict(), buf); size = buf.tell()
                    rec.update(count_params(model, net)); rec["best_step"] = int(res["best_step"])
                    obj, obs_c = {k: v.detach().cpu() for k, v in net.state_dict().items()}, int(Xs["train"].shape[1])
                rec.update({"train_s": round(train_s, 3), "peak_rss_mb": mem.delta_mb, "size_bytes": int(size),
                            "test_top1": round(100.0 * float((P.argmax(1) == data.y["test"]).mean()), 2),
                            "published_test_top1": published_top1(bench, model, seed),
                            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                if THREADS == 1:
                    ad = EFF_ART / bench / model; ad.mkdir(parents=True, exist_ok=True)
                    with open(ad / f"seed{seed}.pkl", "wb") as fh:
                        pickle.dump({"model": model, "config": cfg, "obj": obj, "featuriser": gt, "obs_c": obs_c}, fh)
                od = EFF_OUT / "train" / f"t{THREADS}" / bench / model; od.mkdir(parents=True, exist_ok=True)
                (od / f"seed{seed}.json").write_text(json.dumps(rec, indent=1, default=str))
                log(f"TRAIN t{THREADS} {bench}/{model} s{seed} train={train_s:.2f}s size={size} top1={rec['test_top1']} (pub {rec['published_test_top1']})")
        del data


def latency_phase(benches, models, seeds):
    import torch
    assert THREADS == 1, "latency is measured on one thread"
    torch.set_num_threads(1)
    RG._ensure_torch_classes()
    r = H.R()
    hw = hardware()
    st = lambda a: {"median": round(float(np.median(a)), 4), "p95": round(float(np.percentile(a, 95)), 4),
                    "mean": round(float(np.mean(a)), 4), "n": int(len(a))}
    for bench in benches:
        test = [e for e in r.load_split(bench, "test") if e.label in r.L2I]
        idx = np.random.default_rng(20261118).permutation(len(test))[:LAT_N]
        turns = [test[i] for i in idx]
        warm = turns[:WARMUP]
        for model in models:
            for seed in seeds:
                op = EFF_OUT / "latency" / bench / model / f"seed{seed}.json"
                p = EFF_ART / bench / model / f"seed{seed}.pkl"
                if op.exists():
                    continue
                if not p.exists():
                    log(f"missing artifact {p}"); continue
                A = pickle.load(open(p, "rb")); gt = A["featuriser"]; cfg = A["config"]
                if model in RG.ALL_CPU:
                    clf = A["obj"]
                    if hasattr(clf, "n_jobs"):
                        clf.n_jobs = 1
                    use_tab = model in ("logreg", "rf", "hgb") or (model == "knn" and cfg["feat"] == "tab")

                    def feat(ex, use_tab=use_tab):
                        if use_tab:
                            return r.tab_features(gt, [ex])
                        return r.pack_aligned_trace([ex], r.L2I, gt)[0].reshape(1, -1)

                    def pred(x, clf=clf):
                        return int(np.argmax(clf.predict_proba(x)[0]))
                else:
                    net = RG.build_net(model, cfg, A["obs_c"]); net.load_state_dict(A["obj"]); net.eval()

                    def feat(ex):
                        return torch.from_numpy(r.pack_aligned_trace([ex], r.L2I, gt)[0].astype(np.float32))

                    def pred(x, net=net):
                        with torch.no_grad():
                            return int(net(x).argmax(1)[0])
                for ex in warm:
                    pred(feat(ex))
                tf, tp = [], []
                for ex in turns:
                    t0 = time.perf_counter_ns(); x = feat(ex); t1 = time.perf_counter_ns(); pred(x); t2 = time.perf_counter_ns()
                    tf.append((t1 - t0) / 1e6); tp.append((t2 - t1) / 1e6)
                tf, tp = np.array(tf), np.array(tp)
                rec = {"bench": bench, "model": model, "seed": seed, "featurise_ms": st(tf), "predict_ms": st(tp), "total_ms": st(tf + tp),
                       "warmup": WARMUP, "hardware": hw, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                op.parent.mkdir(parents=True, exist_ok=True); op.write_text(json.dumps(rec, indent=1))
                log(f"LAT {bench}/{model} s{seed} predict={rec['predict_ms']['median']}ms total={rec['total_ms']['median']}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("train", "latency"), required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--bench", default=",".join(BENCHES))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    b, m, s = a.bench.split(","), a.models.split(","), [int(x) for x in a.seeds.split(",")]
    (train_phase if a.phase == "train" else latency_phase)(b, m, s)


if __name__ == "__main__":
    main()
