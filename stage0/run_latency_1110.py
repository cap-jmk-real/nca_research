#!/usr/bin/env python3
"""End-to-end single-turn CPU latency of the TUNED models (stage0/run_rigour_1110.py artifacts, Hermes).

For each model and each of seeds 0,1,2 we take the pickled fitted model + featuriser and time, per
test turn at batch size 1 on ONE CPU thread (torch.set_num_threads(1), OMP/MKL/OPENBLAS=1, n_jobs=1):
  featurise : raw turn (goal + last 4 tool steps) -> TF-IDF -> SVD -> the model's input
              (188 tabular features, or the 5x146 cell tensor)
  predict   : model forward / predict_proba + argmax
  total     : featurise + predict
N_TURNS fixed test turns (seeded permutation), WARMUP untimed turns first. time.perf_counter_ns.
Output: OUT/latency.json with per-model per-seed median/p95/mean (ms) and serialized size.
"""
from __future__ import annotations
import io, json, os, pickle, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_rigour_1110 as RG  # noqa: E402
import run_hpo_1110 as H  # noqa: E402

N_TURNS, WARMUP = int(os.environ.get("LAT_N", 500)), 30


def main():
    import torch
    torch.set_num_threads(1)
    RG._ensure_torch_classes()
    r = H.R()
    test = r.load_split("curated", "test")
    idx = np.random.default_rng(20261110).permutation(len(test))[: N_TURNS + WARMUP]
    turns = [test[i] for i in idx]
    out = {"protocol": __doc__, "n_turns": N_TURNS, "warmup": WARMUP, "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
           "torch": torch.__version__, "models": {}}
    models = RG.ALL_CPU + RG.ALL_GPU
    for model in models:
        for seed in RG.ART_SEEDS:
            p = RG.ART / model / f"seed{seed}.pkl"
            if not p.exists():
                print("missing", p, flush=True); continue
            A = pickle.load(open(p, "rb")); gt = A["featuriser"]; cfg = A["config"]
            if model in RG.ALL_CPU:
                clf = A["obj"]
                if hasattr(clf, "n_jobs"):
                    clf.n_jobs = 1
                size = len(pickle.dumps(clf))
                use_tab = model in ("logreg", "rf", "hgb") or (model == "knn" and cfg["feat"] == "tab")
                def feat(ex):
                    if use_tab:
                        return r.tab_features(gt, [ex])
                    return r.pack_aligned_trace([ex], r.L2I, gt)[0].reshape(1, -1)
                def pred(x):
                    return int(np.argmax(clf.predict_proba(x)[0]))
            else:
                net = RG.build_net(model, cfg, A["obs_c"]); net.load_state_dict(A["obj"]); net.eval()
                buf = io.BytesIO(); torch.save(net.state_dict(), buf); size = buf.tell()
                def feat(ex):
                    return torch.from_numpy(r.pack_aligned_trace([ex], r.L2I, gt)[0].astype(np.float32))
                def pred(x):
                    with torch.no_grad():
                        return int(net(x).argmax(1)[0])
            tf, tp = [], []
            for k, ex in enumerate(turns):
                t0 = time.perf_counter_ns(); x = feat(ex); t1 = time.perf_counter_ns(); pred(x); t2 = time.perf_counter_ns()
                if k >= WARMUP:
                    tf.append((t1 - t0) / 1e6); tp.append((t2 - t1) / 1e6)
            tf, tp = np.array(tf), np.array(tp); tt = tf + tp
            st = lambda a: {"median": round(float(np.median(a)), 4), "p95": round(float(np.percentile(a, 95)), 4), "mean": round(float(a.mean()), 4)}
            out["models"].setdefault(model, {})[f"seed{seed}"] = {"featurise_ms": st(tf), "predict_ms": st(tp), "total_ms": st(tt), "size_bytes": int(size)}
            print(model, seed, out["models"][model][f"seed{seed}"]["total_ms"], flush=True)
    (RG.OUT / "latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
