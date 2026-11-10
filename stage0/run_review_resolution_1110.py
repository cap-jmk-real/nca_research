#!/usr/bin/env python3
"""Review-resolution rerun (2026-11-10). Re-uses run_reviewer_experiments.run_benchmark_split
unchanged except that per-example correctness is kept, so paired tests can resample test
examples with the 3 seeds kept together (no pseudo-replication).

Outputs (stage0/stage0_results/review_1110/):
  multibench.json         per-benchmark summaries for ToolBench, API-Bank, BFCL (E1v2 split), Hermes
  correct_by_seed.npz     per-seed per-example top-1 correctness for every model
  jacobian_audit.json     empirical per-step Jacobian spectral norms of the trained Tree-GNCA
                          (Hermes, all test turns, all K steps, 3 seeds)
  cpu_cost.json           batch-1 CPU latency and serialized size of RF, HGB and Tree-GNCA (Hermes seed 0)
Hermes uses the 3000-step budget of the factorial run; the other benchmarks use 800 steps as before.
"""
from __future__ import annotations

import io
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_reviewer_experiments as R  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402

OUT = R.RESULTS_DIR / "review_1110"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2)
BENCH = (("toolbench", 800), ("apibank", 800), ("bfcl_e1v2", 800), ("curated", 3000))


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def jacobian_audit(states: list[dict], split: str) -> dict:
    """Spectral norm of d h_{t+1} / d h_t for every test turn and every cell step t."""
    torch.set_num_threads(4)
    sp = {s: R.load_split(split, s) for s in ("train", "test")}
    per_seed = []
    for seed, st in zip(SEEDS, states):
        gt = R.fit_tfidf_svd(sp["train"], seed)
        obs_arr, _ = R.pack_aligned_trace(sp["test"], R.L2I, gt)
        obs = torch.from_numpy(obs_arr).float()
        m = R.GraphNCA(obs_c=obs.shape[1], hidden_c=R.HIDDEN_C, n_classes=R.N_CLASSES, steps=4,
                       seq_len=R.SEQ_LEN, topology="tree")
        m.load_state_dict(st)
        m.eval()

        def step_fn(h, o, t):
            embed = m.step_embed[t].view(1, -1, 1)
            hh = h.unsqueeze(0) + embed
            per = m._perceive(o.unsqueeze(0), hh)
            feat = torch.relu(m.w1(per))
            delta, gate = m.w2(feat).chunk(2, dim=1)
            return (h.unsqueeze(0) + torch.sigmoid(gate) * delta).squeeze(0)

        norms = []
        with torch.no_grad():
            _, traj = m.evolve(obs)
        for t in range(m.steps):
            H = traj[t]
            for i in range(0, obs.shape[0], 256):
                hb, ob = H[i:i + 256], obs[i:i + 256]
                J = torch.func.vmap(torch.func.jacrev(lambda h, o: step_fn(h, o, t)))(hb, ob)
                J = J.reshape(J.shape[0], hb[0].numel(), hb[0].numel())
                norms.append(torch.linalg.matrix_norm(J, ord=2).detach().numpy())
        n = np.concatenate(norms)
        per_seed.append({
            "seed": seed, "n_jacobians": int(n.size), "n_test_turns": int(obs.shape[0]), "steps": m.steps,
            "spectral_norm_max": round(float(n.max()), 4), "spectral_norm_p95": round(float(np.percentile(n, 95)), 4),
            "spectral_norm_mean": round(float(n.mean()), 4), "spectral_norm_min": round(float(n.min()), 4),
            "fraction_below_1": round(float((n < 1.0).mean()), 4),
        })
    return {"model": "GraphNCA_tree_LS (K=4)", "split": split, "per_seed": per_seed}


def cpu_cost(state: dict, split: str) -> dict:
    torch.set_num_threads(1)
    sp = {s: R.load_split(split, s) for s in ("train", "test")}
    y = np.array([R.L2I[e.label] for e in sp["train"]])
    gt = R.fit_tfidf_svd(sp["train"], 0)
    Xtr, Xte = R.tab_features(gt, sp["train"]), R.tab_features(gt, sp["test"])
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=0, n_jobs=1).fit(Xtr, y)
    hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=0).fit(Xtr, y)
    obs_arr, _ = R.pack_aligned_trace(sp["test"], R.L2I, gt)
    obs = torch.from_numpy(obs_arr).float()
    m = R.GraphNCA(obs_c=obs.shape[1], hidden_c=R.HIDDEN_C, n_classes=R.N_CLASSES, steps=4,
                   seq_len=R.SEQ_LEN, topology="tree")
    m.load_state_dict(state)
    m.eval()

    def lat(fn, xs, n=1000, warm=100):
        for i in range(warm):
            fn(xs[i % len(xs)])
        ts = []
        for i in range(n):
            t0 = time.perf_counter()
            fn(xs[i % len(xs)])
            ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        return {"p50_ms": round(float(np.median(ts)), 4), "p95_ms": round(float(np.percentile(ts, 95)), 4),
                "mean_ms": round(float(ts.mean()), 4), "n": n}

    with torch.no_grad():
        nca_lat = lat(lambda o: m(o.unsqueeze(0)), obs)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return {
        "cpu": cpu_name(), "threads": 1, "split": split,
        "RandomForest": {"latency": lat(lambda x: rf.predict_proba(x.reshape(1, -1)), Xte),
                         "pickle_bytes": len(pickle.dumps(rf)),
                         "n_tree_nodes": int(sum(e.tree_.node_count for e in rf.estimators_)),
                         "config": "n_estimators=50, max_depth=15"},
        "HistGradientBoosting": {"latency": lat(lambda x: hgb.predict_proba(x.reshape(1, -1)), Xte),
                                 "pickle_bytes": len(pickle.dumps(hgb)),
                                 "config": "max_iter=200, max_leaf_nodes=31, lr=0.1"},
        "GraphNCA_tree_LS": {"latency": nca_lat,
                             "state_dict_bytes": len(buf.getvalue()),
                             "n_params": int(sum(p.numel() for p in m.parameters()))},
        "note": "sklearn predict_proba batch-1 includes Python call overhead; GNCA forward via PyTorch eager.",
    }


def main() -> None:
    t_all = time.time()
    summary = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "seeds": list(SEEDS), "benchmarks": {}}
    correct = {}
    hermes_states = None
    for name, steps in BENCH:
        t0 = time.time()
        res = R.run_benchmark_split(name, seeds=SEEDS, nca_steps_budget=steps, return_correct=True,
                                    keep_states=("GraphNCA_tree_LS",) if name == "curated" else ())
        for k, v in res.pop("correct_by_seed").items():
            correct[f"{name}/{k}"] = np.array(v, dtype=np.int8)
        st = res.pop("_states", {})
        if name == "curated":
            hermes_states = st["GraphNCA_tree_LS"]
        res["elapsed_s"] = round(time.time() - t0, 1)
        summary["benchmarks"][name] = res
        (OUT / "multibench.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(OUT / "correct_by_seed.npz", **correct)
    (OUT / "jacobian_audit.json").write_text(json.dumps(jacobian_audit(hermes_states, "curated"), indent=2) + "\n")
    (OUT / "cpu_cost.json").write_text(json.dumps(cpu_cost(hermes_states[0], "curated"), indent=2) + "\n")
    summary["elapsed_s"] = round(time.time() - t_all, 1)
    (OUT / "multibench.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE {summary['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
