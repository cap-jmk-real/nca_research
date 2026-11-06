#!/usr/bin/env python3
"""
run_full_history_benchmarks.py: Rigorous full-scale comparative benchmark
evaluating Tree Baselines (HistGradientBoosting, Random Forest) vs. Graph NCAs
(Hierarchical Tree DAG A_tree, Meta-Learned Continuous Topology, Star, and 1D Lattice)
under strict methodological parity:
  - Causal history buffer L=5 (Cell 0: goal, Cells 1-4: execution trace steps)
  - Strict session-level holdout partition (curated train, val, test)
  - Identical feature representation: TF-IDF SVD128 + categorical tool history (obs_c=146)
  - Full GPU training on engine (CUDA)
  - 10,000 bootstrap resamples on test split for mean, SD, and 95% confidence intervals.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

# Add repo paths
REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

from common import DATA, load_split, metrics, pct
from features import goal_text, struct_feats
from obs_pack import fit_or_load_tfidf, pack_arch
from eval_utils import rank_from_proba
from nca_text.model import GraphNCA

LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
N_CLASSES = len(LABELS)
SEQ_LEN = 5


def bootstrap_top1(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 10000, seed: int = 42):
    """Vectorized, fast bootstrap evaluation."""
    rng = np.random.RandomState(seed)
    correct = (y_true == y_pred).astype(np.float64)
    n = len(correct)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_accs = correct[idx].mean(axis=1) * 100.0
    mean = float(np.mean(boot_accs))
    sd = float(np.std(boot_accs))
    ci = (float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5)))
    return mean, sd, ci


@torch.no_grad()
def evaluate_neural(model, obs, y, dev, bs=2048):
    model.eval()
    ranked = []
    preds = []
    for i in range(0, len(y), bs):
        logits = model(obs[i:i + bs].to(dev))
        preds += logits.argmax(dim=-1).cpu().tolist()
        ranked += torch.argsort(logits, dim=1, descending=True).cpu().tolist()
    model.train()
    m = metrics(y.tolist(), ranked, N_CLASSES)
    f1 = float(f1_score(y.tolist(), preds, average="macro", zero_division=0) * 100.0)
    m["macro_f1"] = round(f1, 2)
    return m, np.array(preds)


@torch.no_grad()
def latency(model, obs, dev, n=300):
    model = model.to(dev).eval()
    xs = []
    for i in range(n + 30):
        x = obs[i % len(obs)].unsqueeze(0).to(dev)
        t = time.perf_counter()
        _ = model(x).argmax().item()
        ms = (time.perf_counter() - t) * 1000.0
        if i >= 30:
            xs.append(ms)
    return {"p50_ms": round(float(pct(xs, 50)), 3), "p95_ms": round(float(pct(xs, 95)), 3)}


def main():
    parser = argparse.ArgumentParser(description="Full history & tree methodology parity benchmark")
    parser.add_argument("--steps", type=int, default=10000, help="NCA training steps")
    parser.add_argument("--batch", type=int, default=512, help="NCA batch size")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds")
    parser.add_argument("--topologies", type=str, default="tree,meta,star,1d", help="Topologies to benchmark")
    parser.add_argument("--out", type=str, default="stage0_results/full_history_benchmarks.json")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    topologies = [t.strip() for t in args.topologies.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {dev} | Seeds: {seeds} | Topologies: {topologies}", flush=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = DATA / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading curated splits...", flush=True)
    splits = {s: load_split("curated", s) for s in ("train", "val", "test")}
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    print(f"Loaded splits: train={n_train}, val={n_val}, test={n_test}", flush=True)

    benchmark_summary = {
        "seeds": seeds,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "models": {},
    }

    for seed in seeds:
        print(f"\n==================== SEED {seed} ====================", flush=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        # 1. Feature Preparation (TF-IDF SVD128)
        print("Loading / computing TF-IDF SVD128 representations...", flush=True)
        goal_tfidf = fit_or_load_tfidf(splits["train"], seed)
        sp_neural = {
            s: pack_arch("nca1d", splits[s], L2I, "tfidf_svd128", goal_tfidf)
            for s in ("train", "val", "test")
        }
        Xtr, ytr = map(torch.from_numpy, sp_neural["train"])
        Xv, yv = map(torch.from_numpy, sp_neural["val"])
        Xt, yt = map(torch.from_numpy, sp_neural["test"])
        obs_c = Xtr.shape[1]
        print(f"Neural observation shape: Xtr={Xtr.shape}, Xt={Xt.shape}", flush=True)

        # Tabular features for Tree baselines (TF-IDF SVD128 + struct_feats)
        G = {s: [goal_text(e, True) for e in splits[s]] for s in splits}
        X_tab = {
            s: np.hstack([
                goal_tfidf.svd.transform(goal_tfidf.tfidf.transform(G[s])),
                np.stack([struct_feats(e) for e in splits[s]])
            ])
            for s in splits
        }
        y_tab = {s: np.array([L2I[e.label] for e in splits[s]], dtype=np.int64) for s in splits}

        # ---------------------------------------------------------
        # 2. HistGradientBoosting Baseline
        # ---------------------------------------------------------
        print("\n--- Training HistGradientBoosting ---", flush=True)
        t0 = time.time()
        clf = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            max_features=0.8,
            early_stopping=False,
            random_state=seed,
        ).fit(X_tab["train"], y_tab["train"])
        val_preds = clf.predict_proba(X_tab["val"])
        val_rank = [list(r) for r in rank_from_proba(val_preds, clf.classes_)]
        v_top1 = metrics(list(y_tab["val"]), val_rank, N_CLASSES)["top1"]
        print(f"HGB (lr=0.1, it=200): val_top1={v_top1:.2f}% ({round(time.time() - t0, 1)}s)", flush=True)

        test_preds_hgb = clf.predict_proba(X_tab["test"])
        test_rank_hgb = [list(r) for r in rank_from_proba(test_preds_hgb, clf.classes_)]
        test_pred_labels_hgb = np.array([r[0] for r in test_rank_hgb])
        m_hgb = metrics(list(y_tab["test"]), test_rank_hgb, N_CLASSES)
        m_hgb["macro_f1"] = round(float(f1_score(list(y_tab["test"]), test_pred_labels_hgb, average="macro", zero_division=0) * 100.0), 2)
        mean_hgb, sd_hgb, ci_hgb = bootstrap_top1(y_tab["test"], test_pred_labels_hgb, seed=seed)
        m_hgb["boot_mean"] = mean_hgb
        m_hgb["boot_sd"] = sd_hgb
        m_hgb["ci95"] = ci_hgb
        print(f"HGB Test Top-1: {m_hgb['top1']:.2f}% (Boot: {mean_hgb:.2f} ± {sd_hgb:.2f}%, 95% CI: [{ci_hgb[0]:.2f}, {ci_hgb[1]:.2f}])", flush=True)

        if "HistGradientBoosting" not in benchmark_summary["models"]:
            benchmark_summary["models"]["HistGradientBoosting"] = []
        benchmark_summary["models"]["HistGradientBoosting"].append({
            "seed": seed,
            "val_top1": v_top1,
            "test": m_hgb,
            "hparams": {"lr": 0.1, "max_iter": 200},
        })

        # ---------------------------------------------------------
        # 3. Random Forest Baseline
        # ---------------------------------------------------------
        print("\n--- Training Random Forest Baseline ---", flush=True)
        t_rf0 = time.time()
        rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=seed, n_jobs=4)
        rf.fit(X_tab["train"], y_tab["train"])
        rf_val_prob = rf.predict_proba(X_tab["val"])
        rf_val_rank = [list(r) for r in rank_from_proba(rf_val_prob, rf.classes_)]
        rf_val_top1 = metrics(list(y_tab["val"]), rf_val_rank, N_CLASSES)["top1"]

        rf_test_prob = rf.predict_proba(X_tab["test"])
        rf_test_rank = [list(r) for r in rank_from_proba(rf_test_prob, rf.classes_)]
        rf_pred_labels = np.array([r[0] for r in rf_test_rank])
        m_rf = metrics(list(y_tab["test"]), rf_test_rank, N_CLASSES)
        m_rf["macro_f1"] = round(float(f1_score(list(y_tab["test"]), rf_pred_labels, average="macro", zero_division=0) * 100.0), 2)
        mean_rf, sd_rf, ci_rf = bootstrap_top1(y_tab["test"], rf_pred_labels, seed=seed)
        m_rf["boot_mean"] = mean_rf
        m_rf["boot_sd"] = sd_rf
        m_rf["ci95"] = ci_rf
        print(f"RandomForest Test Top-1: {m_rf['top1']:.2f}% (Boot: {mean_rf:.2f} ± {sd_rf:.2f}%) ({round(time.time() - t_rf0, 1)}s)", flush=True)

        if "RandomForest" not in benchmark_summary["models"]:
            benchmark_summary["models"]["RandomForest"] = []
        benchmark_summary["models"]["RandomForest"].append({
            "seed": seed,
            "val_top1": rf_val_top1,
            "test": m_rf,
        })

        # ---------------------------------------------------------
        # 4. Graph NCA Topologies
        # ---------------------------------------------------------
        counts = np.bincount(ytr.numpy(), minlength=N_CLASSES).astype(np.float32)
        cw = torch.tensor(1.0 / np.sqrt(counts + 1), device=dev)
        cw = cw / cw.mean()
        Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)

        for topo in topologies:
            model_key = f"GraphNCA_{topo}"
            print(f"\n--- Training GraphNCA (topology={topo}) ---", flush=True)
            model = GraphNCA(
                obs_c=obs_c,
                hidden_c=16,
                n_classes=N_CLASSES,
                steps=4,
                seq_len=SEQ_LEN,
                topology=topo,
            ).to(dev)

            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            best_val = -1.0
            best_state = None
            t_train0 = time.time()

            for step in range(1, args.steps + 1):
                lr = 1e-3 * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / args.steps)))
                for g in opt.param_groups:
                    g["lr"] = lr
                b_idx = rng.choice(n_train, args.batch)
                b_idx_t = torch.from_numpy(b_idx).to(dev)

                opt.zero_grad(set_to_none=True)
                logits = model(Xtr_d[b_idx_t])
                loss = F.cross_entropy(logits, ytr_d[b_idx_t], weight=cw)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % 500 == 0 or step == args.steps:
                    val_metrics, _ = evaluate_neural(model, Xv, yv, dev)
                    v_top1 = val_metrics["top1"]
                    if v_top1 > best_val:
                        best_val = v_top1
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    if step % 2500 == 0 or step == args.steps:
                        print(f"Step {step:5d} / {args.steps} | Loss: {loss.item():.4f} | Val Top-1: {v_top1:.2f}% (Best: {best_val:.2f}%)", flush=True)

            train_elapsed = round(time.time() - t_train0, 1)
            # Evaluate best checkpoint on test set
            model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
            test_m, test_preds = evaluate_neural(model, Xt, yt, dev)
            mean_nca, sd_nca, ci_nca = bootstrap_top1(yt.numpy(), test_preds, seed=seed)
            test_m["boot_mean"] = mean_nca
            test_m["boot_sd"] = sd_nca
            test_m["ci95"] = ci_nca

            lat_gpu = latency(model, Xt, dev) if dev.type == "cuda" else None
            lat_cpu = latency(model.cpu(), Xt, torch.device("cpu"))
            print(f"{model_key} Test Top-1: {test_m['top1']:.2f}% (Boot: {mean_nca:.2f} ± {sd_nca:.2f}%, 95% CI: [{ci_nca[0]:.2f}, {ci_nca[1]:.2f}]) | CPU Latency: {lat_cpu['p50_ms']} ms", flush=True)

            if model_key not in benchmark_summary["models"]:
                benchmark_summary["models"][model_key] = []
            benchmark_summary["models"][model_key].append({
                "seed": seed,
                "val_top1": best_val,
                "test": test_m,
                "latency_gpu": lat_gpu,
                "latency_cpu": lat_cpu,
                "train_time_s": train_elapsed,
                "params": sum(p.numel() for p in model.parameters()),
            })

        # Save intermediate summary after each seed
        out_path.write_text(json.dumps(benchmark_summary, indent=2))

    # Save final summary
    out_path.write_text(json.dumps(benchmark_summary, indent=2))
    print(f"\n=======================================================", flush=True)
    print(f"Benchmark run complete. Results written to: {out_path}", flush=True)
    print(f"=======================================================", flush=True)


if __name__ == "__main__":
    main()
