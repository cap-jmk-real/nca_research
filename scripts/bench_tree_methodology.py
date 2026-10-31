#!/usr/bin/env python3
"""
bench_tree_methodology.py: Rigorous comparative benchmark evaluating Tree Baselines
(HistGradientBoosting, Random Forest) vs. Graph NCAs (1D Lattice, Hierarchical Tree DAG,
and Meta-Learned GNCA) under strict methodological parity:
  - Causal history buffer L=5 (Cell 0: goal, Cells 1-4: execution trace steps)
  - Strict session-level partitioning (zero session ID leakage across train/test)
  - Identical feature representation (spectral TF-IDF SVD and categorical tool encodings)
  - 10,000 bootstrap resamples for mean, SD, and 95% confidence intervals.
"""

import json
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nca_text.model import GraphNCA, TextNCA, FlatPolicy

KEPT_TOOLS = [
    "delegate_task", "execute_code", "memory", "patch", "process",
    "read_file", "search_files", "session_search", "skill_view",
    "terminal", "todo", "web_search", "write_file"
]
TOOL_TO_IDX = {t: i for i, t in enumerate(KEPT_TOOLS)}
N_CLASSES = len(KEPT_TOOLS)
SEQ_LEN = 5  # Cell 0: goal; Cells 1-4: recent steps


def load_dataset(sample_path: Path):
    with open(sample_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records


def extract_features(records, n_svd=48):
    """Extracts continuous spectral SVD features + categorical tool history."""
    corpus = []
    for r in records:
        corpus.append(r.get("goal", ""))
        for s in r.get("recent_steps", [])[-4:]:
            corpus.append(f"{s.get('tool', '')} {s.get('result', '')}")

    tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(corpus)
    svd = TruncatedSVD(n_components=min(n_svd, 32), random_state=42)
    svd.fit(tfidf.transform(corpus))

    X_grid = []
    X_tabular = []
    y = []
    session_ids = []

    for r in records:
        target = r.get("target_tool", "")
        if target not in TOOL_TO_IDX:
            continue
        session_id = r.get("session_id", "")
        grid = np.zeros((n_svd + N_CLASSES, SEQ_LEN), dtype=np.float32)

        # Cell 0: Goal
        goal_text = r.get("goal", "")
        goal_vec = svd.transform(tfidf.transform([goal_text]))[0]
        grid[:len(goal_vec), 0] = goal_vec

        # Cells 1..4: Steps
        recent = r.get("recent_steps", [])[-4:]
        start_idx = SEQ_LEN - len(recent)
        for i, step in enumerate(recent):
            pos = start_idx + i
            tool = step.get("tool", "")
            res = step.get("result", "")
            step_text = f"{tool} {res}"
            step_vec = svd.transform(tfidf.transform([step_text]))[0]
            grid[:len(step_vec), pos] = step_vec
            if tool in TOOL_TO_IDX:
                grid[n_svd + TOOL_TO_IDX[tool], pos] = 1.0

        X_grid.append(grid)
        X_tabular.append(grid.flatten())
        y.append(TOOL_TO_IDX[target])
        session_ids.append(session_id)

    return (
        np.array(X_grid, dtype=np.float32),
        np.array(X_tabular, dtype=np.float32),
        np.array(y, dtype=np.int64),
        session_ids,
    )


def session_split(session_ids, val_ratio=0.2, seed=42):
    unique_sessions = sorted(list(set(session_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_sessions)
    n_val = max(1, int(len(unique_sessions) * val_ratio))
    val_set = set(unique_sessions[:n_val])
    train_mask = np.array([s not in val_set for s in session_ids])
    test_mask = np.array([s in val_set for s in session_ids])
    return train_mask, test_mask


def bootstrap_top1(y_true, y_pred, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    correct = (y_true == y_pred).astype(np.float64)
    n = len(correct)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)
    boot_accs = [
        np.mean(correct[rng.randint(0, n, size=n)]) * 100.0 for _ in range(n_boot)
    ]
    mean = np.mean(boot_accs)
    sd = np.std(boot_accs)
    ci = (np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5))
    return mean, sd, ci


def run_benchmark():
    print("=====================================================================")
    print("Benchmarking Tree Baseline vs. Graph NCAs (Strict Parity, L=5)")
    print("=====================================================================")
    sample_file = Path(__file__).resolve().parents[1] / "data" / "curated" / "sample.jsonl"
    records = load_dataset(sample_file)
    X_grid, X_tab, y, session_ids = extract_features(records)

    train_m, test_m = session_split(session_ids)
    print(f"Total samples: {len(y)} | Train: {train_m.sum()} | Test: {test_m.sum()}")
    print(f"Unique sessions: {len(set(session_ids))} (Strict session-level holdout)")

    # 1. Tree Baselines (HistGradientBoosting, Random Forest)
    hgb = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    hgb.fit(X_tab[train_m], y[train_m])
    pred_hgb = hgb.predict(X_tab[test_m])
    mean_hgb, sd_hgb, ci_hgb = bootstrap_top1(y[test_m], pred_hgb)
    print(f"HistGradientBoosting:    {mean_hgb:6.2f} ± {sd_hgb:4.2f}%  95% CI: [{ci_hgb[0]:.2f}, {ci_hgb[1]:.2f}]")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tab[train_m], y[train_m])
    pred_rf = rf.predict(X_tab[test_m])
    mean_rf, sd_rf, ci_rf = bootstrap_top1(y[test_m], pred_rf)
    print(f"Random Forest Baseline:  {mean_rf:6.2f} ± {sd_rf:4.2f}%  95% CI: [{ci_rf[0]:.2f}, {ci_rf[1]:.2f}]")

    # 2. Graph NCA Models (PyTorch)
    device = torch.device("cpu")
    obs_c = X_grid.shape[1]
    results = {
        "HistGradientBoosting": (mean_hgb, sd_hgb, ci_hgb),
        "RandomForest": (mean_rf, sd_rf, ci_rf),
    }

    for topology in ["1d", "star", "tree", "meta"]:
        torch.manual_seed(42)
        model = GraphNCA(
            obs_c=obs_c,
            hidden_c=16,
            n_classes=N_CLASSES,
            steps=4,
            seq_len=SEQ_LEN,
            topology=topology,
        ).to(device)

        opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        crit = nn.CrossEntropyLoss()

        X_tr = torch.tensor(X_grid[train_m], device=device)
        y_tr = torch.tensor(y[train_m], device=device)
        X_te = torch.tensor(X_grid[test_m], device=device)

        # Train loop
        model.train()
        for epoch in range(250):
            opt.zero_grad()
            out = model(X_tr)
            loss = crit(out, y_tr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out_te = model(X_te)
            pred_nca = out_te.argmax(dim=-1).cpu().numpy()

        mean, sd, ci = bootstrap_top1(y[test_m], pred_nca)
        results[topology] = (mean, sd, ci)
        name = f"GraphNCA ({topology.capitalize()})"
        print(f"{name:<25}: {mean:6.2f} ± {sd:4.2f}%  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    print("\nBenchmark completed successfully with exact methodology parity.")
    return results


if __name__ == "__main__":
    run_benchmark()
