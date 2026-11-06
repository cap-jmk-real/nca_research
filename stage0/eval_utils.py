"""Shared evaluation helpers for E1/E2 (v2 split).

- Rankings from sklearn classifiers are mapped through ``clf.classes_`` so that
  column positions are converted to global label ids (bug in v1: argsort
  positions were compared with label ids).
- Top-1 predictions are always ``ranked[:, 0]`` so bootstrap, top-1 and
  per-example predictions are consistent.
- Paired bootstrap resamples the same test indices for both systems.
- Macro-F1 is reported two ways: ``macro_f1_gold`` averages over classes present
  in the gold test labels; ``macro_f1_all`` averages over every class present in
  train ∪ test (passed explicitly, so a class absent from test but predicted
  contributes F1=0 and an absent, never-predicted class contributes 0 as well).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def rank_from_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return (n, k) array of global label ids sorted by descending probability.

    Ties broken by lower global label id (stable)."""
    proba = np.asarray(proba)
    classes = np.asarray(classes)
    order = np.argsort(-proba, axis=1, kind="stable")
    return classes[order]


def top1_from_ranked(ranked) -> np.ndarray:
    return np.asarray([r[0] for r in ranked], dtype=np.int64)


def majority_label(y_train) -> int:
    vals, counts = np.unique(np.asarray(y_train), return_counts=True)
    return int(vals[np.argmax(counts)])  # ties -> lowest id


def majority_ranked(y_train, n_test: int, n_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y_train), minlength=n_classes)
    order = np.argsort(-counts, kind="stable")
    return np.tile(order, (n_test, 1))


def macro_f1s(y_true, y_pred, labels_all) -> dict:
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    gold = sorted(set(y_true.tolist()))
    return {
        "macro_f1_gold": round(100 * float(f1_score(y_true, y_pred, labels=gold, average="macro", zero_division=0)), 2),
        "macro_f1_all": round(100 * float(f1_score(y_true, y_pred, labels=sorted(labels_all), average="macro", zero_division=0)), 2),
        "macro_f1_gold_labels": gold,
        "macro_f1_all_labels": sorted(int(x) for x in labels_all),
    }


def full_metrics(y_true, ranked, labels_all) -> dict:
    y_true = np.asarray(y_true)
    ranked = np.asarray(ranked)
    pred = ranked[:, 0]
    out = {
        "n": int(len(y_true)),
        "top1": round(100 * float((pred == y_true).mean()), 2),
        "top3": round(100 * float(np.mean([y in r[:3] for y, r in zip(y_true, ranked)])), 2),
    }
    out.update(macro_f1s(y_true, pred, labels_all))
    return out


def bootstrap_top1(correct, n_boot: int = 10000, seed: int = 0) -> dict:
    correct = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1) * 100
    return {
        "boot_mean": round(float(accs.mean()), 3),
        "boot_sd": round(float(accs.std()), 3),
        "ci95": [round(float(np.percentile(accs, 2.5)), 3), round(float(np.percentile(accs, 97.5)), 3)],
    }


def paired_bootstrap(correct_a, correct_b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap of top-1 accuracy difference A - B (percentage points).

    p-value: two-sided, from the bootstrap distribution of the difference centred
    at the observed difference (shift method), +1 smoothing."""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.shape == b.shape
    d = a - b
    obs = float(d.mean() * 100)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1) * 100
    centred = boots - obs
    p = (1 + np.sum(np.abs(centred) >= abs(obs))) / (n_boot + 1)
    return {
        "diff_pp": round(obs, 3),
        "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "p_two_sided": round(float(p), 5),
        "n": int(len(d)),
        "n_boot": n_boot,
        "a_only_correct": int(np.sum((a == 1) & (b == 0))),
        "b_only_correct": int(np.sum((a == 0) & (b == 1))),
    }


def write_predictions(path: Path, ids, y_true, ranked, i2l) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    ranked = np.asarray(ranked)
    for ex_id, y, r in zip(ids, y_true, ranked):
        r = np.atleast_1d(r)
        row = {"id": ex_id, "gold": i2l[int(y)], "pred": i2l[int(r[0])], "correct": bool(int(r[0]) == int(y))}
        if ranked.ndim == 2:
            row["top3"] = [i2l[int(x)] for x in r[:3]]
        lines.append(json.dumps(row))
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
