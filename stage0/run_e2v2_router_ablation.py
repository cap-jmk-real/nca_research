#!/usr/bin/env python3
"""E2 v2: router ablation on bfcl_e1v2/test with E1v2 best checkpoints (star actuator).

Routers: random, nca1d (1d best ckpt), none (actuator only), learned (meta best ckpt).
Dual rule unchanged from v1: if the router predicts ``respond`` output ``respond``,
else the actuator's prediction. Test contains BFCL irrelevance rows (gold ``respond``).
Reports majority-class row, macro-F1, per-example predictions, irrelevance triage with
Wilson CI and paired bootstrap of each router vs ``none`` and vs majority.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR)); sys.path.insert(0, str(REPO_DIR / "stage0"))

from run_e1v2_benchmark import I2L, L2I, N_CLASSES, RESULTS_DIR, SPLIT_NAME, eval_nca_from_disk, sha256_file, verify_split  # noqa: E402
from run_e2_router_ablation import _wilson  # noqa: E402

ROUTERS = ("random", "nca1d", "none", "learned")


def dual(router_top1: np.ndarray, act_top1: np.ndarray, router: str) -> np.ndarray:
    if router == "none":
        return act_top1.copy()
    return np.where(router_top1 == L2I["respond"], L2I["respond"], act_top1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--variant", default="run", choices=["run", "run_synth"])
    args = ap.parse_args()
    if args.dry_run:
        args.seeds, args.n_boot = "0", 1000

    from common import SPLITS, load_split, maybe_research_backup, require_nca_data_root
    import eval_utils as eu

    require_nca_data_root()
    verify_split(SPLITS / SPLIT_NAME)
    res = args.results_dir / ("dry_run" if args.dry_run else args.variant)
    ck = res / "checkpoints"; pred_dir = res / "predictions_e2"; pred_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_split(SPLIT_NAME, "train"), load_split(SPLIT_NAME, "test")
    ids = [e.session_id for e in test]
    ytr = np.array([L2I[e.label] for e in train]); y = np.array([L2I[e.label] for e in test])
    labels_all = sorted(set(ytr.tolist()) | set(y.tolist()))
    irr = y == L2I["respond"]
    summary = {"experiment": "E2v2", "split_holdout": f"{SPLIT_NAME}/test", "n_test": len(y),
               "irrelevance_n": int(irr.sum()), "actuator_topology": "star", "routers": {}, "paired": {}}
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = np.random.default_rng(seed)
        feat = ck / f"seed{seed}_features.pkl"
        need = {t: ck / f"seed{seed}_{t}_best.pt" for t in ("star", "1d", "meta")}
        for t, p in need.items():
            if not p.is_file():
                raise SystemExit(f"E2v2 missing E1v2 checkpoint {p} (no fallback)")
        act = eval_nca_from_disk(need["star"], feat, test, "star")[0][:, 0]
        rt = {"nca1d": eval_nca_from_disk(need["1d"], feat, test, "1d")[0][:, 0],
              "learned": eval_nca_from_disk(need["meta"], feat, test, "meta")[0][:, 0],
              "random": rng.integers(0, N_CLASSES, size=len(y)), "none": act}
        maj = np.full(len(y), eu.majority_label(ytr))
        correct = {"Majority": maj == y}
        rows = {"Majority": maj}
        for r in ROUTERS:
            rows[r] = dual(rt[r], act, r)
            correct[r] = rows[r] == y
        for name, pred in rows.items():
            m = {"top1": round(100 * float(correct[name].mean()), 2), **eu.macro_f1s(y, pred, labels_all),
                 **eu.bootstrap_top1(correct[name], args.n_boot, seed)}
            if name in ROUTERS:
                hit = int(((rt[name] == L2I["respond"]) & irr).sum()) if name != "none" else int((act[irr] == L2I["respond"]).sum())
                m.update(router_top1=round(100 * float((rt[name] == y).mean()), 2), irrelevance_triage_correct=hit,
                         irrelevance_triage_percent=round(100 * hit / max(1, irr.sum()), 2),
                         irrelevance_wilson_ci95=_wilson(hit, int(irr.sum())))
            m["predictions_sha256"] = eu.write_predictions(pred_dir / f"{name}_seed{seed}.jsonl", ids, y, pred, I2L)
            summary["routers"].setdefault(name, []).append({"seed": seed, **m})
        summary["paired"][f"seed{seed}"] = {
            **{f"{r}_minus_none": eu.paired_bootstrap(correct[r], correct["none"], args.n_boot, seed) for r in ROUTERS if r != "none"},
            **{f"{r}_minus_Majority": eu.paired_bootstrap(correct[r], correct["Majority"], args.n_boot, seed) for r in ROUTERS}}
        (res / "e2v2_router_ablation.json").write_text(json.dumps(summary, indent=2))
        maybe_research_backup(f"e2v2-seed{seed}")
    print("E2v2 complete", flush=True)


if __name__ == "__main__":
    main()
