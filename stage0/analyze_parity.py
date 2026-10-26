#!/usr/bin/env python3
"""Input-parity analysis: mean ± sd over seeds; 10k paired bootstrap on curated test rows;
session-level bootstrap on Hermes OOD (cluster_ci from eval_hermes_ood.py)."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, load_split
from train_eval import LABELS, build, pack
from obs_pack import fit_or_load_tfidf

B = 10000
RNG = np.random.default_rng(20261026)
BASE = DATA / "stage0_results" / "sota_baselines_20261026"
CURATED_FLOOR_PP = 52.9


def row_boot(d: np.ndarray) -> list[float]:
    n = len(d)
    bs = [d[RNG.integers(0, n, n)].mean() for _ in range(B)]
    return [round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)]


def session_boot(d: np.ndarray, sess: np.ndarray) -> list[float]:
    us = np.unique(sess)
    mem = [np.where(sess == u)[0] for u in us]
    bs = []
    for _ in range(B):
        ix = np.concatenate([mem[i] for i in RNG.integers(0, len(us), len(us))])
        bs.append(d[ix].mean())
    return [round(100 * np.percentile(bs, 2.5), 2), round(100 * np.percentile(bs, 97.5), 2)]


def load_v2(seed: int) -> dict:
    return json.load(open(BASE / f"rowpreds_v2_s{seed}.json"))


def stack_hgb(split: str, seeds: range) -> np.ndarray:
    gold = np.array(load_v2(0)["gold"][split])
    return np.stack([(np.array(load_v2(s)["hgb"][split]) == gold).astype(float) for s in seeds])


def find_run(arch: str, text_feat: str, seed: int) -> Path | None:
    for d in sorted(glob.glob(str(DATA / "stage0_runs" / "*"))):
        ck = Path(d) / "best.pt"
        if not ck.is_file():
            continue
        a = torch.load(ck, map_location="cpu")["args"]
        if a.get("distill"):
            continue
        if a.get("arch") != arch or a.get("seed") != seed:
            continue
        if a.get("text_feat", "hash48") != text_feat:
            continue
        return Path(d)
    return None


def score_run(run: Path, examples: list) -> np.ndarray:
    ck = torch.load(run / "best.pt", map_location="cpu")
    a = ck["args"]
    tf = fit_or_load_tfidf(load_split("curated", "train"), a["seed"]) if a.get("text_feat") == "tfidf_svd128" else None
    X, y = pack(a["arch"], examples, a.get("text_feat", "hash48"), tf)
    X = torch.from_numpy(X)
    m = build(a["arch"], X, a["hidden"], a["nca_steps"])
    m.load_state_dict(ck["state"])
    m.eval()
    with torch.no_grad():
        lg = torch.cat([m(X[i : i + 1024]) for i in range(0, len(X), 1024)])
    return (lg.argmax(1).numpy() == y).astype(float)


def stack_neural(arch: str, text_feat: str, examples: list, seeds: range) -> np.ndarray | None:
    rows = []
    for s in seeds:
        run = find_run(arch, text_feat, s)
        if run is None:
            return None
        rows.append(score_run(run, examples))
    return np.stack(rows)


def ms(C: np.ndarray) -> tuple[float, float | None]:
    t = 100 * C.mean(1)
    sd = round(float(t.std(ddof=1)), 2) if len(t) > 1 else None
    return round(float(t.mean()), 2), sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    seeds = range(3)
    out_dir = args.out or (DATA / "stage0_results" / f"rowpreds_{date.today().isoformat()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_ex = load_split("curated", "test")
    hermes_ex = load_split("hermes_ood", "test")
    sess = np.array([e.session_id for e in hermes_ex])
    R: dict = {"bootstrap": B, "curated_test": {"floor_top1_pp": CURATED_FLOOR_PP}, "hermes_ood": {}}

    hgb_t = stack_hgb("test", seeds)
    R["curated_test"]["hgb"] = dict(zip(("top1_mean", "top1_sd"), ms(hgb_t)))
    best_neural = None
    best_acc = -1.0
    for arch in ("nca1d", "graph", "flat", "gru", "tx"):
        for feat in ("hash48", "tfidf_svd128"):
            C = stack_neural(arch, feat, test_ex, seeds)
            if C is None:
                continue
            key = f"{arch}_{feat}"
            R["curated_test"][key] = dict(zip(("top1_mean", "top1_sd"), ms(C)))
            if feat == "tfidf_svd128":
                H = stack_neural(arch, "hash48", test_ex, seeds)
                if H is not None:
                    d = C.mean(0) - H.mean(0)
                    R["curated_test"][f"{key}_minus_hash48"] = {
                        "diff_pp": round(100 * d.mean(), 2),
                        "ci95_row": row_boot(d),
                    }
                mc = C.mean(0)
                d = mc - hgb_t.mean(0)
                R["curated_test"][f"{key}_minus_hgb"] = {
                    "diff_pp": round(100 * d.mean(), 2),
                    "ci95_row": row_boot(d),
                }
                acc = float(C.mean())
                if acc > best_acc:
                    best_acc = acc
                    best_neural = (arch, mc)
    if best_neural:
        arch, mc = best_neural
        d = mc - hgb_t.mean(0)
        R["curated_test"]["best_neural_tfidf_minus_hgb"] = {
            "arch": arch,
            "diff_pp": round(100 * d.mean(), 2),
            "ci95_row": row_boot(d),
        }

    hgb_h = stack_hgb("hermes", seeds)
    mc_h = hgb_h.mean(0)
    floor = (np.array(load_v2(0)["gold"]["hermes"]) == LABELS.index("terminal")).astype(float)
    R["hermes_ood"] = {
        "n_rows": len(floor),
        "n_sessions": int(len(np.unique(sess))),
        "floor_terminal_top1": round(100 * floor.mean(), 2),
        "hgb": dict(zip(("top1_mean", "top1_sd"), ms(hgb_h))),
        "hgb_minus_floor": {
            "diff_pp": round(100 * (mc_h - floor).mean(), 2),
            "ci95_session": session_boot(mc_h - floor, sess),
        },
    }
    for arch in ("nca1d", "graph", "flat", "gru", "tx"):
        for feat in ("hash48", "tfidf_svd128"):
            C = stack_neural(arch, feat, hermes_ex, seeds)
            if C is None:
                continue
            mc = C.mean(0)
            key = f"{arch}_{feat}"
            R["hermes_ood"][key] = dict(zip(("top1_mean", "top1_sd"), ms(C)))
            R["hermes_ood"][f"{key}_minus_floor"] = {
                "diff_pp": round(100 * (mc - floor).mean(), 2),
                "ci95_session": session_boot(mc - floor, sess),
            }

    (out_dir / "parity_analysis.json").write_text(json.dumps(R, indent=2) + "\n")
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
