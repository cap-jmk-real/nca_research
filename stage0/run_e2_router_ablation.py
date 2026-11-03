#!/usr/bin/env python3
"""E2: router ablation with fixed GraphNCA star actuator (E1 checkpoints). Public BFCL holdout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

RESULTS_DIR = REPO_DIR / "stage0" / "stage0_results"
CKPT_DIR = RESULTS_DIR / "e1_checkpoints"
SPLIT_NAME = "bfcl_e1"
ROUTERS = ("random", "nca1d", "none", "learned")
LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
I2L = {i: t for t, i in L2I.items()}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return (round(100 * (centre - margin) / denom, 2), round(100 * (centre + margin) / denom, 2))


def _ensure_e1(dry_run: bool) -> None:
    if list(CKPT_DIR.glob("seed*_star.pt")):
        return
    cmd = [sys.executable, str(REPO_DIR / "stage0" / "run_e1_benchmark.py")]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.extend(["--seeds", "0,1,2"])
    subprocess.check_call(cmd)


def _load_gnca(topology: str, ckpt_path: Path, obs_c: int, dev):
    import torch
    from nca_text.model import GraphNCA

    model = GraphNCA(
        obs_c=obs_c,
        hidden_c=16,
        n_classes=len(LABELS),
        steps=4,
        seq_len=5,
        topology=topology,
    ).to(dev)
    blob = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state = blob.get("best_state") or blob.get("model")
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_batch(model, obs, dev, bs=512):
    import torch

    preds = []
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            preds.extend(model(obs[i : i + bs].to(dev)).argmax(dim=-1).cpu().tolist())
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--routers", type=str, default=",".join(ROUTERS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "e2_router_ablation.json"))
    ap.add_argument("--skip-e3", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        args.seeds = "0"

    _ensure_e1(args.dry_run)

    import torch

    from common import load_split
    from obs_pack import fit_or_load_tfidf, pack_arch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    routers = [r.strip() for r in args.routers.split(",") if r.strip()]

    train = load_split(SPLIT_NAME, "train")
    test = load_split(SPLIT_NAME, "test")
    summary: dict = {
        "experiment": "E2",
        "split_holdout": f"{SPLIT_NAME}/test",
        "n_test": len(test),
        "dry_run": args.dry_run,
        "routers": {},
        "actuator_topology": "star",
        "config": "configs/dual_system_e2.yaml",
    }

    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        goal_tfidf = fit_or_load_tfidf(train, seed)
        obs, y = pack_arch("nca1d", test, L2I, "tfidf_svd128", goal_tfidf)
        obs_t = torch.from_numpy(obs)
        y_true = y.tolist()
        gold_names = [I2L[i] for i in y_true]
        obs_c = obs_t.shape[1]

        def _ckpt(topo: str) -> Path:
            p = CKPT_DIR / f"seed{seed}_{topo}.pt"
            return p if p.is_file() else CKPT_DIR / f"seed0_{topo}.pt"

        actuator = _load_gnca("star", _ckpt("star"), obs_c, dev)
        act_idx = _predict_batch(actuator, obs_t, dev)
        act_names = [I2L[i] for i in act_idx]

        router_models = {}
        for topo, key in (("1d", "nca1d"), ("meta", "learned")):
            p = _ckpt(topo)
            if p.is_file():
                router_models[key] = _load_gnca(topo, p, obs_c, dev)

        for router in routers:
            if router == "random":
                r_idx = rng.integers(0, len(LABELS), size=len(y_true)).tolist()
            elif router == "none":
                r_idx = act_idx
            elif router == "nca1d":
                r_idx = _predict_batch(router_models["nca1d"], obs_t, dev)
            elif router == "learned":
                r_idx = _predict_batch(router_models["learned"], obs_t, dev)
            else:
                raise SystemExit(f"Unknown router {router}")

            r_names = [I2L[i] for i in r_idx]
            final_names = []
            for rn, an in zip(r_names, act_names, strict=True):
                if router == "none":
                    final_names.append(an)
                elif rn == "respond":
                    final_names.append("respond")
                else:
                    final_names.append(an)

            dual_ok = sum(1 for g, f in zip(gold_names, final_names) if g == f)
            router_ok = sum(1 for g, r in zip(gold_names, r_names) if g == r)
            irrel_idx = [i for i, g in enumerate(gold_names) if g == "respond"]
            irrel_hit = sum(1 for i in irrel_idx if r_names[i] == "respond")

            summary["routers"].setdefault(router, []).append(
                {
                    "seed": seed,
                    "n": len(test),
                    "dual_top1_percent": round(100 * dual_ok / len(test), 2),
                    "router_top1_percent": round(100 * router_ok / len(test), 2),
                    "irrelevance_n": len(irrel_idx),
                    "irrelevance_triage_correct": irrel_hit,
                    "irrelevance_triage_percent": round(100 * irrel_hit / len(irrel_idx), 2)
                    if irrel_idx
                    else None,
                    "irrelevance_wilson_ci95": _wilson(irrel_hit, len(irrel_idx)) if irrel_idx else None,
                    "actuator_checkpoint": str(_ckpt("star")),
                }
            )

        (RESULTS_DIR / f"e2_seed{seed}.json").write_text(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"E2 complete → {out_path}", flush=True)

    if not args.skip_e3 and not args.dry_run:
        subprocess.check_call(
            [
                sys.executable,
                str(REPO_DIR / "stage0" / "export_meta_adjacency.py"),
                "--checkpoint-dir",
                str(CKPT_DIR),
                "--seeds",
                args.seeds if not args.dry_run else "0",
            ]
        )


if __name__ == "__main__":
    main()
