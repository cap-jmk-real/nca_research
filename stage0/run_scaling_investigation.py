#!/usr/bin/env python3
"""Scale Investigation: What happens when Graph NCA is scaled to tree parameter counts?

Evaluates Tree Graph NCA across parameter scales:
- Standard (59.2k params): hidden_c=16, update_hidden=96
- Scale 1 (150k params): hidden_c=32, update_hidden=192
- Scale 2 (427k params): hidden_c=64, update_hidden=384
- Scale 3 (1.36M params): hidden_c=128, update_hidden=768
- Scale 4 (4.75M params): hidden_c=256, update_hidden=1536

Evaluates on Curated Hermes splits across seeds {0, 1, 2} using cosine AdamW schedules (3,000 steps).
Records:
- Train Top-1
- Curated Validation Top-1
- Curated Test Top-1
- Overfitting Gap (Train - Test)
- Inference Latency (Batch-1 CPU)
- Parameter Efficiency (% / 1k params)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

from common import load_split
from features import goal_text
from nca_text.model import GraphNCA
from obs_pack import SVD_DIM, fit_or_load_tfidf, pack_arch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_JSON = REPO_DIR / "stage0" / "stage0_results" / "parameter_scaling_investigation.json"
LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
N_CLASSES = len(LABELS)


def load_all_data():
    sp = {s: load_split("curated", s) for s in ("train", "val", "test")}
    y = {s: np.array([L2I[e.label] for e in sp[s]], dtype=np.int64) for s in sp}

    gt0 = fit_or_load_tfidf(sp["train"], 0)
    base_Xs = {}
    for s in ("train", "val", "test"):
        obs_arr, _ = pack_arch("nca1d", sp[s], L2I, "tfidf_svd128", gt0)
        base_Xs[s] = obs_arr

    return sp, y, base_Xs


def get_seed_tensors(sp, y, base_Xs, seed: int):
    gt = fit_or_load_tfidf(sp["train"], seed)
    Xs = {}
    for s in ("train", "val", "test"):
        obs_arr = base_Xs[s].copy()
        if seed != 0:
            texts = [goal_text(ex, True) for ex in sp[s]]
            vt = gt.svd.transform(gt.tfidf.transform(texts)).astype(np.float32)
            obs_arr[:, :SVD_DIM, 0] = vt
        Xs[s] = torch.from_numpy(obs_arr).float().to(DEVICE)
    ys = {s: torch.from_numpy(y[s]).to(DEVICE) for s in ("train", "val", "test")}
    return Xs, ys


def train_and_eval(
    Xs, ys, hidden_c: int, update_hidden: int,
    seed: int, steps: int = 3000, batch_size: int = 256,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_tr, act_tr = Xs["train"], ys["train"]
    obs_va, act_va = Xs["val"], ys["val"]
    obs_te, act_te = Xs["test"], ys["test"]

    obs_c = obs_tr.shape[1]

    model = GraphNCA(
        obs_c=obs_c,
        hidden_c=hidden_c,
        n_classes=N_CLASSES,
        steps=4,
        update_hidden=update_hidden,
        topology="tree",
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    n_tr = obs_tr.shape[0]
    t0 = time.time()

    best_val_top1 = 0.0
    best_state = None

    for step in range(1, steps + 1):
        idx = torch.randint(0, n_tr, (batch_size,), device=DEVICE)
        b_obs = obs_tr[idx]
        b_act = act_tr[idx]

        logits = model(b_obs)
        loss = F.cross_entropy(logits, b_act, label_smoothing=0.05)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % 500 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                va_logits = model(obs_va)
                va_pred = va_logits.argmax(dim=-1)
                va_top1 = (va_pred == act_va).float().mean().item()
                if va_top1 > best_val_top1:
                    best_val_top1 = va_top1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        tr_sub = obs_tr[:5000]
        tr_pred = model(tr_sub).argmax(dim=-1)
        tr_top1 = (tr_pred == act_tr[:5000]).float().mean().item()

        va_pred = model(obs_va).argmax(dim=-1)
        va_top1 = (va_pred == act_va).float().mean().item()

        te_pred = model(obs_te).argmax(dim=-1)
        te_top1 = (te_pred == act_te).float().mean().item()

        # Batch-1 CPU latency
        cpu_model = GraphNCA(
            obs_c=obs_c, hidden_c=hidden_c, n_classes=N_CLASSES,
            steps=4, update_hidden=update_hidden, topology="tree"
        )
        cpu_model.eval()
        single_x = obs_te[0:1].cpu()
        for _ in range(10):
            _ = cpu_model(single_x)
        t_cpu_0 = time.perf_counter()
        n_passes = 50
        for _ in range(n_passes):
            _ = cpu_model(single_x)
        lat_ms = ((time.perf_counter() - t_cpu_0) / n_passes) * 1000.0

    return {
        "n_params": n_params,
        "train_top1": tr_top1,
        "val_top1": va_top1,
        "test_top1": te_top1,
        "overfit_gap": tr_top1 - te_top1,
        "cpu_lat_ms": lat_ms,
        "eff_per_1k": (te_top1 * 100.0) / (n_params / 1000.0),
        "train_time_s": train_time,
    }


def main():
    print(f"=== Graph NCA Parameter Scaling Investigation on {DEVICE} ===")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...", flush=True)
    sp, y, base_Xs = load_all_data()

    scales = [
        ("Standard_59k", 16, 96),
        ("TreeScale_150k", 32, 192),
        ("TreeScale_427k", 64, 384),
        ("Large_1.36M", 128, 768),
        ("RFScale_4.75M", 256, 1536),
    ]

    seeds = [0, 1, 2]
    results = {}

    for name, hc, uh in scales:
        print(f"\n--- Evaluating Scale: {name} (hidden_c={hc}, update_hidden={uh}) ---", flush=True)
        scale_runs = []
        for seed in seeds:
            print(f"  Seed {seed}...", end="", flush=True)
            Xs, ys = get_seed_tensors(sp, y, base_Xs, seed)
            run_res = train_and_eval(
                Xs, ys, hidden_c=hc, update_hidden=uh,
                seed=seed, steps=3000, batch_size=256,
            )
            print(f" test_top1={run_res['test_top1']*100:.2f}%, val_top1={run_res['val_top1']*100:.2f}%, train_top1={run_res['train_top1']*100:.2f}%, params={run_res['n_params']:,}, lat={run_res['cpu_lat_ms']:.2f}ms ({run_res['train_time_s']:.1f}s)", flush=True)
            scale_runs.append(run_res)

        test_accs = [r["test_top1"] for r in scale_runs]
        val_accs = [r["val_top1"] for r in scale_runs]
        train_accs = [r["train_top1"] for r in scale_runs]
        overfit_gaps = [r["overfit_gap"] for r in scale_runs]
        latencies = [r["cpu_lat_ms"] for r in scale_runs]
        effs = [r["eff_per_1k"] for r in scale_runs]

        summary = {
            "name": name,
            "hidden_c": hc,
            "update_hidden": uh,
            "n_params": scale_runs[0]["n_params"],
            "test_top1_mean": float(np.mean(test_accs)),
            "test_top1_std": float(np.std(test_accs)),
            "val_top1_mean": float(np.mean(val_accs)),
            "val_top1_std": float(np.std(val_accs)),
            "train_top1_mean": float(np.mean(train_accs)),
            "train_top1_std": float(np.std(train_accs)),
            "overfit_gap_mean": float(np.mean(overfit_gaps)),
            "cpu_lat_ms_mean": float(np.mean(latencies)),
            "eff_per_1k_mean": float(np.mean(effs)),
            "runs": scale_runs,
        }
        results[name] = summary
        print(f"  Summary for {name}: {summary['test_top1_mean']*100:.2f}% +/- {summary['test_top1_std']*100:.2f}%, Overfit gap: {summary['overfit_gap_mean']*100:.2f}%, Eff: {summary['eff_per_1k_mean']:.4f}%/1k, Lat: {summary['cpu_lat_ms_mean']:.2f}ms", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll scaling experiments completed successfully! Results written to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
