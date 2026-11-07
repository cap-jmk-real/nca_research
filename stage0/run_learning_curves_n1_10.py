#!/usr/bin/env python3
"""run_learning_curves_n1_10.py: Real step-by-step empirical learning curves for
cellular recurrence depths n in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} and baseline topologies
on the curated Hermes benchmark (N_train=57,675, N_val=3,465, N_test=3,530).

Logs instantaneous validation Top-1, cumulative best-checkpoint validation Top-1,
training accuracy, validation loss, and held-out test Top-1 across dense step
checkpoints from step 0 to 10,000.
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
from nca_text.model import GraphNCA
from obs_pack import fit_or_load_tfidf, pack_arch

RESULTS_DIR = REPO_DIR / "stage0" / "stage0_results"
OUT_JSON = RESULTS_DIR / "learning_curves_n1_10.json"
LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
N_CLASSES = len(LABELS)
SEQ_LEN = 5
HIDDEN_C = 16

EVAL_STEPS = [
    0,
    50,
    100,
    200,
    400,
    600,
    800,
    1000,
    1500,
    2000,
    2500,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
]


class FlatMLPTwin(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 76) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.flatten(1))


@torch.no_grad()
def eval_split(model: nn.Module, X: torch.Tensor, y: torch.Tensor, bs: int = 2048, eps: float = 0.05) -> dict:
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total_loss = 0.0
    n = X.shape[0]
    for i in range(0, n, bs):
        xb = X[i : i + bs]
        yb = y[i : i + bs]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, label_smoothing=eps)
        total_loss += float(loss.item()) * len(yb)
        preds = logits.argmax(dim=-1)
        correct_top1 += int((preds == yb).sum().item())
        top3 = torch.topk(logits, k=3, dim=-1).indices
        correct_top3 += int((top3 == yb.unsqueeze(-1)).any(dim=-1).sum().item())
    model.train()
    return {
        "top1": round(100.0 * correct_top1 / max(n, 1), 2),
        "top3": round(100.0 * correct_top3 / max(n, 1), 2),
        "loss": round(total_loss / max(n, 1), 4),
    }


def train_curve(
    name: str,
    model: nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xv: torch.Tensor,
    yv: torch.Tensor,
    Xt: torch.Tensor,
    yt: torch.Tensor,
    steps: int = 10000,
    bs: int = 512,
    lr: float = 1e-3,
    wd: float = 0.05,
    eps: float = 0.05,
    seed: int = 0,
) -> dict:
    dev = Xtr.device
    torch.manual_seed(seed)
    g_cuda = torch.Generator(device=dev)
    g_cuda.manual_seed(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    eval_set = set(EVAL_STEPS)
    n_train = Xtr.shape[0]

    best_val = -1.0
    best_test_at_best_val = 0.0
    best_step = 0
    trajectory = []
    t0 = time.time()

    # Step 0 evaluation (before any gradient update)
    if 0 in eval_set:
        vm0 = eval_split(model, Xv, yv, eps=eps)
        tm0 = eval_split(model, Xt, yt, eps=eps)
        best_val = vm0["top1"]
        best_test_at_best_val = tm0["top1"]
        trajectory.append(
            {
                "step": 0,
                "train_acc": vm0["top1"],
                "val_top1": vm0["top1"],
                "val_top3": vm0["top3"],
                "val_loss": vm0["loss"],
                "best_val_top1": best_val,
                "test_top1": tm0["top1"],
                "best_test_top1": best_test_at_best_val,
            }
        )

    for step in range(1, steps + 1):
        if step <= 500:
            cur_lr = lr * float(step) / 500.0
        else:
            prog = (step - 500) / max(steps - 500, 1)
            cur_lr = lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        bi = torch.randint(0, n_train, (bs,), device=dev, generator=g_cuda)
        opt.zero_grad(set_to_none=True)
        logits = model(Xtr[bi])
        loss = F.cross_entropy(logits, ytr[bi], label_smoothing=eps)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in eval_set:
            tr_acc = round(100.0 * float((logits.argmax(dim=-1) == ytr[bi]).float().mean().item()), 2)
            vm = eval_split(model, Xv, yv, eps=eps)
            tm = eval_split(model, Xt, yt, eps=eps)
            if vm["top1"] > best_val:
                best_val = vm["top1"]
                best_test_at_best_val = tm["top1"]
                best_step = step
            trajectory.append(
                {
                    "step": step,
                    "train_acc": tr_acc,
                    "val_top1": vm["top1"],
                    "val_top3": vm["top3"],
                    "val_loss": vm["loss"],
                    "best_val_top1": best_val,
                    "test_top1": tm["top1"],
                    "best_test_top1": best_test_at_best_val,
                }
            )

    elapsed = round(time.time() - t0, 2)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{name:18s}] params={n_params:6d} | best_val={best_val:.2f}% (at step {best_step:5d}) "
        f"| test@best_val={best_test_at_best_val:.2f}% | final_val={trajectory[-1]['val_top1']:.2f}% ({elapsed:.1f}s)",
        flush=True,
    )
    return {
        "name": name,
        "params": n_params,
        "best_val_top1": best_val,
        "best_step": best_step,
        "test_top1_at_best_val": best_test_at_best_val,
        "final_val_top1": trajectory[-1]["val_top1"],
        "final_test_top1": trajectory[-1]["test_top1"],
        "elapsed_s": elapsed,
        "trajectory": trajectory,
    }


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Real Learning Curves for n in {{1..10}} on {dev} ===", flush=True)
    splits = {s: load_split("curated", s) for s in ("train", "val", "test")}
    gt = fit_or_load_tfidf(splits["train"], 0)
    packed = {s: pack_arch("nca1d", splits[s], L2I, "tfidf_svd128", gt) for s in ("train", "val", "test")}

    Xtr = torch.from_numpy(packed["train"][0]).float().to(dev)
    ytr = torch.from_numpy(packed["train"][1]).long().to(dev)
    Xv = torch.from_numpy(packed["val"][0]).float().to(dev)
    yv = torch.from_numpy(packed["val"][1]).long().to(dev)
    Xt = torch.from_numpy(packed["test"][0]).float().to(dev)
    yt = torch.from_numpy(packed["test"][1]).long().to(dev)
    obs_c = Xtr.shape[1]

    results = {"eval_steps": EVAL_STEPS, "n_runs": {}, "baselines": {}}

    # 1. Run Tree GraphNCA for every n in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    for n_steps in range(1, 11):
        torch.manual_seed(0)
        model = GraphNCA(
            obs_c=obs_c,
            hidden_c=HIDDEN_C,
            n_classes=N_CLASSES,
            steps=n_steps,
            rule_depth=2,
            seq_len=SEQ_LEN,
            topology="tree",
        ).to(dev)
        res = train_curve(
            f"Tree_n{n_steps}",
            model,
            Xtr,
            ytr,
            Xv,
            yv,
            Xt,
            yt,
            steps=10000,
            seed=0,
        )
        res["n_steps"] = n_steps
        results["n_runs"][f"n={n_steps}"] = res

    # 2. Run Real Baseline Curves (Star n=4, 1D Lattice n=4, Meta n=4, Flat MLP)
    for topo_name, topo_key in [("Star_n4", "star"), ("1D_n4", "1d"), ("Meta_n4", "meta")]:
        torch.manual_seed(0)
        m_base = GraphNCA(
            obs_c=obs_c,
            hidden_c=HIDDEN_C,
            n_classes=N_CLASSES,
            steps=4,
            rule_depth=2,
            seq_len=SEQ_LEN,
            topology=topo_key,
        ).to(dev)
        results["baselines"][topo_name] = train_curve(
            topo_name,
            m_base,
            Xtr,
            ytr,
            Xv,
            yv,
            Xt,
            yt,
            steps=10000,
            seed=0,
        )

    torch.manual_seed(0)
    m_mlp = FlatMLPTwin(obs_c * SEQ_LEN, N_CLASSES, hidden_dim=76).to(dev)
    results["baselines"]["Flat_MLP"] = train_curve(
        "Flat_MLP",
        m_mlp,
        Xtr,
        ytr,
        Xv,
        yv,
        Xt,
        yt,
        steps=10000,
        seed=0,
    )

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nSaved complete n in {{1..10}} learning curves to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
