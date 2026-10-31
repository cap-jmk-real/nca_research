#!/usr/bin/env python3
"""run_grokking_deep_nca.py: Deep Graph NCAs and Grokking Analysis under Extended Training.

Explores:
1. Extended training (40,000 steps) with strong weight decay (0.05) to study grokking
   (delayed generalization / representation compression).
2. Deeper NCAs along two orthogonal axes:
   - Temporal / Recurrence Depth (steps = 4 vs steps = 8 message-passing iterations)
   - Rule Network Depth (rule_depth = 2 vs rule_depth = 4 with GroupNorm residual blocks)
3. Full tracking of learning trajectories (train_loss, train_acc, val_loss, val_acc, weight_norm).
4. Full test set evaluation (3,530 samples) with 10,000 bootstrap iterations for 95% CI.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

from common import DATA, load_split, metrics, pct
from features import goal_text, struct_feats
from obs_pack import fit_or_load_tfidf, pack_arch
from nca_text.model import GraphNCA

LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
N_CLASSES = len(LABELS)
SEQ_LEN = 5


def bootstrap_top1(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 10000, seed: int = 42):
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
def evaluate_neural(model, obs, y, dev, bs=2048, cw=None):
    model.eval()
    ranked = []
    preds = []
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(y), bs):
        x_b = obs[i:i + bs].to(dev)
        y_b = y[i:i + bs].to(dev)
        logits = model(x_b)
        loss = F.cross_entropy(logits, y_b, weight=cw)
        total_loss += loss.item() * len(y_b)
        n_batches += len(y_b)
        preds += logits.argmax(dim=-1).cpu().tolist()
        ranked += torch.argsort(logits, dim=1, descending=True).cpu().tolist()
    model.train()
    m = metrics(y.tolist(), ranked, N_CLASSES)
    f1 = float(f1_score(y.tolist(), preds, average="macro", zero_division=0) * 100.0)
    m["macro_f1"] = round(f1, 2)
    m["loss"] = round(total_loss / max(n_batches, 1), 4)
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


def compute_weight_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_sq += float(p.pow(2).sum().item())
    return math.sqrt(total_sq)


def main():
    parser = argparse.ArgumentParser(description="Extended Training & Deep Graph NCAs (Grokking Benchmark)")
    parser.add_argument("--steps", type=int, default=40000, help="Extended training steps")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for representation compression")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Validation evaluation interval")
    parser.add_argument("--out", type=str, default="stage0_results/grokking_deep_nca_results.json")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Grokking & Deep NCA Experiment on {dev} ===", flush=True)
    print(f"Steps: {args.steps} | Batch: {args.batch} | Weight Decay: {args.weight_decay} | Seed: {args.seed}", flush=True)

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

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("Loading TF-IDF SVD128 representations...", flush=True)
    goal_tfidf = fit_or_load_tfidf(splits["train"], args.seed)
    sp_neural = {
        s: pack_arch("nca1d", splits[s], L2I, "tfidf_svd128", goal_tfidf)
        for s in ("train", "val", "test")
    }
    Xtr, ytr = map(torch.from_numpy, sp_neural["train"])
    Xv, yv = map(torch.from_numpy, sp_neural["val"])
    Xt, yt = map(torch.from_numpy, sp_neural["test"])
    obs_c = Xtr.shape[1]
    print(f"Observation tensor shape: Xtr={Xtr.shape}, Xt={Xt.shape}", flush=True)

    counts = np.bincount(ytr.numpy(), minlength=N_CLASSES).astype(np.float32)
    cw = torch.tensor(1.0 / np.sqrt(counts + 1), device=dev)
    cw = cw / cw.mean()
    Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)

    # 4 Architectures spanning depth dimensions:
    configs = [
        {"name": "GraphNCA_base_d2_s4", "rule_depth": 2, "steps": 4, "topology": "tree", "desc": "Standard Shallow NCA (depth=2, steps=4)"},
        {"name": "GraphNCA_deep_recur_d2_s8", "rule_depth": 2, "steps": 8, "topology": "tree", "desc": "Deep Recurrence NCA (depth=2, steps=8)"},
        {"name": "GraphNCA_deep_rule_d4_s4", "rule_depth": 4, "steps": 4, "topology": "tree", "desc": "Deep Rule NCA (depth=4, steps=4)"},
        {"name": "GraphNCA_deep_unified_d4_s8", "rule_depth": 4, "steps": 8, "topology": "tree", "desc": "Deep Unified NCA (depth=4, steps=8)"},
    ]

    all_results = {
        "config": vars(args),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "experiments": {},
    }

    for cfg in configs:
        model_name = cfg["name"]
        rd = cfg["rule_depth"]
        st = cfg["steps"]
        topo = cfg["topology"]
        print(f"\n==================================================================", flush=True)
        print(f"Training {model_name}: {cfg['desc']} for {args.steps} steps", flush=True)
        print(f"==================================================================", flush=True)

        model = GraphNCA(
            obs_c=obs_c,
            hidden_c=16,
            n_classes=N_CLASSES,
            steps=st,
            rule_depth=rd,
            seq_len=SEQ_LEN,
            topology=topo,
        ).to(dev)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}", flush=True)

        # AdamW with sustained weight decay for grokking
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        
        # Extended schedule: linear warmup for 1000 steps, then cosine decay down to 1e-4 at step 40000
        def get_lr(step: int) -> float:
            if step <= 1000:
                return 1e-3 * float(step) / 1000.0
            progress = (step - 1000) / max(args.steps - 1000, 1)
            return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1.0 + math.cos(math.pi * progress))

        trajectory = []
        best_val = -1.0
        best_state = None
        t0 = time.time()

        for step in range(1, args.steps + 1):
            lr = get_lr(step)
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

            if step % args.eval_interval == 0 or step == args.steps:
                train_acc = float((logits.argmax(dim=-1) == ytr_d[b_idx_t]).float().mean().item() * 100.0)
                val_m, _ = evaluate_neural(model, Xv, yv, dev, cw=cw)
                v_top1 = val_m["top1"]
                v_loss = val_m["loss"]
                w_norm = compute_weight_norm(model)

                if v_top1 > best_val:
                    best_val = v_top1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                log_entry = {
                    "step": step,
                    "lr": round(lr, 6),
                    "train_loss": round(float(loss.item()), 4),
                    "train_acc": round(train_acc, 2),
                    "val_loss": v_loss,
                    "val_top1": v_top1,
                    "weight_norm": round(w_norm, 3),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                trajectory.append(log_entry)

                print(
                    f"Step {step:5d}/{args.steps} | Train Loss: {loss.item():.4f} (Acc: {train_acc:.1f}%) | "
                    f"Val Loss: {v_loss:.4f} (Top-1: {v_top1:.2f}%, Best: {best_val:.2f}%) | "
                    f"||W||: {w_norm:.2f} ({log_entry['elapsed_s']}s)",
                    flush=True,
                )

        train_time = round(time.time() - t0, 1)

        # 1. Final checkpoint evaluation (to detect grokking at the end of training)
        final_test_m, final_preds = evaluate_neural(model, Xt, yt, dev)
        mean_fin, sd_fin, ci_fin = bootstrap_top1(yt.numpy(), final_preds, seed=args.seed)
        final_test_m["boot_mean"] = mean_fin
        final_test_m["boot_sd"] = sd_fin
        final_test_m["ci95"] = ci_fin

        # 2. Best checkpoint evaluation
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
        best_test_m, best_preds = evaluate_neural(model, Xt, yt, dev)
        mean_best, sd_best, ci_best = bootstrap_top1(yt.numpy(), best_preds, seed=args.seed)
        best_test_m["boot_mean"] = mean_best
        best_test_m["boot_sd"] = sd_best
        best_test_m["ci95"] = ci_best

        lat_gpu = latency(model, Xt, dev) if dev.type == "cuda" else None
        lat_cpu = latency(model.cpu(), Xt, torch.device("cpu"))

        print(f"\n--- {model_name} Final Results ---", flush=True)
        print(f"Best Val Top-1: {best_val:.2f}% | Test Top-1 (Best): {best_test_m['top1']:.2f}% (95% CI: [{ci_best[0]:.2f}, {ci_best[1]:.2f}])", flush=True)
        print(f"Final Step Test Top-1: {final_test_m['top1']:.2f}% (95% CI: [{ci_fin[0]:.2f}, {ci_fin[1]:.2f}])", flush=True)
        print(f"CPU Latency (p50): {lat_cpu['p50_ms']} ms | Training Time: {train_time}s", flush=True)

        all_results["experiments"][model_name] = {
            "config": cfg,
            "params": param_count,
            "train_time_s": train_time,
            "best_val_top1": best_val,
            "best_test": best_test_m,
            "final_test": final_test_m,
            "latency_gpu": lat_gpu,
            "latency_cpu": lat_cpu,
            "trajectory": trajectory,
        }

        # Save checkpoint data continuously
        out_path.write_text(json.dumps(all_results, indent=2))

    print(f"\nAll experiments complete. Results saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
