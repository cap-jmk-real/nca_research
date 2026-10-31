#!/usr/bin/env python3
"""run_ultra_grokking_nca.py: Ultra-Long Training (200,000+ Steps) for Grokking Analysis on the Best NCA.

Key Optimizations:
1. Model: GraphNCA with Deep Recurrence (rule_depth=2, steps=8, topology=tree/star).
2. Adapted Loss Function:
   - Removes destructive inverse-frequency class weights that caused loss explosion on validation.
   - Employs Label-Smoothed Cross-Entropy (epsilon=0.05) to align loss minimization with Top-1 accuracy
     and provide a bounded minimum that allows weight decay to compress representation rank.
3. Sustained Weight Decay (0.05) with AdamW over 200,000 steps to observe the grokking phase transition.
4. Continuous checkpointing (every 10,000 steps saved to disk).
5. Comprehensive evaluation on 3,530 test split with 10,000 bootstrap resamplings.
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
def evaluate_neural(model, obs, y, dev, bs=2048, label_smoothing=0.05):
    model.eval()
    ranked = []
    preds = []
    total_loss = 0.0
    n_samples = 0
    for i in range(0, len(y), bs):
        x_b = obs[i:i + bs].to(dev)
        y_b = y[i:i + bs].to(dev)
        logits = model(x_b)
        loss = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
        total_loss += loss.item() * len(y_b)
        n_samples += len(y_b)
        preds += logits.argmax(dim=-1).cpu().tolist()
        ranked += torch.argsort(logits, dim=1, descending=True).cpu().tolist()
    model.train()
    m = metrics(y.tolist(), ranked, N_CLASSES)
    f1 = float(f1_score(y.tolist(), preds, average="macro", zero_division=0) * 100.0)
    m["macro_f1"] = round(f1, 2)
    m["loss"] = round(total_loss / max(n_samples, 1), 4)
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
    parser = argparse.ArgumentParser(description="Ultra-Long Training (200k+ Steps) for Grokking Analysis")
    parser.add_argument("--steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for representation compression")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing for loss alignment")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Validation evaluation interval")
    parser.add_argument("--ckpt_interval", type=int, default=10000, help="Checkpoint save interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", type=str, default="stage0_results/ultra_grokking_results.json")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================================", flush=True)
    print(f"=== ULTRA-LONG GROKKING BENCHMARK ON {dev} ({args.steps:,} STEPS) ===", flush=True)
    print(f"Model: GraphNCA (steps=8, rule_depth=2, topology=tree)")
    print(f"Loss: Label-Smoothed Cross-Entropy (eps={args.label_smoothing}) | WD: {args.weight_decay}")
    print(f"==================================================================", flush=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = DATA / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading curated splits...", flush=True)
    splits = {s: load_split("curated", s) for s in ("train", "val", "test")}
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    print(f"Loaded splits: train={n_train:,}, val={n_val:,}, test={n_test:,}", flush=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("Encoding observations...", flush=True)
    goal_tfidf = fit_or_load_tfidf(splits["train"], args.seed)
    sp_neural = {
        s: pack_arch("nca1d", splits[s], L2I, "tfidf_svd128", goal_tfidf)
        for s in ("train", "val", "test")
    }
    Xtr, ytr = map(torch.from_numpy, sp_neural["train"])
    Xv, yv = map(torch.from_numpy, sp_neural["val"])
    Xt, yt = map(torch.from_numpy, sp_neural["test"])
    obs_c = Xtr.shape[1]
    print(f"Obs dimensions: C={obs_c}, V={SEQ_LEN} | Xtr={Xtr.shape}", flush=True)

    Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)

    # Initialize Champion Architecture: Deep Recurrence GraphNCA
    model = GraphNCA(
        obs_c=obs_c,
        hidden_c=16,
        n_classes=N_CLASSES,
        steps=8,
        rule_depth=2,
        seq_len=SEQ_LEN,
        topology="tree",
    ).to(dev)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", flush=True)

    # AdamW optimizer with sustained weight decay
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Extended schedule: warmup for 2,000 steps, then cosine decay down to 5e-5 at step 200,000
    def get_lr(step: int) -> float:
        if step <= 2000:
            return args.lr * float(step) / 2000.0
        progress = (step - 2000) / max(args.steps - 2000, 1)
        return 5e-5 + 0.5 * (args.lr - 5e-5) * (1.0 + math.cos(math.pi * progress))

    trajectory = []
    best_val = -1.0
    best_state = None
    t0 = time.time()

    print(f"\nBeginning ultra-long optimization loop...", flush=True)

    for step in range(1, args.steps + 1):
        lr = get_lr(step)
        for g in opt.param_groups:
            g["lr"] = lr

        b_idx = rng.choice(n_train, args.batch)
        b_idx_t = torch.from_numpy(b_idx).to(dev)

        opt.zero_grad(set_to_none=True)
        logits = model(Xtr_d[b_idx_t])
        
        # ADAPTED LOSS: Label-smoothed unweighted cross-entropy
        loss = F.cross_entropy(logits, ytr_d[b_idx_t], label_smoothing=args.label_smoothing)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Validation & Logging
        if step % args.eval_interval == 0 or step == args.steps:
            train_acc = float((logits.argmax(dim=-1) == ytr_d[b_idx_t]).float().mean().item() * 100.0)
            val_m, _ = evaluate_neural(model, Xv, yv, dev, label_smoothing=args.label_smoothing)
            v_top1 = val_m["top1"]
            v_top3 = val_m["top3"]
            v_loss = val_m["loss"]
            w_norm = compute_weight_norm(model)

            if v_top1 > best_val:
                best_val = v_top1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                # Save best checkpoint
                torch.save(best_state, ckpt_dir / "best_graph_nca_grokking.pt")

            log_entry = {
                "step": step,
                "lr": round(lr, 6),
                "train_loss": round(float(loss.item()), 4),
                "train_acc": round(train_acc, 2),
                "val_loss": v_loss,
                "val_top1": v_top1,
                "val_top3": v_top3,
                "weight_norm": round(w_norm, 3),
                "best_val": best_val,
                "elapsed_s": round(time.time() - t0, 1),
            }
            trajectory.append(log_entry)

            print(
                f"Step {step:6d}/{args.steps} | Train L: {loss.item():.4f} (Acc: {train_acc:.1f}%) | "
                f"Val L: {v_loss:.4f} (Top-1: {v_top1:.2f}%, Top-3: {v_top3:.2f}%, Best: {best_val:.2f}%) | "
                f"||W||: {w_norm:.2f} ({log_entry['elapsed_s']:.1f}s)",
                flush=True,
            )

        # Periodic checkpoint
        if step % args.ckpt_interval == 0:
            torch.save(model.state_dict(), ckpt_dir / f"graph_nca_step_{step:06d}.pt")
            intermediate = {
                "step": step,
                "best_val_top1": best_val,
                "params": param_count,
                "trajectory": trajectory,
            }
            out_path.write_text(json.dumps(intermediate, indent=2))

    train_time = round(time.time() - t0, 1)

    # Final evaluation:
    # 1. Best checkpoint evaluation
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    best_test_m, best_preds = evaluate_neural(model, Xt, yt, dev, label_smoothing=args.label_smoothing)
    mean_best, sd_best, ci_best = bootstrap_top1(yt.numpy(), best_preds, seed=args.seed)
    best_test_m["boot_mean"] = mean_best
    best_test_m["boot_sd"] = sd_best
    best_test_m["ci95"] = ci_best

    lat_gpu = latency(model, Xt, dev) if dev.type == "cuda" else None
    lat_cpu = latency(model.cpu(), Xt, torch.device("cpu"))

    final_results = {
        "config": vars(args),
        "params": param_count,
        "train_time_s": train_time,
        "best_val_top1": best_val,
        "best_test": best_test_m,
        "latency_gpu": lat_gpu,
        "latency_cpu": lat_cpu,
        "trajectory": trajectory,
    }
    out_path.write_text(json.dumps(final_results, indent=2))
    print(f"\n==================================================================", flush=True)
    print(f"Grokking training complete! Elapsed: {train_time}s ({round(train_time/60, 1)}m)", flush=True)
    print(f"Best Val Top-1: {best_val:.2f}% | Test Top-1: {best_test_m['top1']:.2f}% (95% CI: [{ci_best[0]:.2f}, {ci_best[1]:.2f}])", flush=True)
    print(f"Results saved to: {out_path}", flush=True)
    print(f"Best model weights saved to: {ckpt_dir / 'best_graph_nca_grokking.pt'}", flush=True)
    print(f"==================================================================", flush=True)


if __name__ == "__main__":
    main()
