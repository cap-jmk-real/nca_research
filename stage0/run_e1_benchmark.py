#!/usr/bin/env python3
"""E1: BFCL-derived next-tool classification (public HF data only). Checkpoints + holdout test."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

RESULTS_DIR = REPO_DIR / "stage0" / "stage0_results"
CKPT_DIR = RESULTS_DIR / "e1_checkpoints"
SPLIT_NAME = "bfcl_e1"

from graph_topology_export import meta_export_record  # noqa: E402


def _ensure_data(smoke: bool) -> None:
    from common import SPLITS

    if (SPLITS / SPLIT_NAME / "test.jsonl").is_file():
        return
    cmd = [sys.executable, str(REPO_DIR / "stage0" / "e1_prepare_bfcl_splits.py")]
    if smoke:
        cmd.append("--smoke")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--topologies", type=str, default="tree,meta,star,1d")
    parser.add_argument("--dry-run", action="store_true", help="Smoke: finishes in under ~2 minutes")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "e1_benchmark.json"))
    parser.add_argument("--skip-e3", action="store_true", help="Do not run export_meta_adjacency after training")
    args = parser.parse_args()

    if args.dry_run:
        args.steps = min(args.steps, 150)
        args.seeds = "0"
        args.topologies = "1d,meta,star"
        args.batch = min(args.batch, 128)

    from common import require_nca_data_root

    require_nca_data_root()
    _ensure_data(smoke=args.dry_run)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    from common import load_split, maybe_research_backup, metrics
    from features import goal_text, struct_feats
    from nca_text.model import GraphNCA
    from obs_pack import fit_or_load_tfidf, pack_arch
    from run_full_history_benchmarks import bootstrap_top1, evaluate_neural, latency
    from eval_utils import rank_from_proba

    LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
    L2I = {t: i for i, t in enumerate(LABELS)}
    N_CLASSES = len(LABELS)
    SEQ_LEN = 5

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    topologies = [t.strip() for t in args.topologies.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"E1 device={dev} seeds={seeds} topologies={topologies} steps={args.steps}", flush=True)
    splits = {s: load_split(SPLIT_NAME, s) for s in ("train", "val", "test")}
    n_train, n_val, n_test = len(splits["train"]), len(splits["val"]), len(splits["test"])
    print(f"Split {SPLIT_NAME}: train={n_train} val={n_val} test={n_test}", flush=True)

    summary: dict = {
        "experiment": "E1",
        "split": SPLIT_NAME,
        "seeds": seeds,
        "steps": args.steps,
        "dry_run": args.dry_run,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "models": {},
        "checkpoint_dir": str(CKPT_DIR),
    }

    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        goal_tfidf = fit_or_load_tfidf(splits["train"], seed)
        sp_neural = {
            s: pack_arch("nca1d", splits[s], L2I, "tfidf_svd128", goal_tfidf)
            for s in ("train", "val", "test")
        }
        Xtr, ytr = map(torch.from_numpy, sp_neural["train"])
        Xv, yv = map(torch.from_numpy, sp_neural["val"])
        Xt, yt = map(torch.from_numpy, sp_neural["test"])
        obs_c = Xtr.shape[1]

        G = {s: [goal_text(e, True) for e in splits[s]] for s in splits}
        X_tab = {
            s: np.hstack(
                [
                    goal_tfidf.svd.transform(goal_tfidf.tfidf.transform(G[s])),
                    np.stack([struct_feats(e) for e in splits[s]]),
                ]
            )
            for s in splits
        }
        y_tab = {s: np.array([L2I[e.label] for e in splits[s]], dtype=np.int64) for s in splits}

        hgb_iters = 30 if args.dry_run else 200
        clf = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=hgb_iters,
            max_leaf_nodes=31,
            random_state=seed,
        ).fit(X_tab["train"], y_tab["train"])
        val_rank_h = [list(r) for r in rank_from_proba(clf.predict_proba(X_tab["val"]), clf.classes_)]
        v_hgb = metrics(list(y_tab["val"]), val_rank_h, N_CLASSES)["top1"]
        test_rank_h = [list(r) for r in rank_from_proba(clf.predict_proba(X_tab["test"]), clf.classes_)]
        test_pred_h = np.array([r[0] for r in test_rank_h])
        m_hgb = metrics(list(y_tab["test"]), test_rank_h, N_CLASSES)
        mean_h, sd_h, ci_h = bootstrap_top1(y_tab["test"], test_pred_h, seed=seed)
        m_hgb.update(boot_mean=mean_h, boot_sd=sd_h, ci95=ci_h)
        summary["models"].setdefault("HistGradientBoosting", []).append(
            {"seed": seed, "val_top1": v_hgb, "test": m_hgb}
        )

        rf = RandomForestClassifier(
            n_estimators=10 if args.dry_run else 50,
            max_depth=15,
            random_state=seed,
            n_jobs=2,
        ).fit(X_tab["train"], y_tab["train"])
        rf_val_rank = [list(r) for r in rank_from_proba(rf.predict_proba(X_tab["val"]), rf.classes_)]
        rf_test_rank = [list(r) for r in rank_from_proba(rf.predict_proba(X_tab["test"]), rf.classes_)]
        rf_test_pred = np.array([r[0] for r in rf_test_rank])
        m_rf = metrics(list(y_tab["test"]), rf_test_rank, N_CLASSES)
        mean_rf, sd_rf, ci_rf = bootstrap_top1(y_tab["test"], rf_test_pred, seed=seed)
        m_rf.update(boot_mean=mean_rf, boot_sd=sd_rf, ci95=ci_rf)
        summary["models"].setdefault("RandomForest", []).append(
            {
                "seed": seed,
                "val_top1": metrics(list(y_tab["val"]), rf_val_rank, N_CLASSES)["top1"],
                "test": m_rf,
            }
        )

        counts = np.bincount(ytr.numpy(), minlength=N_CLASSES).astype(np.float32)
        cw = torch.tensor(1.0 / np.sqrt(counts + 1), device=dev)
        cw = cw / cw.mean()
        Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)
        batch = min(args.batch, n_train)

        for topo in topologies:
            model_key = f"GraphNCA_{topo}"
            ckpt_path = CKPT_DIR / f"seed{seed}_{topo}.pt"
            start_step = 1
            best_val = -1.0
            best_state = None

            model = GraphNCA(
                obs_c=obs_c,
                hidden_c=16,
                n_classes=N_CLASSES,
                steps=4,
                seq_len=SEQ_LEN,
                topology=topo,
            ).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

            if ckpt_path.is_file() and not args.fresh:
                blob = torch.load(ckpt_path, map_location=dev, weights_only=False)
                model.load_state_dict(blob["model"])
                if blob.get("optimizer"):
                    opt.load_state_dict(blob["optimizer"])
                start_step = int(blob.get("step", 0)) + 1
                best_val = float(blob.get("best_val", -1))
                best_state = blob.get("best_state")
                print(f"Resuming {model_key} seed={seed} from step {start_step}", flush=True)

            t0 = time.time()
            eval_every = 50 if args.dry_run else 500
            for step in range(start_step, args.steps + 1):
                lr = 1e-3 * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / args.steps)))
                for g in opt.param_groups:
                    g["lr"] = lr
                b_idx_t = torch.from_numpy(rng.choice(n_train, batch)).to(dev)
                opt.zero_grad(set_to_none=True)
                logits = model(Xtr_d[b_idx_t])
                loss = F.cross_entropy(logits, ytr_d[b_idx_t], weight=cw)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % eval_every == 0 or step == args.steps:
                    val_metrics, _ = evaluate_neural(model, Xv, yv, dev)
                    v_top1 = val_metrics["top1"]
                    if v_top1 > best_val:
                        best_val = v_top1
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(
                        f"{model_key} seed={seed} step {step}/{args.steps} loss={loss.item():.3f} val={v_top1:.2f} best={best_val:.2f}",
                        flush=True,
                    )
                    torch.save(
                        {
                            "step": step,
                            "best_val": best_val,
                            "best_state": best_state,
                            "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "topology": topo,
                            "seed": seed,
                        },
                        ckpt_path,
                    )

            if best_state is None:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
            test_m, test_preds = evaluate_neural(model, Xt, yt, dev)
            mean_nca, sd_nca, ci_nca = bootstrap_top1(yt.numpy(), test_preds, seed=seed)
            test_m["boot_mean"] = mean_nca
            test_m["boot_sd"] = sd_nca
            test_m["ci95"] = ci_nca
            lat_cpu = latency(model.cpu(), Xt, torch.device("cpu"))

            summary["models"].setdefault(model_key, []).append(
                {
                    "seed": seed,
                    "val_top1": best_val,
                    "test": test_m,
                    "latency_cpu": lat_cpu,
                    "train_time_s": round(time.time() - t0, 1),
                    "checkpoint": str(ckpt_path),
                }
            )

            if topo == "meta" and best_state is not None:
                adj_path = CKPT_DIR / f"seed{seed}_meta_adjacency.json"
                adj_path.write_text(
                    json.dumps(
                        meta_export_record(
                            best_state,
                            seed=seed,
                            checkpoint=str(ckpt_path),
                        ),
                        indent=2,
                    )
                )

        (RESULTS_DIR / f"e1_seed{seed}.json").write_text(json.dumps(summary, indent=2))
        out_path.write_text(json.dumps(summary, indent=2))
        maybe_research_backup(f"e1-seed{seed}")

    print(f"E1 complete → {out_path}", flush=True)
    maybe_research_backup("e1")

    if not args.skip_e3 and "meta" in topologies:
        subprocess.check_call(
            [
                sys.executable,
                str(REPO_DIR / "stage0" / "export_meta_adjacency.py"),
                "--checkpoint-dir",
                str(CKPT_DIR),
                "--seeds",
                args.seeds,
                "--out",
                str(RESULTS_DIR / "e3_meta_adjacency_audit.json"),
            ]
        )


if __name__ == "__main__":
    main()
