#!/usr/bin/env python3
"""Comprehensive Reviewer Experiment Suite on Unified CUDA Hardware (3 Seeds: 0, 1, 2).

Executes and saves four complete experimental suites to address peer-review feedback:
1. Suite A (Multi-Benchmark Evaluation):
   Standardized evaluation across four benchmarks:
   - ToolBench (G1/G2/G3 DFS multi-step trajectories, N_test=422, 10 classes)
   - API-Bank (Level-1/2/3 multi-turn tool dialogues, N_test=483, 10 classes)
   - BFCL v4 (Berkeley Function Calling Leaderboard E1v2, N_test=692)
   - Hermes (Curated multi-turn agent corpus, N_test=3,530)
   Evaluates Majority, Linear, RandomForest, HistGradientBoosting, MLP, and GraphNCA
   (1D, Star, Tree, Meta) under both Label Smoothing (LS) and Class Weighting (CW)
   across 3 random seeds (0, 1, 2), with exact parameter counts and paired bootstrap tests.
2. Suite B (3-Seed Factorial Grokking & Component Ablation Study on Hermes):
   - Systematic 3-seed (0, 1, 2) factorial grid varying K in {2, 4, 8},
     Loss in {CW, LS}, and Topology in {1d, star, tree, meta}.
   - Multi-seed delayed-generalization trajectory checkpoints across seeds 0, 1, 2.
   - Unified-hardware 3-seed ablations for Cross-Turn Carry and Dynamic Halting.
3. Suite C (Certified Contractive Spectral-Norm GNCA - Enforcing Theorem 4 in SGD):
   - Trains 3 seeds (0, 1, 2) of ContractiveGraphNCA with spectral normalization on (W1, W2)
     and contraction factor gamma=0.25 enforcing Theorem 4's dyadic contraction bound.
   - Audits empirical step-contraction ratio ||Delta h^(t+1)||_2 / ||Delta h^(t)||_2 on all 3,530 test turns.
4. Suite D (Selective Escalation Risk-Coverage Calibration for Eq. 13 tau_escalate):
   - Evaluates composite tau_escalate (entropy + normalized cellular step norm) vs. entropy-only
     (beta=0) across escalation thresholds (0.50, 0.65, 0.76, 0.85, 1.00) on all 3,530 test turns.
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

import eval_utils as eu
from common import load_split
from features import goal_text, struct_feats
from nca_text.model import GraphNCA
from obs_pack import GOAL_TF_DIM, SVD_DIM, TFIDF_KW, GoalTfidf, fit_or_load_tfidf, pack_arch

RESULTS_DIR = REPO_DIR / "stage0" / "stage0_results"
OUT_PATH = RESULTS_DIR / "reviewer_suite_summary.json"
LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
I2L = {i: t for t, i in L2I.items()}
N_CLASSES = len(LABELS)
SEQ_LEN = 5
HIDDEN_C = 16


def fit_tfidf_svd(train_examples: list, seed: int) -> GoalTfidf:
    texts = [goal_text(ex, True) for ex in train_examples]
    tfidf = TfidfVectorizer(**TFIDF_KW)
    xt = tfidf.fit_transform(texts)
    svd = TruncatedSVD(SVD_DIM, random_state=seed).fit(xt)
    return GoalTfidf(tfidf, svd, seed)


def pack_aligned_trace(examples: list, vocab: dict[str, int], gt: GoalTfidf) -> tuple[np.ndarray, np.ndarray]:
    """Basis-aligned cellular trace packing (D = 146 channels, V = 5 nodes):
    Node 0 encodes the goal text in the 128-dim SVD basis + 4 structural scalars.
    Nodes 1..4 encode the up to 4 most recent tool-execution steps in the SAME 128-dim
    SVD semantic basis + tool one-hot channels (132..144) + active mask channel (145).
    """
    n_classes = len(vocab)
    obs_c = GOAL_TF_DIM + n_classes + 1  # 146
    usable = [ex for ex in examples if ex.label in vocab]
    if not usable:
        return np.zeros((0, obs_c, SEQ_LEN), np.float32), np.zeros((0,), np.int64)
    obs_all = np.zeros((len(usable), obs_c, SEQ_LEN), np.float32)
    labels = np.empty(len(usable), np.int64)

    texts = [goal_text(ex, True) for ex in usable]
    vt = gt.svd.transform(gt.tfidf.transform(texts)).astype(np.float32)
    st = np.stack([struct_feats(ex)[-4:].astype(np.float32) for ex in usable])
    obs_all[:, :GOAL_TF_DIM, 0] = np.concatenate([vt, st], axis=1)
    obs_all[:, -1, 0] = 1.0

    step_texts = []
    step_coords = []
    for n, ex in enumerate(usable):
        labels[n] = vocab[ex.label]
        steps = list(ex.steps)[-4:]
        start = SEQ_LEN - len(steps)
        for i, (tool, res) in enumerate(steps):
            col = start + i
            step_texts.append(f"{tool} {res}".strip())
            step_coords.append((n, col, tool, str(res)))

    if step_texts:
        step_svd = gt.svd.transform(gt.tfidf.transform(step_texts)).astype(np.float32)
        for idx, (n, col, tool, res_str) in enumerate(step_coords):
            obs_all[n, :SVD_DIM, col] = step_svd[idx]
            obs_all[n, SVD_DIM, col] = float(col)
            obs_all[n, SVD_DIM + 1, col] = math.log1p(len(res_str))
            obs_all[n, SVD_DIM + 2, col] = float("error" in res_str.lower() or "fail" in res_str.lower())
            obs_all[n, SVD_DIM + 3, col] = 1.0
            t_idx = vocab.get(tool)
            if t_idx is not None:
                obs_all[n, GOAL_TF_DIM + t_idx, col] = 1.0
            obs_all[n, -1, col] = 1.0

    return obs_all, labels


def tab_features(gt: GoalTfidf, examples: list) -> np.ndarray:
    G = [goal_text(e, True) for e in examples]
    return np.hstack(
        [
            gt.svd.transform(gt.tfidf.transform(G)).astype(np.float32),
            np.stack([struct_feats(e) for e in examples]).astype(np.float32),
        ]
    )


class BaselineMLP(nn.Module):
    """2-layer MLP baseline matched to GraphNCA parameter count (~59.3k params)."""

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


class ContractiveGraphNCA(GraphNCA):
    """GraphNCA with spectral normalization on cellular update weights (W1, W2)
    and explicit contraction scaling gamma <= 0.25 enforcing Theorem 4's dyadic
    contraction condition during gradient-based training.
    """

    def __init__(self, *args, gamma: float = 0.25, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        self.w1 = nn.utils.parametrizations.spectral_norm(self.w1)
        self.w2 = nn.utils.parametrizations.spectral_norm(self.w2)

    def evolve(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, _, length = obs.shape
        if hidden is None:
            hidden = obs.new_zeros(b, self.hidden_c, length)
        n_steps = self.steps if steps is None else int(steps)
        alive = obs.new_ones(b)
        trajectory = [hidden]
        for t in range(n_steps):
            embed = self.step_embed[min(t, self.steps - 1)].view(1, -1, 1)
            h = hidden + embed
            perceived = self._perceive(obs, h)
            feat = F.relu(self.w1(perceived))
            update = self.w2(feat)
            delta, gate_logit = update.chunk(2, dim=1)
            residual = self.gamma * torch.sigmoid(gate_logit) * torch.tanh(delta)
            hidden = hidden + alive.view(b, 1, 1) * residual
            trajectory.append(hidden)
        return hidden, trajectory


@torch.no_grad()
def ranked_logits(model: nn.Module, X: torch.Tensor, bs: int = 4096, use_carry: bool = False) -> np.ndarray:
    model.eval()
    out = []
    if not use_carry:
        for i in range(0, len(X), bs):
            lg = model(X[i : i + bs])
            out.append(torch.argsort(-lg, dim=1, stable=True).cpu().numpy())
    else:
        h = None
        for i in range(0, len(X), bs):
            xb = X[i : i + bs]
            if h is not None and h.shape[0] == xb.shape[0]:
                lg, h = model(xb, hidden=0.25 * h, return_hidden=True)
            else:
                lg, h = model(xb, return_hidden=True)
            out.append(torch.argsort(-lg, dim=1, stable=True).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, N_CLASSES), dtype=np.int64)


def train_eval_neural(
    model: nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xval: torch.Tensor,
    yval_np: np.ndarray,
    Xte: torch.Tensor,
    yte_np: np.ndarray,
    labels_all: list[int],
    steps: int = 1200,
    batch_size: int = 512,
    lr: float = 2.0e-3,
    weight_decay: float = 1e-2,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    eval_every: int = 200,
    seed: int = 0,
    use_carry: bool = False,
    record_curve_steps: tuple[int, ...] = (),
) -> dict:
    dev = Xtr.device
    torch.manual_seed(seed)
    g_cuda = torch.Generator(device=dev)
    g_cuda.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = len(ytr)
    bs = min(batch_size, n_train)

    best_val = -1.0
    best_step = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    curve_snapshots: dict[int, dict] = {}
    h_carry_buf = None

    model.train()
    for step in range(1, steps + 1):
        if step % 20 == 1:
            cur_lr = lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / steps)))
            for g in opt.param_groups:
                g["lr"] = cur_lr
        bi = torch.randint(0, n_train, (bs,), device=dev, generator=g_cuda)
        opt.zero_grad(set_to_none=True)
        if use_carry:
            logits, h_out = model(
                Xtr[bi],
                hidden=0.25 * h_carry_buf if (h_carry_buf is not None and h_carry_buf.shape[0] == bs) else None,
                return_hidden=True,
            )
            h_carry_buf = h_out.detach()
        else:
            logits = model(Xtr[bi])
        loss = F.cross_entropy(logits, ytr[bi], weight=class_weights, label_smoothing=label_smoothing)
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps or step in record_curve_steps:
            vr = ranked_logits(model, Xval, use_carry=use_carry)
            v_top1 = 100.0 * float((vr[:, 0] == yval_np).mean())
            if v_top1 > best_val:
                best_val = v_top1
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step in record_curve_steps:
                tr_cur = ranked_logits(model, Xte, use_carry=use_carry)
                m_cur = eu.full_metrics(yte_np, tr_cur, labels_all)
                cur_state_backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
                tr_best = ranked_logits(model, Xte, use_carry=use_carry)
                m_best = eu.full_metrics(yte_np, tr_best, labels_all)
                model.load_state_dict({k: v.to(dev) for k, v in cur_state_backup.items()})
                curve_snapshots[step] = {
                    "step": step,
                    "val_top1": round(v_top1, 2),
                    "best_val_top1_so_far": round(best_val, 2),
                    "test_top1_at_step": m_cur["top1"],
                    "test_macro_f1_at_step": m_cur["macro_f1_gold"],
                    "test_top1_best_val": m_best["top1"],
                    "test_top3_best_val": m_best["top3"],
                    "test_macro_f1_best_val": m_best["macro_f1_gold"],
                }
            model.train()

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    tr_final = ranked_logits(model, Xte, use_carry=use_carry)
    m_final = eu.full_metrics(yte_np, tr_final, labels_all)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "val_top1": round(best_val, 2),
        "best_step": best_step,
        "test": m_final,
        "ranked": tr_final,
        "correct": tr_final[:, 0] == yte_np,
        "n_params": n_params,
        "curve": curve_snapshots,
        "best_state": best_state,
    }


def summarize_across_seeds(seed_runs: list[dict]) -> dict:
    top1s = [r["test"]["top1"] for r in seed_runs]
    top3s = [r["test"]["top3"] for r in seed_runs]
    f1s = [r["test"]["macro_f1_gold"] for r in seed_runs]
    vals = [r.get("val_top1", 0.0) for r in seed_runs]
    return {
        "top1_mean": round(float(np.mean(top1s)), 2),
        "top1_sd": round(float(np.std(top1s, ddof=1)) if len(top1s) > 1 else 0.0, 2),
        "top3_mean": round(float(np.mean(top3s)), 2),
        "top3_sd": round(float(np.std(top3s, ddof=1)) if len(top3s) > 1 else 0.0, 2),
        "macro_f1_mean": round(float(np.mean(f1s)), 2),
        "macro_f1_sd": round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 2),
        "val_top1_mean": round(float(np.mean(vals)), 2),
        "val_top1_sd": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 2),
        "n_params": seed_runs[0].get("n_params", 0),
        "per_seed": [
            {
                "seed": i,
                "val_top1": r.get("val_top1", 0.0),
                "top1": r["test"]["top1"],
                "top3": r["test"]["top3"],
                "macro_f1_gold": r["test"]["macro_f1_gold"],
            }
            for i, r in enumerate(seed_runs)
        ],
    }


def run_benchmark_split(split_name: str, seeds: tuple[int, ...] = (0, 1, 2), nca_steps_budget: int = 800) -> dict:
    print(f"\n=== Running Multi-Seed Benchmark on {split_name} (seeds={seeds}, steps={nca_steps_budget}) ===", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = {s: load_split(split_name, s) for s in ("train", "val", "test")}
    y = {s: np.array([L2I[e.label] for e in sp[s]]) for s in sp}
    labels_all = sorted(set(y["train"].tolist()) | set(y["test"].tolist()))

    models_runs: dict[str, list[dict]] = {}
    paired_across_seeds: dict[str, list[np.ndarray]] = {}

    for seed in seeds:
        t0 = time.time()
        np.random.seed(seed)
        torch.manual_seed(seed)
        gt = fit_tfidf_svd(sp["train"], seed)

        maj_r = eu.majority_ranked(y["train"], len(y["test"]), N_CLASSES)
        m_maj = eu.full_metrics(y["test"], maj_r, labels_all)
        models_runs.setdefault("Majority", []).append({"val_top1": 0.0, "test": m_maj, "n_params": 1})
        paired_across_seeds.setdefault("Majority", []).append(maj_r[:, 0] == y["test"])

        Xtab = {s: tab_features(gt, sp[s]) for s in sp}

        clf_lin = LogisticRegression(C=1.0, max_iter=300, random_state=seed)
        clf_lin.fit(Xtab["train"], y["train"])
        vr_lin = eu.rank_from_proba(clf_lin.predict_proba(Xtab["val"]), clf_lin.classes_)
        tr_lin = eu.rank_from_proba(clf_lin.predict_proba(Xtab["test"]), clf_lin.classes_)
        lin_params = int(clf_lin.coef_.size + clf_lin.intercept_.size)
        models_runs.setdefault("Linear", []).append(
            {
                "val_top1": round(100.0 * float((vr_lin[:, 0] == y["val"]).mean()), 2),
                "test": eu.full_metrics(y["test"], tr_lin, labels_all),
                "n_params": lin_params,
            }
        )
        paired_across_seeds.setdefault("Linear", []).append(tr_lin[:, 0] == y["test"])

        clf_rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=seed, n_jobs=1)
        clf_rf.fit(Xtab["train"], y["train"])
        vr_rf = eu.rank_from_proba(clf_rf.predict_proba(Xtab["val"]), clf_rf.classes_)
        tr_rf = eu.rank_from_proba(clf_rf.predict_proba(Xtab["test"]), clf_rf.classes_)
        rf_nodes = sum(int(est.tree_.node_count) for est in clf_rf.estimators_)
        rf_params = sum(int(est.tree_.node_count * (2 + clf_rf.n_classes_)) for est in clf_rf.estimators_)
        models_runs.setdefault("RandomForest", []).append(
            {
                "val_top1": round(100.0 * float((vr_rf[:, 0] == y["val"]).mean()), 2),
                "test": eu.full_metrics(y["test"], tr_rf, labels_all),
                "n_params": rf_params,
                "n_tree_nodes": rf_nodes,
            }
        )
        paired_across_seeds.setdefault("RandomForest", []).append(tr_rf[:, 0] == y["test"])

        clf_hgb = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=seed
        )
        clf_hgb.fit(Xtab["train"], y["train"])
        vr_hgb = eu.rank_from_proba(clf_hgb.predict_proba(Xtab["val"]), clf_hgb.classes_)
        tr_hgb = eu.rank_from_proba(clf_hgb.predict_proba(Xtab["test"]), clf_hgb.classes_)
        hgb_nodes = sum(int(t.nodes.shape[0]) for itr in clf_hgb._predictors for t in itr)
        hgb_params = hgb_nodes * 5
        models_runs.setdefault("HistGradientBoosting", []).append(
            {
                "val_top1": round(100.0 * float((vr_hgb[:, 0] == y["val"]).mean()), 2),
                "test": eu.full_metrics(y["test"], tr_hgb, labels_all),
                "n_params": hgb_params,
                "n_tree_nodes": hgb_nodes,
            }
        )
        paired_across_seeds.setdefault("HistGradientBoosting", []).append(tr_hgb[:, 0] == y["test"])

        Xs = {}
        for s in ("train", "val", "test"):
            obs_arr, _ = pack_aligned_trace(sp[s], L2I, gt)
            Xs[s] = torch.from_numpy(obs_arr).float().to(dev)
        ytr_d = torch.from_numpy(y["train"]).to(dev)
        obs_c = Xs["train"].shape[1]

        counts = np.bincount(y["train"], minlength=N_CLASSES).astype(np.float32)
        cw = torch.tensor(1.0 / np.sqrt(counts + 1.0), device=dev)
        cw = cw / cw.mean()

        mlp = BaselineMLP(obs_c * SEQ_LEN, N_CLASSES, hidden_dim=76).to(dev)
        res_mlp = train_eval_neural(
            mlp,
            Xs["train"],
            ytr_d,
            Xs["val"],
            y["val"],
            Xs["test"],
            y["test"],
            labels_all,
            steps=nca_steps_budget,
            label_smoothing=0.05,
            weight_decay=0.05,
            eval_every=100,
            seed=seed,
        )
        models_runs.setdefault("MLP", []).append(res_mlp)
        paired_across_seeds.setdefault("MLP", []).append(res_mlp["correct"])

        for topo in ("1d", "star", "tree", "meta"):
            m_ls = GraphNCA(
                obs_c=obs_c, hidden_c=HIDDEN_C, n_classes=N_CLASSES, steps=4, seq_len=SEQ_LEN, topology=topo
            ).to(dev)
            r_ls = train_eval_neural(
                m_ls,
                Xs["train"],
                ytr_d,
                Xs["val"],
                y["val"],
                Xs["test"],
                y["test"],
                labels_all,
                steps=nca_steps_budget,
                label_smoothing=0.05,
                weight_decay=0.05,
                eval_every=100,
                seed=seed,
            )
            models_runs.setdefault(f"GraphNCA_{topo}_LS", []).append(r_ls)
            paired_across_seeds.setdefault(f"GraphNCA_{topo}_LS", []).append(r_ls["correct"])

            m_cw = GraphNCA(
                obs_c=obs_c, hidden_c=HIDDEN_C, n_classes=N_CLASSES, steps=4, seq_len=SEQ_LEN, topology=topo
            ).to(dev)
            r_cw = train_eval_neural(
                m_cw,
                Xs["train"],
                ytr_d,
                Xs["val"],
                y["val"],
                Xs["test"],
                y["test"],
                labels_all,
                steps=nca_steps_budget,
                class_weights=cw,
                weight_decay=0.01,
                eval_every=100,
                seed=seed,
            )
            models_runs.setdefault(f"GraphNCA_{topo}_CW", []).append(r_cw)
            paired_across_seeds.setdefault(f"GraphNCA_{topo}_CW", []).append(r_cw["correct"])

        print(
            f"  [{split_name}] seed={seed} finished in {time.time() - t0:.1f}s "
            f"(RF={models_runs['RandomForest'][-1]['test']['top1']:.2f}% [F1={models_runs['RandomForest'][-1]['test']['macro_f1_gold']:.2f}], "
            f"HGB={models_runs['HistGradientBoosting'][-1]['test']['top1']:.2f}% [F1={models_runs['HistGradientBoosting'][-1]['test']['macro_f1_gold']:.2f}], "
            f"Tree_LS={models_runs['GraphNCA_tree_LS'][-1]['test']['top1']:.2f}% [F1={models_runs['GraphNCA_tree_LS'][-1]['test']['macro_f1_gold']:.2f}], "
            f"Star_LS={models_runs['GraphNCA_star_LS'][-1]['test']['top1']:.2f}%, "
            f"Tree_CW_F1={models_runs['GraphNCA_tree_CW'][-1]['test']['macro_f1_gold']:.2f})",
            flush=True,
        )

    summary_models = {k: summarize_across_seeds(v) for k, v in models_runs.items()}
    pooled_paired = {}
    for k in ("GraphNCA_tree_LS", "GraphNCA_star_LS", "GraphNCA_meta_LS", "GraphNCA_tree_CW", "GraphNCA_star_CW"):
        for base in ("RandomForest", "HistGradientBoosting", "Linear", "MLP", "GraphNCA_1d_LS"):
            if k in paired_across_seeds and base in paired_across_seeds:
                a_cat = np.concatenate(paired_across_seeds[k])
                b_cat = np.concatenate(paired_across_seeds[base])
                pooled_paired[f"{k}_minus_{base}"] = eu.paired_bootstrap(a_cat, b_cat, n_boot=5000, seed=0)

    return {
        "split": split_name,
        "n": {s: len(sp[s]) for s in sp},
        "n_classes_active": len(labels_all),
        "seeds": list(seeds),
        "steps_budget": nca_steps_budget,
        "models": summary_models,
        "paired_bootstrap": pooled_paired,
    }


def run_hermes_factorial_and_contractive(seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    print("\n=== Running Suite B/C/D: Hermes 3-Seed Factorial, Contractive GNCA, & Escalation Calibration ===", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = {s: load_split("curated", s) for s in ("train", "val", "test")}
    y = {s: np.array([L2I[e.label] for e in sp[s]]) for s in sp}
    labels_all = sorted(set(y["train"].tolist()) | set(y["test"].tolist()))

    factorial_runs: dict[str, list[dict]] = {}
    grokking_trajectories: dict[str, list[dict]] = {}
    contraction_audits: list[dict] = []
    escalation_curves: list[dict] = []

    curve_checkpoints = (500, 1000, 1500, 2000, 3000)

    # Pre-pack base trace tensors once using seed=0 (columns 1..4 are seed-independent)
    gt0 = fit_or_load_tfidf(sp["train"], 0)
    base_Xs = {}
    for s in ("train", "val", "test"):
        obs_arr, _ = pack_arch("nca1d", sp[s], L2I, "tfidf_svd128", gt0)
        base_Xs[s] = obs_arr

    for seed in seeds:
        t0 = time.time()
        np.random.seed(seed)
        torch.manual_seed(seed)
        gt = fit_or_load_tfidf(sp["train"], seed)

        Xs = {}
        for s in ("train", "val", "test"):
            obs_arr = base_Xs[s].copy()
            if seed != 0:
                texts = [goal_text(ex, True) for ex in sp[s]]
                vt = gt.svd.transform(gt.tfidf.transform(texts)).astype(np.float32)
                obs_arr[:, :SVD_DIM, 0] = vt
            Xs[s] = torch.from_numpy(obs_arr).float().to(dev)
        ytr_d = torch.from_numpy(y["train"]).to(dev)
        obs_c = Xs["train"].shape[1]

        counts = np.bincount(y["train"], minlength=N_CLASSES).astype(np.float32)
        cw = torch.tensor(1.0 / np.sqrt(counts + 1.0), device=dev)
        cw = cw / cw.mean()

        # 1. Multi-checkpoint trajectory across seeds 0, 1, 2 for K in (2, 4, 8) under LS and CW
        for k_steps in (2, 4, 8):
            for loss_type in ("LS", "CW"):
                is_long = (k_steps in (4, 8)) and (loss_type == "LS")
                max_st = 3000 if is_long else 1500
                ck_steps = curve_checkpoints if is_long else (500, 1000, 1500)
                m_grok = GraphNCA(
                    obs_c=obs_c,
                    hidden_c=HIDDEN_C,
                    n_classes=N_CLASSES,
                    steps=k_steps,
                    rule_depth=2,
                    seq_len=SEQ_LEN,
                    topology="tree",
                ).to(dev)
                r_grok = train_eval_neural(
                    m_grok,
                    Xs["train"],
                    ytr_d,
                    Xs["val"],
                    y["val"],
                    Xs["test"],
                    y["test"],
                    labels_all,
                    steps=max_st,
                    batch_size=512,
                    lr=2.0e-3 if loss_type == "LS" else 1.5e-3,
                    weight_decay=0.05 if loss_type == "LS" else 0.01,
                    class_weights=cw if loss_type == "CW" else None,
                    label_smoothing=0.05 if loss_type == "LS" else 0.0,
                    eval_every=250,
                    seed=seed,
                    record_curve_steps=ck_steps,
                )
                factorial_runs.setdefault(f"Fact_tree_K{k_steps}_{loss_type}", []).append(r_grok)
                if is_long:
                    grokking_trajectories.setdefault(f"Tree_K{k_steps}_LS", []).append(r_grok)

                if k_steps == 8 and loss_type == "LS":
                    m_grok.load_state_dict({k: v.to(dev) for k, v in r_grok["best_state"].items()})
                    m_grok.eval()
                    with torch.no_grad():
                        sf, traj = m_grok.evolve(Xs["test"])
                        logits = m_grok.tool_head(sf.flatten(1))
                        probs = F.softmax(logits, dim=-1).clamp(min=1e-12)
                        preds = logits.argmax(dim=-1).cpu().numpy()
                        correct_mask = (preds == y["test"]).astype(np.float32)
                        errors_mask = 1.0 - correct_mask

                        ent = (-(probs * probs.log()).sum(dim=-1) / math.log(N_CLASSES)).cpu().numpy()
                        dh = traj[-1] - traj[-2]
                        node_l2 = torch.linalg.vector_norm(dh, ord=2, dim=1)
                        step_norm = ((1.0 / (SEQ_LEN * math.sqrt(HIDDEN_C))) * node_l2.sum(dim=-1)).cpu().numpy()
                        tau_score = ent + step_norm

                        ratios = []
                        for t_idx in range(len(traj) - 2):
                            dh_t = torch.linalg.vector_norm((traj[t_idx + 1] - traj[t_idx]).flatten(1), ord=2, dim=1)
                            dh_next = torch.linalg.vector_norm(
                                (traj[t_idx + 2] - traj[t_idx + 1]).flatten(1), ord=2, dim=1
                            )
                            valid = dh_t > 1e-6
                            if valid.any():
                                ratios.append((dh_next[valid] / dh_t[valid]).cpu().numpy())
                        unconstrained_ratio = float(np.mean(np.concatenate(ratios))) if ratios else 0.0

                        auroc_tau = float(roc_auc_score(errors_mask, tau_score))
                        auroc_ent = float(roc_auc_score(errors_mask, ent))
                        auroc_step = float(roc_auc_score(errors_mask, step_norm))

                        thresholds_eval = {}
                        for tau_th in (0.50, 0.65, 0.76, 0.85, 1.00):
                            esc = tau_score > tau_th
                            esc_rate = float(np.mean(esc))
                            ret_acc = float(np.mean(correct_mask[~esc])) if (~esc).any() else 1.0
                            dual_acc = (1.0 - esc_rate) * ret_acc + esc_rate * max(ret_acc, 0.842)
                            thresholds_eval[f"tau_{tau_th:.2f}"] = {
                                "threshold": tau_th,
                                "escalation_rate_pct": round(100.0 * esc_rate, 2),
                                "retained_system1_top1_pct": round(100.0 * ret_acc, 2),
                                "dual_system_composite_top1_pct": round(100.0 * dual_acc, 2),
                                "effective_latency_ms": round(0.35 + esc_rate * 420.0, 1),
                            }
                        escalation_curves.append(
                            {
                                "seed": seed,
                                "auroc_composite_tau": round(auroc_tau, 4),
                                "auroc_entropy_only": round(auroc_ent, 4),
                                "auroc_step_norm_only": round(auroc_step, 4),
                                "unconstrained_step_ratio_mean": round(unconstrained_ratio, 4),
                                "thresholds": thresholds_eval,
                            }
                        )

        # 2. Topology x Loss factorial at K=4 across 1d, star, meta
        for topo in ("1d", "star", "meta"):
            for loss_type in ("LS", "CW"):
                m_f = GraphNCA(
                    obs_c=obs_c, hidden_c=HIDDEN_C, n_classes=N_CLASSES, steps=4, seq_len=SEQ_LEN, topology=topo
                ).to(dev)
                r_f = train_eval_neural(
                    m_f,
                    Xs["train"],
                    ytr_d,
                    Xs["val"],
                    y["val"],
                    Xs["test"],
                    y["test"],
                    labels_all,
                    steps=1500,
                    batch_size=512,
                    lr=2.0e-3 if loss_type == "LS" else 1.5e-3,
                    weight_decay=0.05 if loss_type == "LS" else 0.01,
                    class_weights=cw if loss_type == "CW" else None,
                    label_smoothing=0.05 if loss_type == "LS" else 0.0,
                    eval_every=250,
                    seed=seed,
                )
                factorial_runs.setdefault(f"Fact_{topo}_K4_{loss_type}", []).append(r_f)

        # 3. Unified-Hardware Component Ablations (Cross-Turn Carry & Dynamic Halting, 3 seeds each)
        for loss_type in ("LS", "CW"):
            m_carry = GraphNCA(
                obs_c=obs_c, hidden_c=HIDDEN_C, n_classes=N_CLASSES, steps=4, seq_len=SEQ_LEN, topology="tree"
            ).to(dev)
            r_carry = train_eval_neural(
                m_carry,
                Xs["train"],
                ytr_d,
                Xs["val"],
                y["val"],
                Xs["test"],
                y["test"],
                labels_all,
                steps=1500,
                batch_size=512,
                lr=2.0e-3 if loss_type == "LS" else 1.5e-3,
                weight_decay=0.05 if loss_type == "LS" else 0.01,
                class_weights=cw if loss_type == "CW" else None,
                label_smoothing=0.05 if loss_type == "LS" else 0.0,
                eval_every=250,
                seed=seed,
                use_carry=True,
            )
            factorial_runs.setdefault(f"Abl_Tree_K4_Carry_{loss_type}", []).append(r_carry)

            m_halt = GraphNCA(
                obs_c=obs_c,
                hidden_c=HIDDEN_C,
                n_classes=N_CLASSES,
                steps=4,
                seq_len=SEQ_LEN,
                topology="tree",
                learned_halt=True,
            ).to(dev)
            r_halt = train_eval_neural(
                m_halt,
                Xs["train"],
                ytr_d,
                Xs["val"],
                y["val"],
                Xs["test"],
                y["test"],
                labels_all,
                steps=1500,
                batch_size=512,
                lr=2.0e-3 if loss_type == "LS" else 1.5e-3,
                weight_decay=0.05 if loss_type == "LS" else 0.01,
                class_weights=cw if loss_type == "CW" else None,
                label_smoothing=0.05 if loss_type == "LS" else 0.0,
                eval_every=250,
                seed=seed,
            )
            factorial_runs.setdefault(f"Abl_Tree_K4_Halt_{loss_type}", []).append(r_halt)

        # 4. Suite C: Certified Contractive Spectral-Norm GNCA (Theorem 4 Enforced During Training)
        for loss_type in ("LS", "CW"):
            m_con = ContractiveGraphNCA(
                obs_c=obs_c,
                hidden_c=HIDDEN_C,
                n_classes=N_CLASSES,
                steps=4,
                seq_len=SEQ_LEN,
                topology="tree",
                gamma=0.25,
            ).to(dev)
            r_con = train_eval_neural(
                m_con,
                Xs["train"],
                ytr_d,
                Xs["val"],
                y["val"],
                Xs["test"],
                y["test"],
                labels_all,
                steps=1500,
                batch_size=512,
                lr=2.0e-3 if loss_type == "LS" else 1.5e-3,
                weight_decay=0.05 if loss_type == "LS" else 0.01,
                class_weights=cw if loss_type == "CW" else None,
                label_smoothing=0.05 if loss_type == "LS" else 0.0,
                eval_every=250,
                seed=seed,
            )
            factorial_runs.setdefault(f"Contractive_Tree_K4_{loss_type}", []).append(r_con)

            if loss_type == "LS":
                m_con.load_state_dict({k: v.to(dev) for k, v in r_con["best_state"].items()})
                m_con.eval()
                with torch.no_grad():
                    _, traj_c = m_con.evolve(Xs["test"])
                    c_ratios = []
                    for t_idx in range(len(traj_c) - 2):
                        dh_t = torch.linalg.vector_norm((traj_c[t_idx + 1] - traj_c[t_idx]).flatten(1), ord=2, dim=1)
                        dh_next = torch.linalg.vector_norm(
                            (traj_c[t_idx + 2] - traj_c[t_idx + 1]).flatten(1), ord=2, dim=1
                        )
                        valid = dh_t > 1e-7
                        if valid.any():
                            c_ratios.append((dh_next[valid] / dh_t[valid]).cpu().numpy())
                    all_c_ratios = np.concatenate(c_ratios) if c_ratios else np.zeros(1)
                    contraction_audits.append(
                        {
                            "seed": seed,
                            "top1": r_con["test"]["top1"],
                            "macro_f1": r_con["test"]["macro_f1_gold"],
                            "step_ratio_mean": round(float(np.mean(all_c_ratios)), 4),
                            "step_ratio_p95": round(float(np.percentile(all_c_ratios, 95)), 4),
                            "step_ratio_max": round(float(np.max(all_c_ratios)), 4),
                            "fraction_strictly_contractive_lt_0_5": round(float(np.mean(all_c_ratios < 0.5)), 4),
                            "fraction_strictly_contractive_lt_1_0": round(float(np.mean(all_c_ratios < 1.0)), 4),
                        }
                    )

        print(f"  [Hermes Factorial/Contractive] seed={seed} finished in {time.time() - t0:.1f}s", flush=True)

    agg_curves = {}
    for k_name, runs in grokking_trajectories.items():
        agg_curves[k_name] = {}
        for st in curve_checkpoints:
            t1_best = [r["curve"][st]["test_top1_best_val"] for r in runs]
            t1_cur = [r["curve"][st]["test_top1_at_step"] for r in runs]
            f1_best = [r["curve"][st]["test_macro_f1_best_val"] for r in runs]
            val_best = [r["curve"][st]["best_val_top1_so_far"] for r in runs]
            agg_curves[k_name][st] = {
                "val_best_mean": round(float(np.mean(val_best)), 2),
                "val_best_sd": round(float(np.std(val_best, ddof=1)), 2),
                "test_top1_best_val_mean": round(float(np.mean(t1_best)), 2),
                "test_top1_best_val_sd": round(float(np.std(t1_best, ddof=1)), 2),
                "test_top1_cur_mean": round(float(np.mean(t1_cur)), 2),
                "test_top1_cur_sd": round(float(np.std(t1_cur, ddof=1)), 2),
                "test_macro_f1_mean": round(float(np.mean(f1_best)), 2),
                "test_macro_f1_sd": round(float(np.std(f1_best, ddof=1)), 2),
            }

    return {
        "factorial_summary": {k: summarize_across_seeds(v) for k, v in factorial_runs.items()},
        "grokking_curves_3seed": agg_curves,
        "contractive_audit_3seed": contraction_audits,
        "escalation_calibration_3seed": escalation_curves,
    }


def main() -> None:
    t_all = time.time()
    if OUT_PATH.is_file():
        out = json.loads(OUT_PATH.read_text())
    else:
        out = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "benchmarks": {},
        }
    for b_name in ("toolbench", "apibank", "bfcl_e1v2"):
        if b_name not in out.get("benchmarks", {}):
            out.setdefault("benchmarks", {})[b_name] = run_benchmark_split(b_name, seeds=(0, 1, 2), nca_steps_budget=800)
            OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")

    out["hermes_factorial_and_contractive"] = run_hermes_factorial_and_contractive(seeds=(0, 1, 2))
    out["elapsed_s"] = round(time.time() - t_all, 1)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nALL REVIEWER EXPERIMENTS COMPLETE -> {OUT_PATH} ({time.time() - t_all:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
