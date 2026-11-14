#!/usr/bin/env python3
"""Train a BFCL-adapted Tree GraphNCA strictly on `train` IDs from scripts/bfcl_gnca_split.json
and evaluate router top-1 & respond-gate accuracy on `heldout` IDs vs. the Hermes checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

from bfcl_label_map import gold_label_from_answer, map_bfcl_function  # noqa: E402
from common import DATA, Example  # noqa: E402
from nca_text.model import GraphNCA  # noqa: E402
from obs_pack import fit_or_load_tfidf, pack_arch  # noqa: E402

BFCL_DATA_DIR = Path(
    "/home/s4il0r/nca_e_logs/official_bfcl_v4_20261111/venv/lib/python3.11/site-packages/bfcl_eval/data"
)
SPLIT_JSON = REPO_DIR / "scripts" / "bfcl_gnca_split.json"
OUT_CKPT = DATA / "stage0_results" / "checkpoints" / "best_graph_nca_bfcl_split.pt"
HERMES_CKPT = DATA / "stage0_results" / "checkpoints" / "best_graph_nca_grokking.pt"
OUT_REPORT = REPO_DIR / "scripts" / "bfcl_gnca_train_eval.json"

LABELS = json.loads((REPO_DIR / "stage0" / "labels.json").read_text())
L2I = {t: i for i, t in enumerate(LABELS)}
RESPOND_IDX = L2I["respond"]

SINGLE_AST_CATS = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]
IRREL_CATS = ["irrelevance", "live_irrelevance"]
MULTI_TURN_CATS = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _user_text(turn_msgs) -> str:
    if isinstance(turn_msgs, list):
        for m in turn_msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
        if turn_msgs and isinstance(turn_msgs[0], dict):
            return str(turn_msgs[0].get("content") or "")
    return str(turn_msgs or "")


def build_examples(split_map: dict) -> tuple[list[tuple[str, Example]], list[tuple[str, Example]]]:
    train_pool: list[tuple[str, Example]] = []
    heldout: list[tuple[str, Example]] = []

    def add(cat: str, eid: str, ex: Example):
        if eid in split_map[cat]["train"]:
            train_pool.append((cat, ex))
        elif eid in split_map[cat]["heldout"]:
            heldout.append((cat, ex))

    for cat in SINGLE_AST_CATS:
        q_rows = _load_jsonl(BFCL_DATA_DIR / f"BFCL_v4_{cat}.json")
        a_rows = {str(r["id"]): r for r in _load_jsonl(BFCL_DATA_DIR / "possible_answer" / f"BFCL_v4_{cat}.json")}
        for row in q_rows:
            eid = str(row["id"])
            goal = _user_text(row["question"][0])
            lbl = gold_label_from_answer(a_rows.get(eid), irrelevance=False)
            if lbl in L2I:
                add(cat, eid, Example(eid, lbl, goal, []))

    for cat in IRREL_CATS:
        for row in _load_jsonl(BFCL_DATA_DIR / f"BFCL_v4_{cat}.json"):
            eid = str(row["id"])
            goal = _user_text(row["question"][0])
            add(cat, eid, Example(eid, "respond", goal, []))

    for row in _load_jsonl(BFCL_DATA_DIR / "BFCL_v4_live_relevance.json"):
        eid = str(row["id"])
        goal = _user_text(row["question"][0])
        fns = row.get("function") or []
        lbl = map_bfcl_function(str(fns[0].get("name", ""))) if fns else "execute_code"
        if lbl in L2I:
            add("live_relevance", eid, Example(eid, lbl, goal, []))

    for cat in MULTI_TURN_CATS:
        q_rows = _load_jsonl(BFCL_DATA_DIR / f"BFCL_v4_{cat}.json")
        a_rows = {str(r["id"]): r for r in _load_jsonl(BFCL_DATA_DIR / "possible_answer" / f"BFCL_v4_{cat}.json")}
        for row in q_rows:
            eid = str(row["id"])
            ans = a_rows.get(eid, {})
            gt_turns = ans.get("ground_truth") or []
            steps: list[tuple[str, str]] = []
            for t_idx, turn_msgs in enumerate(row["question"]):
                goal = _user_text(turn_msgs)
                calls = gt_turns[t_idx] if t_idx < len(gt_turns) else []
                for call_str in calls:
                    fn_name = str(call_str).split("(", 1)[0].strip()
                    lbl = map_bfcl_function(fn_name)
                    if lbl in L2I:
                        add(cat, eid, Example(eid, lbl, goal, list(steps)))
                        steps.append((lbl, str(call_str)[:120]))
                # End-of-turn (or empty turn when func/param is missing) -> respond
                add(cat, eid, Example(eid, "respond", goal, list(steps)))

    return train_pool, heldout


@torch.no_grad()
def eval_model(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, dev: torch.device, bs: int = 1024):
    model.eval()
    preds = []
    for i in range(0, len(y), bs):
        logits = model(X[i : i + bs].to(dev))
        preds.append(logits.argmax(dim=-1).cpu())
    p = torch.cat(preds)
    top1 = float((p == y).float().mean().item() * 100.0)
    gate_acc = float(((p == RESPOND_IDX) == (y == RESPOND_IDX)).float().mean().item() * 100.0)
    is_resp = y == RESPOND_IDX
    resp_recall = float(((p[is_resp] == RESPOND_IDX).float().mean().item() * 100.0)) if is_resp.any() else 0.0
    tool_mask = ~is_resp
    tool_pass = float(((p[tool_mask] != RESPOND_IDX).float().mean().item() * 100.0)) if tool_mask.any() else 0.0
    return {
        "n": int(len(y)),
        "top1": round(top1, 2),
        "gate_acc": round(gate_acc, 2),
        "respond_recall": round(resp_recall, 2),
        "tool_pass_rate": round(tool_pass, 2),
    }, p


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_map = json.loads(SPLIT_JSON.read_text())["split"]
    train_pool, heldout = build_examples(split_map)

    # Split train_pool 85/15 by session_id hash (heldout untouched)
    tr_items, va_items = [], []
    for cat, ex in train_pool:
        h = int(hashlib.sha256(("val-" + ex.session_id).encode()).hexdigest(), 16) % 20
        (va_items if h < 3 else tr_items).append((cat, ex))

    tr_ex = [ex for _, ex in tr_items]
    va_ex = [ex for _, ex in va_items]
    ho_ex = [ex for _, ex in heldout]
    print(f"Examples: train={len(tr_ex)}, val={len(va_ex)}, heldout={len(ho_ex)}", flush=True)

    gt = fit_or_load_tfidf([], 0)
    Xtr, ytr = map(torch.from_numpy, pack_arch("nca1d", tr_ex, L2I, "tfidf_svd128", gt))
    Xva, yva = map(torch.from_numpy, pack_arch("nca1d", va_ex, L2I, "tfidf_svd128", gt))
    Xho, yho = map(torch.from_numpy, pack_arch("nca1d", ho_ex, L2I, "tfidf_svd128", gt))
    obs_c = Xtr.shape[1]

    torch.manual_seed(0)
    model = GraphNCA(obs_c, hidden_c=16, n_classes=len(LABELS), steps=8, rule_depth=2, seq_len=5, topology="tree").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = 5000
    bs = 512
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)

    Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)
    best_val = -1.0
    best_state = None
    t0 = time.time()
    n_tr = len(ytr_d)

    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n_tr, (bs,), device=dev)
        logits = model(Xtr_d[idx])
        loss = F.cross_entropy(logits, ytr_d[idx], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            vm, _ = eval_model(model, Xva, yva, dev)
            if vm["top1"] > best_val:
                best_val = vm["top1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"step={step:4d} loss={loss.item():.4f} val_top1={vm['top1']:.2f}% gate={vm['gate_acc']:.2f}%", flush=True)

    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_CKPT)
    print(f"Saved {OUT_CKPT} (best_val={best_val:.2f}%, elapsed={time.time()-t0:.1f}s)", flush=True)

    # Evaluate both checkpoints on heldout
    report = {"n_train": len(tr_ex), "n_val": len(va_ex), "n_heldout": len(ho_ex), "models": {}}
    ho_cats = [cat for cat, _ in heldout]

    for name, ckpt in [("hermes_grokking", HERMES_CKPT), ("bfcl_split_trained", OUT_CKPT)]:
        m = GraphNCA(obs_c, hidden_c=16, n_classes=len(LABELS), steps=8, rule_depth=2, seq_len=5, topology="tree").to(dev)
        st = torch.load(ckpt, map_location="cpu")
        if isinstance(st, dict) and "state" in st:
            st = st["state"]
        m.load_state_dict(st)
        overall, preds = eval_model(m, Xho, yho, dev)

        groups = {
            "single_turn_ast": set(SINGLE_AST_CATS),
            "irrelevance": set(IRREL_CATS),
            "live_relevance": {"live_relevance"},
            "multi_turn": set(MULTI_TURN_CATS),
        }
        by_group = {}
        for gname, gcats in groups.items():
            mask = torch.tensor([c in gcats for c in ho_cats])
            if mask.any():
                yg = yho[mask]
                pg = preds[mask]
                is_r = yg == RESPOND_IDX
                by_group[gname] = {
                    "n": int(mask.sum().item()),
                    "top1": round(float((pg == yg).float().mean().item() * 100.0), 2),
                    "gate_acc": round(float(((pg == RESPOND_IDX) == is_r).float().mean().item() * 100.0), 2),
                    "tool_pass_rate": round(float((pg[~is_r] != RESPOND_IDX).float().mean().item() * 100.0), 2) if (~is_r).any() else None,
                    "respond_recall": round(float((pg[is_r] == RESPOND_IDX).float().mean().item() * 100.0), 2) if is_r.any() else None,
                }
        report["models"][name] = {"checkpoint": str(ckpt), "overall_heldout": overall, "by_group": by_group}

    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
