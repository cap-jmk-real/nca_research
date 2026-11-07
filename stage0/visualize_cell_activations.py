#!/usr/bin/env python3
"""visualize_cell_activations.py: Illustrative examples of how the trained Tree GNCA routes
real held-out Hermes test turns, compared with the decision path of a depth-4 decision tree
trained on the identical packed observation tensor.

For 4 correctly classified test turns with different predicted tools we plot:
  (a) the real 5-cell tree graph (goal cell + last 4 trace steps), node colour = exact additive
      contribution of each cell's final hidden state to the predicted-class logit
      (tool_head is linear over the flattened (channel, cell) state, so the decomposition is exact);
  (b) per-cell update magnitude ||h_t - h_{t-1}||_2 over the K=8 cellular steps;
  (c) the root-to-leaf path of a depth-4 sklearn DecisionTreeClassifier on the same inputs.

Outputs: paper/figures/fig5_cell_activation_examples.pdf and
         stage0/stage0_results/cell_activation_examples.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from sklearn.tree import DecisionTreeClassifier

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

from common import load_split  # noqa: E402
from nca_text.model import GraphNCA  # noqa: E402
from obs_pack import GOAL_TF_DIM, fit_or_load_tfidf, pack_arch  # noqa: E402

RES = REPO_DIR / "stage0" / "stage0_results"
CKPT = RES / "checkpoints" / "best_graph_nca_grokking.pt"
OUT_PDF = REPO_DIR / "paper" / "figures" / "fig5_cell_activation_examples.pdf"
OUT_JSON = RES / "cell_activation_examples.json"
LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
V = 5
TARGET_CLASSES = ["terminal", "read_file", "patch", "respond", "search_files", "write_file", "todo"]
DT_DEPTH = 4


def cell_names(ex) -> list[str]:
    steps = ex.steps[-(V - 1):]
    names = ["Goal"] + ["(empty)"] * (V - 1)
    start = V - len(steps)
    for i, (tool, _) in enumerate(steps):
        lag = len(steps) - i
        names[start + i] = f"t-{lag}: {tool}"
    return names


def feat_name(idx: int, obs_c: int) -> str:
    ch, v = divmod(idx, V)
    cell = "goal" if v == 0 else f"cell{v}"
    if v == 0 and ch < GOAL_TF_DIM - 4:
        return f"goal TF-IDF/SVD[{ch}]"
    if v == 0 and ch < GOAL_TF_DIM:
        return f"goal struct[{ch - (GOAL_TF_DIM - 4)}]"
    if GOAL_TF_DIM <= ch < GOAL_TF_DIM + len(LABELS):
        return f"{cell} tool={LABELS[ch - GOAL_TF_DIM]}"
    if ch == obs_c - 1:
        return f"{cell} occupied"
    return f"{cell} hash[{ch}]"


def tree_positions() -> dict[int, tuple[float, float]]:
    # parent(i) = (i-1)//2 : 0 -> {1, 2}, 1 -> {3, 4}
    return {0: (0.5, 1.0), 1: (0.25, 0.55), 2: (0.80, 0.55), 3: (0.0, 0.08), 4: (0.52, 0.08)}


def draw_gnca(ax, names, contrib, upd_total, vmax):
    pos = tree_positions()
    for i in range(1, V):
        p = (i - 1) // 2
        ax.plot([pos[p][0], pos[i][0]], [pos[p][1], pos[i][1]], color="0.55", lw=1.2, zorder=1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    smax = max(upd_total.max(), 1e-6)
    for i in range(V):
        x, y = pos[i]
        size = 110 + 330 * upd_total[i] / smax
        ax.scatter([x], [y], s=size, c=[cmap(norm(contrib[i]))], edgecolors="k", linewidths=0.8, zorder=2)
        ax.text(x, y, f"{contrib[i]:+.1f}", ha="center", va="center", fontsize=5.4, zorder=3,
                color="white" if abs(contrib[i]) > 0.6 * vmax else "black")
        lbl = names[i] if len(names[i]) <= 22 else names[i][:21] + "…"
        lbl = lbl.replace(": ", ":\n")
        off = {0: (0.14, 0.0, "left"), 1: (-0.12, 0.0, "right"), 2: (0.12, 0.0, "left"),
               3: (0.0, -0.27, "center"), 4: (0.0, -0.27, "center")}[i]
        ax.text(x + off[0], y + off[1], lbl, ha=off[2], va="center", fontsize=5.6, linespacing=0.95)
    ax.set_xlim(-0.42, 1.25)
    ax.set_ylim(-0.25, 1.15)
    ax.axis("off")


def draw_dt_path(ax, dt, x_row, obs_c, gold):
    t = dt.tree_
    node, depth = 0, 0
    path = []
    while t.children_left[node] != -1:
        f, thr = t.feature[node], t.threshold[node]
        go_left = x_row[f] <= thr
        path.append((node, f, thr, go_left))
        node = t.children_left[node] if go_left else t.children_right[node]
        depth += 1
    leaf_cls = LABELS[int(np.argmax(t.value[node][0]))]
    leaf_p = float(np.max(t.value[node][0]) / np.sum(t.value[node][0]))
    # full binary skeleton of depth DT_DEPTH, highlight taken path
    coords = {}

    def place(n, d, lo, hi):
        coords[n] = ((lo + hi) / 2, 1.0 - d / (DT_DEPTH + 0.6))
        if t.children_left[n] != -1:
            mid = (lo + hi) / 2
            place(t.children_left[n], d + 1, lo, mid)
            place(t.children_right[n], d + 1, mid, hi)

    place(0, 0, 0.0, 1.0)
    on_path = {p[0] for p in path} | {node}
    for n, (x, y) in coords.items():
        for c in (t.children_left[n], t.children_right[n]):
            if c != -1:
                hot = n in on_path and c in on_path
                ax.plot([x, coords[c][0]], [y, coords[c][1]], color="C3" if hot else "0.8",
                        lw=1.6 if hot else 0.6, zorder=1)
    for n, (x, y) in coords.items():
        hot = n in on_path
        leaf = t.children_left[n] == -1
        ax.scatter([x], [y], s=28 if hot else 9, marker="s" if leaf else "o",
                   c="C3" if hot else "0.75", zorder=2)
    for k, (n, f, thr, left) in enumerate(path):
        x, y = coords[n]
        rule = f"{feat_name(f, obs_c)} {'≤' if left else '>'} {thr:.2f}"
        ax.text(1.02, 1.0 - k / (DT_DEPTH + 0.6), rule, fontsize=5.4, va="center", ha="left",
                transform=ax.transData, color="C3")
    ok = leaf_cls == gold
    ax.text(1.02, 1.0 - len(path) / (DT_DEPTH + 0.6),
            f"leaf → {leaf_cls} ({100 * leaf_p:.0f}%)" + ("" if ok else "  ✗ wrong"),
            fontsize=5.6, va="center", ha="left", fontweight="bold", color="black" if ok else "C3")
    ax.set_xlim(-0.03, 2.25)
    ax.set_ylim(-0.12, 1.08)
    ax.axis("off")
    return leaf_cls, [feat_name(p[1], obs_c) for p in path]


def main() -> None:
    torch.manual_seed(0)
    splits = {s: load_split("curated", s) for s in ("train", "test")}
    gt = fit_or_load_tfidf(splits["train"], 0)
    Xtr, ytr = pack_arch("nca1d", splits["train"], L2I, "tfidf_svd128", gt)
    Xte, yte = pack_arch("nca1d", splits["test"], L2I, "tfidf_svd128", gt)
    test_ex = [ex for ex in splits["test"] if ex.label in L2I]
    obs_c = Xte.shape[1]

    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    steps = state["step_embed"].shape[0]
    model = GraphNCA(obs_c=obs_c, hidden_c=16, n_classes=len(LABELS), steps=steps,
                     rule_depth=2, seq_len=V, topology="tree")
    model.load_state_dict(state)
    model.eval()

    X = torch.from_numpy(Xte).float()
    with torch.no_grad():
        hT, traj = model.evolve(X)
        logits = model.tool_head(hT.flatten(1))
    probs = torch.softmax(logits, -1).numpy()
    pred = probs.argmax(-1)
    acc = float((pred == yte).mean() * 100)
    print(f"Loaded {CKPT.name}: K={steps} steps, test top-1 = {acc:.2f}% on {len(yte)} turns")

    W = model.tool_head.weight.detach().numpy().reshape(len(LABELS), 16, V)  # (k, c, v)
    H = hT.numpy()  # (N, 16, V)
    T = torch.stack(traj, 1).numpy()  # (N, K+1, 16, V)
    upd = np.linalg.norm(np.diff(T, axis=1), axis=2)  # (N, K, V)

    dt = DecisionTreeClassifier(max_depth=DT_DEPTH, random_state=0)
    dt.fit(Xtr.reshape(len(Xtr), -1), ytr)
    Xte_flat = Xte.reshape(len(Xte), -1)
    dt_pred = dt.predict(Xte_flat)
    dt_acc = float((dt_pred == yte).mean() * 100)
    print(f"Depth-{DT_DEPTH} decision tree test top-1 = {dt_acc:.2f}%")

    chosen = []
    used = set()
    for cls in TARGET_CLASSES:
        k = L2I[cls]
        cand = [i for i in range(len(yte))
                if pred[i] == k and yte[i] == k and len(test_ex[i].steps) >= 3
                and 15 <= len(test_ex[i].goal or "") <= 95]
        if not cand:
            continue
        # prefer turns where the DT agrees, then highest GNCA confidence
        cand.sort(key=lambda i: (dt_pred[i] == k, probs[i, k]), reverse=True)
        chosen.append(cand[0])
        used.add(cls)
        if len(chosen) == 4:
            break

    contribs = [np.einsum("cv,cv->v", W[pred[i]], H[i]) for i in chosen]
    vmax = max(np.abs(c).max() for c in contribs)
    umax = max(upd[i].max() for i in chosen)

    fig = plt.figure(figsize=(7.1, 1.95 * len(chosen) + 0.3))
    gs = fig.add_gridspec(len(chosen), 3, width_ratios=[1.05, 1.0, 1.75], wspace=0.12, hspace=0.78)
    records = []
    for r, i in enumerate(chosen):
        ex = test_ex[i]
        names = cell_names(ex)
        k = int(pred[i])
        ax0 = fig.add_subplot(gs[r, 0])
        draw_gnca(ax0, names, contribs[r], upd[i].sum(0), vmax)
        top3 = np.argsort(-probs[i])[:3]
        goal = ex.goal if len(ex.goal) <= 100 else ex.goal[:99] + "…"
        ax0.text(-0.02, 1.13, f"({chr(97 + r)}) Goal: “{goal}”   [gold: {LABELS[yte[i]]}]",
                 transform=ax0.transAxes, fontsize=6.6, ha="left", va="bottom", fontweight="bold")

        ax1 = fig.add_subplot(gs[r, 1])
        im = ax1.imshow(upd[i].T, aspect="auto", cmap="viridis", vmin=0, vmax=umax)
        ax1.set_yticks(range(V))
        ax1.set_yticklabels(["goal", "c1", "c2", "c3", "c4"], fontsize=5.6)
        ax1.set_xticks(range(steps))
        ax1.set_xticklabels([str(s + 1) for s in range(steps)], fontsize=5.6)
        ax1.set_xlabel("cellular step $t$", fontsize=6, labelpad=1)
        ax1.set_title("GNCA top-3: " + ", ".join(f"{LABELS[j]} {100 * probs[i, j]:.0f}%" for j in top3),
                      fontsize=5.8, pad=2)

        ax2 = fig.add_subplot(gs[r, 2])
        leaf_cls, rules = draw_dt_path(ax2, dt, Xte_flat[i], obs_c, LABELS[yte[i]])
        ax2.set_title(f"Depth-{DT_DEPTH} decision tree path", fontsize=5.8, pad=2, loc="right")

        records.append({
            "test_index": int(i), "goal": ex.goal, "gold": LABELS[yte[i]],
            "trace": [t for t, _ in ex.steps[-(V - 1):]], "cells": names,
            "gnca_pred": LABELS[k], "gnca_top3": {LABELS[j]: round(float(probs[i, j]), 4) for j in top3},
            "logit_contribution_per_cell": [round(float(c), 3) for c in contribs[r]],
            "update_norm_per_step_cell": np.round(upd[i], 4).tolist(),
            "dt_pred": leaf_cls, "dt_path_features": rules,
        })

    cax = fig.add_axes([0.385, 0.035, 0.17, 0.008])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5)
    cb.set_label(r"update norm $\|h_t-h_{t-1}\|_2$", fontsize=5.6, labelpad=1)
    cax2 = fig.add_axes([0.07, 0.035, 0.17, 0.008])
    sm = plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), cmap="RdBu_r")
    cb2 = fig.colorbar(sm, cax=cax2, orientation="horizontal")
    cb2.ax.tick_params(labelsize=5)
    cb2.set_label("cell contribution to predicted logit", fontsize=5.6, labelpad=1)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.965, bottom=0.085)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=200, bbox_inches="tight")
    OUT_JSON.write_text(json.dumps({"checkpoint": CKPT.name, "steps": steps, "gnca_test_top1": round(acc, 2),
                                    "dt_depth": DT_DEPTH, "dt_test_top1": round(dt_acc, 2),
                                    "examples": records}, indent=2))
    print(json.dumps(records, indent=1)[:4000])


if __name__ == "__main__":
    main()
