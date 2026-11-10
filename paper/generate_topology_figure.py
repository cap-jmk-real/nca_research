#!/usr/bin/env python3
"""generate_topology_figure.py: Figure 1, visual overview of the information-flow topology of
every algorithm employed in the paper.

Top row: GNCA spatial topologies (V=5 cells: goal + last 4 tool turns).
Bottom row: Baseline architectures (Flat MLP, Random Forest, HistGradientBoosting)
and the Receptive Field Hop Distance matrix comparing communication efficiency.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca_text.model import build_adjacency  # noqa: E402

OUT = REPO / "paper" / "figures" / "fig1_topologies"
META = json.load(open(REPO / "stage0" / "stage0_results" / "e3_meta_adjacency_audit.json"))
A_META = np.array(META["trained_meta_graphs"][0]["adjacency_row_softmax"])

# Accessible Okabe-Ito colors
GOAL_C = "#D55E00"   # Vermilion (Goal intent)
CELL_C = "#56B4E9"   # Sky blue (Trace turns)
EDGE_C = "#444444"
TREE_C = "#009E73"   # Green (Random Forest)
HGB_C  = "#0072B2"   # Blue (HistGradientBoosting)

NAMES = ["goal", "$t{-}4$", "$t{-}3$", "$t{-}2$", "$t{-}1$"]
THR = 0.01


def adj(topo: str) -> np.ndarray:
    return A_META if topo == "meta" else build_adjacency(topo, 5).numpy()


def hops_from_goal(A: np.ndarray) -> list[int]:
    dist = [-1] * 5
    dist[0] = 0
    q = deque([0])
    while q:
        j = q.popleft()
        for i in range(5):
            if A[i, j] > THR and dist[i] < 0:
                dist[i] = dist[j] + 1
                q.append(i)
    return dist


LAYOUTS = {
    "1d":   {0: (0.5, 0.5), 1: (1.35, 0.5), 2: (2.2, 0.5), 3: (3.05, 0.5), 4: (3.9, 0.5)},
    "star": {0: (2.2, 0.5), 1: (1.0, 1.15), 2: (3.4, 1.15), 3: (1.0, -0.15), 4: (3.4, -0.15)},
    "tree": {0: (2.2, 1.25), 1: (1.2, 0.45), 2: (3.2, 0.45), 3: (0.6, -0.35), 4: (1.8, -0.35)},
}
LAYOUTS["meta"] = LAYOUTS["tree"]


def draw_graph(ax, topo: str, title: str, subtitle: str) -> None:
    A = adj(topo)
    pos = LAYOUTS[topo]
    directed = (topo == "meta")
    drawn = set()

    # Draw edges
    if not directed:
        for i in range(5):
            for j in range(5):
                if i == j or A[i, j] <= THR:
                    continue
                key = tuple(sorted((i, j)))
                if key in drawn:
                    continue
                drawn.add(key)
                arr = FancyArrowPatch(
                    pos[j], pos[i],
                    arrowstyle="<|-|>",
                    mutation_scale=8,
                    color=EDGE_C,
                    lw=1.3,
                    shrinkA=13, shrinkB=13,
                    zorder=1,
                )
                ax.add_patch(arr)
    else:
        # For Meta-learned, display the key directed edges with weight >= 0.20
        edges_to_draw = [
            (0, 1, A[1, 0]),  # goal -> t-4
            (0, 2, A[2, 0]),  # goal -> t-3
            (1, 0, A[0, 1]),  # t-4 -> goal
            (1, 4, A[4, 1]),  # t-4 -> t-1
            (1, 3, A[3, 1]),  # t-4 -> t-2
        ]
        for src, dst, w in edges_to_draw:
            rad = 0.18 if (dst, src) in [(0, 1), (1, 0)] else 0.0
            arr = FancyArrowPatch(
                pos[src], pos[dst],
                arrowstyle="-|>",
                mutation_scale=7,
                connectionstyle=f"arc3,rad={rad}",
                color=EDGE_C,
                lw=0.8 + 2.8 * w,
                shrinkA=13, shrinkB=13,
                zorder=1,
            )
            ax.add_patch(arr)
            mx = (pos[src][0] + pos[dst][0]) / 2
            my = (pos[src][1] + pos[dst][1]) / 2
            dx = pos[dst][0] - pos[src][0]
            dy = pos[dst][1] - pos[src][1]
            d = np.hypot(dx, dy)
            nx, ny = -dy / d, dx / d
            ax.text(
                mx + 0.16 * nx, my + 0.16 * ny, f"{w:.2f}",
                fontsize=5.2, ha="center", va="center", color="#222222",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="#CCCCCC", lw=0.4, alpha=0.92),
                zorder=2,
            )

    # Draw nodes
    for k, (x, y) in pos.items():
        ax.scatter([x], [y], s=260, c=GOAL_C if k == 0 else CELL_C, edgecolors="#222222", linewidths=0.9, zorder=3)
        ax.text(
            x, y, NAMES[k], ha="center", va="center", fontsize=5.8, zorder=4,
            color="white" if k == 0 else "black", fontweight="bold" if k == 0 else "normal",
        )

    ax.set_title(title, fontsize=7.2, fontweight="bold", pad=5)
    ax.text(
        0.5, -0.06, subtitle,
        transform=ax.transAxes, ha="center", va="top", fontsize=5.3,
        color="#333333", linespacing=1.15,
    )

    ax.set_xlim(0.0, 4.4)
    ax.set_ylim(-0.6, 1.55)
    ax.axis("off")


def box(ax, x, y, w, h, text, fc="#F6F6F6", fs=5.2, ec="#333333"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        fc=fc, ec=ec, lw=0.6, zorder=2,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, p, q, lw=0.8, color=EDGE_C):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw, zorder=1,
    ))


def mini_tree(ax, x0, y0, s, depth=3, hot=None, color="#009E73"):
    hot = hot or []
    nodes = {(0, 0): (x0, y0)}
    for d in range(1, depth + 1):
        for k in range(2 ** d):
            px, py = nodes[(d - 1, k // 2)]
            nodes[(d, k)] = (px + (-1 if k % 2 == 0 else 1) * s * 0.5 ** d * 1.5, py - s * 0.50)
    for (d, k), (x, y) in nodes.items():
        if d:
            px, py = nodes[(d - 1, k // 2)]
            on = (d, k) in hot and (d - 1, k // 2) in hot + [(0, 0)]
            ax.plot([px, x], [py, y], color=color if on else "#D0D0D0", lw=1.2 if on else 0.5, zorder=1)
    for (d, k), (x, y) in nodes.items():
        on = (d, k) in hot or d == 0
        ax.scatter([x], [y], s=10 if d < depth else 7, marker="o" if d < depth else "s",
                   c=color if on else "#D0D0D0", zorder=2)


def draw_mlp(ax):
    ax.set_xlim(0, 4.4)
    ax.set_ylim(-0.6, 1.55)
    for k in range(5):
        y = 1.35 - 0.28 * k
        ax.scatter([0.35], [y], s=90, c=GOAL_C if k == 0 else CELL_C, edgecolors="k", linewidths=0.6, zorder=3)
        ax.text(0.35, y, NAMES[k], fontsize=4.2, ha="center", va="center", zorder=4,
                color="white" if k == 0 else "black")
        arrow(ax, (0.55, y), (1.05, 0.78), lw=0.5)
    box(ax, 1.45, 0.78, 0.7, 1.45, "concat\n$5{\\times}146$\n$=730$", fs=4.9)
    box(ax, 2.35, 0.78, 0.55, 1.05, "76", fc="#E8E8F8")
    box(ax, 3.05, 0.78, 0.55, 1.05, "76", fc="#E8E8F8")
    box(ax, 3.75, 0.78, 0.55, 0.75, "13\ntools", fc="#FCE8D8", fs=4.8)
    for a, b in [(1.8, 2.07), (2.63, 2.77), (3.33, 3.47)]:
        arrow(ax, (a, 0.78), (b, 0.78))
    _mb = json.loads((Path(__file__).resolve().parents[1] / "stage0/stage0_results/review_1110/multibench.json").read_text())
    ax.set_title(f"(e) Flat MLP ({_mb['benchmarks']['curated']['models']['MLP']['n_params']/1e3:.1f}k params)", fontsize=7.2, fontweight="bold", pad=5)
    ax.text(
        0.5, -0.06,
        "No graph: all cells concatenated;\ncell identity is solely input offset",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.3, color="#333333", linespacing=1.15,
    )
    ax.axis("off")


def draw_rf(ax):
    ax.set_xlim(0, 4.4)
    ax.set_ylim(-0.6, 1.55)
    for n, x in enumerate([0.55, 1.55, 2.55]):
        hot = [(1, n % 2), (2, (n % 2) * 2 + 1), (3, ((n % 2) * 2 + 1) * 2)]
        mini_tree(ax, x, 1.45, 0.85, depth=3, hot=hot, color=TREE_C)
        # Dedicated collector arrow to majority vote
        arrow(ax, (x, 0.15), (3.4, 0.55), lw=0.5)
    ax.text(3.05, 0.95, "…", fontsize=9, ha="center")
    box(ax, 3.8, 0.55, 0.75, 0.55, "majority\nvote", fc="#FCE8D8", fs=4.9)
    _cc = json.loads((Path(__file__).resolve().parents[1] / "stage0/stage0_results/review_1110/cpu_cost.json").read_text())
    ax.set_title(f"(f) Random Forest ({_cc['RandomForest']['n_tree_nodes']/1e3:.0f}k nodes)", fontsize=7.2, fontweight="bold", pad=5)
    ax.text(
        0.5, -0.06,
        "Deep bootstrap trees voted in parallel;\neach split tests a single feature",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.3, color="#333333", linespacing=1.15,
    )
    ax.axis("off")


def draw_hgb(ax):
    ax.set_xlim(0, 4.4)
    ax.set_ylim(-0.6, 1.55)
    xs = [0.45, 1.45, 2.45]
    for n, x in enumerate(xs):
        mini_tree(ax, x, 1.05, 0.60, depth=2, hot=[(1, n % 2), (2, (n % 2) * 3)], color=HGB_C)
        ax.text(x, 0.35, f"$f_{n + 1}$", fontsize=5.5, ha="center", va="top")
        if n < 2:
            ax.text(x + 0.5, 0.35, "+", fontsize=7.5, ha="center", va="top")
    ax.text(2.95, 0.35, "+ … =", fontsize=6.0, ha="left", va="top")
    box(ax, 3.85, 0.35, 0.55, 0.45, "logit", fc="#FCE8D8", fs=4.9)

    # Clean residual arc strictly inside subplot boundaries
    ax.annotate("", xy=(1.45, 1.25), xytext=(0.45, 1.25),
                arrowprops=dict(arrowstyle="-|>", color=HGB_C, lw=0.7, connectionstyle="arc3,rad=-0.35"))
    ax.annotate("", xy=(2.45, 1.25), xytext=(1.45, 1.25),
                arrowprops=dict(arrowstyle="-|>", color=HGB_C, lw=0.7, connectionstyle="arc3,rad=-0.35"))
    ax.text(1.45, 1.42, "fit residual", fontsize=5.0, ha="center", va="center", color=HGB_C, fontweight="bold")

    ax.set_title("(g) HistGradientBoosting", fontsize=7.2, fontweight="bold", pad=5)
    ax.text(
        0.5, -0.06,
        "Shallow sequential additive trees;\neach tree fits prior pseudo-residuals",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.3, color="#333333", linespacing=1.15,
    )
    ax.axis("off")


def draw_hops(ax):
    rows = [("1D chain", "1d"), ("Star", "star"), ("Tree", "tree"), ("Meta (learned)", "meta")]
    M = np.array([hops_from_goal(adj(t)) for _, t in rows], float)
    ax.imshow(M, cmap="Oranges", vmin=0, vmax=4.6, aspect="auto")
    for r in range(M.shape[0]):
        for c in range(5):
            ax.text(c, r, f"{int(M[r, c])}", ha="center", va="center", fontsize=6.2,
                    color="white" if M[r, c] >= 3 else "black", fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(NAMES, fontsize=5.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r for r, _ in rows], fontsize=5.5)
    ax.tick_params(length=0)
    ax.set_title("(h) Goal Hop Distance", fontsize=7.2, fontweight="bold", pad=5)
    ax.text(
        0.5, -0.22,
        "Steps for goal intent to reach cell;\nStar & Tree achieve $O(1)$/$O(\\log V)$ hops",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.3, color="#333333", linespacing=1.15,
    )


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), gridspec_kw={"hspace": 0.58, "wspace": 0.20})
    draw_graph(axes[0, 0], "1d", "(a) 1D Lattice (TextNCA)", "Nearest-neighbor line: goal needs\n$V{-}1{=}4$ steps to reach latest call")
    draw_graph(axes[0, 1], "star", "(b) Star GNCA", "Central hub: goal broadcasts to\nevery cell in exactly 1 step")
    draw_graph(axes[0, 2], "tree", "(c) Tree GNCA", "Binary call tree: recent calls are\nsiblings with $O(\\log V)$ diameter")
    draw_graph(axes[0, 3], "meta", "(d) Meta-Learned GNCA", "Learned continuous edge weights;\nconverges to directed tree backbone")
    draw_mlp(axes[1, 0])
    draw_rf(axes[1, 1])
    draw_hgb(axes[1, 2])
    draw_hops(axes[1, 3])
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.10)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=250, bbox_inches="tight")
    print("hops:", {t: hops_from_goal(adj(t)) for t in ("1d", "star", "tree", "meta")})


if __name__ == "__main__":
    main()
