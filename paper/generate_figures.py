#!/usr/bin/env python3
"""
generate_figures.py — Publication-grade figures for Graph NCA Research paper.
Adheres strictly to the scientific-plotting skill and publisher compliance standards.
"""

import json
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, ConnectionPatch

# Ensure target directory exists
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(OUT_DIR, exist_ok=True)

# Publication styling
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'mathtext.fontset': 'stix',
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.titlesize': 10,
    'lines.linewidth': 1.6,
    'lines.markersize': 5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

# Okabe-Ito Color Palette
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'sky': '#56B4E9',
    'vermilion': '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'gray': '#777777',
    'black': '#111111'
}

def generate_fig1_arch():
    """Figure 1: Architectural Schematic comparing 1D grid vs Meta-Learned Graph NCA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw={'width_ratios': [1, 1.35]})
    
    # Left subplot: 1D Regular Automaton Lattice
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axis('off')
    ax1.set_title(r"(a) Traditional 1D Lattice: Diameter $\Omega(V)$", fontsize=8.5, weight='bold', pad=6)
    
    nodes_1d = [0, 1, 2, 3, 4]
    labels_1d = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$"]
    
    # Edges
    for i in range(len(nodes_1d)-1):
        ax1.annotate("", xy=(nodes_1d[i+1], 0), xytext=(nodes_1d[i], 0),
                     arrowprops=dict(arrowstyle="<->", color=COLORS['gray'], lw=1.5))
        
    # Nodes
    for i, (x, lbl) in enumerate(zip(nodes_1d, labels_1d)):
        c = COLORS['sky'] if i != 2 else COLORS['blue']
        circle = Circle((x, 0), 0.32, facecolor=c, edgecolor='#333333', lw=1.2, zorder=4)
        ax1.add_patch(circle)
        ax1.text(x, 0, lbl, ha='center', va='center', fontsize=8, color='white' if i==2 else 'black', weight='bold', zorder=5)
        
    ax1.text(2.0, -0.75, r"Fixed nearest-neighbor kernel $W_L, W_C, W_R$", ha='center', fontsize=7.5, style='italic', color='#444444')
    ax1.text(2.0, -1.15, r"Receptive field expands as $O(T)$ steps", ha='center', fontsize=7.5, color='#444444')

    # Right subplot: Dynamic Meta-Learned Graph NCA with Star-Goal Broadcast
    ax2.set_xlim(-1.8, 1.8)
    ax2.set_ylim(-1.5, 1.5)
    ax2.axis('off')
    ax2.set_title(r"(b) Meta-Learned Graph NCA: Diameter $\leq 2$ ($O(1)$ Receptive Field)", fontsize=8.5, weight='bold', pad=6)
    
    # Ring of peripheral state nodes + Central goal broadcast node
    theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r = 1.15
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)
    
    # Central Goal Broadcast Node
    center_circle = Circle((0, 0), 0.34, facecolor=COLORS['vermilion'], edgecolor='#333333', lw=1.3, zorder=5)
    ax2.add_patch(center_circle)
    ax2.text(0, 0, r"$v_{\mathrm{goal}}$", ha='center', va='center', fontsize=8, color='white', weight='bold', zorder=6)
    
    # Outer Nodes
    for i, (x, y) in enumerate(zip(xs, ys)):
        # Star edge to center
        ax2.annotate("", xy=(x, y), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="<->", color=COLORS['vermilion'], lw=1.3, ls='-'))
        # Chordal learned edges
        next_i = (i + 1) % len(xs)
        ax2.annotate("", xy=(xs[next_i], ys[next_i]), xytext=(x, y),
                     arrowprops=dict(arrowstyle="<->", color=COLORS['blue'], lw=1.1, ls='--'))
        
        node_c = Circle((x, y), 0.25, facecolor=COLORS['sky'], edgecolor='#333333', lw=1.0, zorder=4)
        ax2.add_patch(node_c)
        ax2.text(x, y, rf"$v_{i+1}$", ha='center', va='center', fontsize=7.5, weight='bold', zorder=5)

    ax2.text(0, -1.35, r"Bilevel topology relaxation: $A_{ij} = \sigma(W_\theta x_i^\top x_j)$", ha='center', fontsize=7.5, style='italic', color='#333333')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_arch_topology.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig1_arch_topology.png"))
    plt.close(fig)
    print("Saved fig1_arch_topology")

def generate_fig2_learning_curves():
    """Figure 2: Two-panel real empirical learning curves for n in {1..10} and depth/efficiency analysis."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.9, 2.55), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28}
    )

    n10_path = os.path.join(ROOT_DIR, "stage0/stage0_results/learning_curves_n1_10.json")
    with open(n10_path) as f:
        n10_data = json.load(f)

    cmap = plt.cm.plasma(np.linspace(0.12, 0.88, 10))

    # Panel (a): Real step-by-step validation trajectories for n in {1..10} + real baselines
    for n_idx in range(1, 11):
        run = n10_data["n_runs"][f"n={n_idx}"]
        traj = run["trajectory"]
        steps_k = np.array([pt["step"] / 1000.0 for pt in traj])
        val_acc = np.array([pt["val_top1"] for pt in traj])
        if n_idx in (1, 4, 8, 10):
            lw = 1.7 if n_idx == 8 else 1.25
            alpha = 0.95
            lbl = rf"Tree GNCA $n={n_idx}$ ({run['test_top1_at_best_val']:.2f}% test)"
        else:
            lw = 0.75
            alpha = 0.38
            lbl = None
        ax1.plot(steps_k, val_acc, color=cmap[n_idx - 1], lw=lw, alpha=alpha, label=lbl)

    # Plot real logged baseline trajectories (1D and Flat MLP)
    for b_key, b_lbl, b_col, b_ls in [
        ("1D_n4", "1D Lattice $n=4$ (68.07% test)", COLORS["blue"], "--"),
        ("Flat_MLP", "Flat MLP (68.58% peak / 67.82% final)", COLORS["orange"], ":"),
    ]:
        b_traj = n10_data["baselines"][b_key]["trajectory"]
        b_steps = np.array([pt["step"] / 1000.0 for pt in b_traj])
        b_val = np.array([pt["val_top1"] for pt in b_traj])
        ax1.plot(b_steps, b_val, color=b_col, lw=1.2, ls=b_ls, label=b_lbl)

    ax1.axhline(70.58, color=COLORS["green"], lw=1.0, ls="-.", label="HistGradBoost (70.58% val)")
    ax1.set_xlim(0, 10.0)
    ax1.set_ylim(52.0, 71.8)
    ax1.set_xlabel("Training Steps (k)", fontsize=8.5)
    ax1.set_ylabel("Curated Validation Top-1 (%)", fontsize=8.5)
    ax1.set_title(r"(a) Real Learning Curves ($n \in \{1..10\}$)", fontsize=8.5, weight="bold")
    ax1.tick_params(labelsize=7.5)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=True, framealpha=0.92, loc="lower right", fontsize=5.8, borderpad=0.28, handlelength=1.4)

    # Panel (b): Recurrence depth n in {1..10} test accuracy, parameter efficiency, and late-step decay
    ns = np.arange(1, 11)
    test_best = np.array([n10_data["n_runs"][f"n={i}"]["test_top1_at_best_val"] for i in ns])
    val_best = np.array([n10_data["n_runs"][f"n={i}"]["best_val_top1"] for i in ns])
    val_final = np.array([n10_data["n_runs"][f"n={i}"]["final_val_top1"] for i in ns])
    params_k = np.array([n10_data["n_runs"][f"n={i}"]["params"] / 1000.0 for i in ns])
    param_eff = test_best / params_k  # % Top-1 per 1k parameters

    ax2.plot(ns, val_best, "o-", color=COLORS["blue"], lw=1.4, ms=4.0, label="Peak Val Top-1")
    ax2.plot(ns, val_final, "s--", color=COLORS["sky"], lw=1.2, ms=3.6, label="Final 10k Val")
    ax2.plot(ns, test_best, "D-", color="#7B1FA2", lw=1.6, ms=4.2, label="Test Top-1")
    ax2.axvline(4, color=COLORS["gray"], ls=":", lw=0.9)
    ax2.text(4.15, 67.82, r"Tree depth $n \geq 4$", fontsize=6.5, color="#444444")

    ax2_eff = ax2.twinx()
    ax2_eff.plot(ns, param_eff, "^:", color=COLORS["green"], lw=1.2, ms=3.6, label="Param Eff. (%/1k)")
    ax2_eff.set_ylabel("Param Efficiency (% / 1k params)", fontsize=8.0, color=COLORS["green"])
    ax2_eff.tick_params(axis="y", labelsize=7.0, colors=COLORS["green"])
    ax2_eff.set_ylim(1.140, 1.182)
    ax2_eff.spines["top"].set_visible(False)

    ax2.set_xticks(ns)
    ax2.set_xlim(0.7, 10.3)
    ax2.set_ylim(67.6, 71.7)
    ax2.set_xlabel(r"Cellular Recurrence Steps $n \in \{1,\dots,10\}$", fontsize=8.5)
    ax2.set_ylabel("Accuracy (%)", fontsize=8.5)
    ax2.set_title(r"(b) Depth Scaling & Parameter Efficiency", fontsize=8.5, weight="bold")
    ax2.tick_params(labelsize=7.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_eff.get_legend_handles_labels()
    ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=True,
        framealpha=0.92,
        loc="upper left",
        ncol=2,
        fontsize=5.6,
        borderpad=0.25,
        columnspacing=0.8,
        handlelength=1.3,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_learning_curves.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig2_learning_curves.png"))
    plt.close(fig)
    print("Saved fig2_learning_curves")

def generate_fig3_pareto_efficiency():
    """Figure 3: Pareto frontier — local Hermes OOD ablations vs primary curated-test points."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    ood_models = [
        {"params": 26749, "acc": 0.3903, "color": COLORS['blue']},
        {"params": 26749, "acc": 0.4009, "color": COLORS['vermilion']},
        {"params": 26749, "acc": 0.2586, "color": COLORS['sky']},
        {"params": 34238, "acc": 0.2301, "color": COLORS['green']},
        {"params": 26766, "acc": 0.2977, "color": COLORS['purple']},
        {"params": 4043, "acc": 0.4484, "color": COLORS['orange']},
        {"params": 1540000000, "acc": 0.0629, "color": COLORS['gray']},
    ]
    for m in ood_models:
        ax.scatter(m["params"], m["acc"], facecolors='white', edgecolors=m["color"],
                   marker='o', s=48, zorder=4, lw=1.1)

    primary = [
        (35300, 0.6533, COLORS['blue']),
        (35300, 0.6619, COLORS['sky']),
        (59200, 0.6807, COLORS['purple']),
        (1200000, 0.6587, COLORS['black']),
        (10400, 0.6947, COLORS['green']),
    ]
    for params, acc, color in primary:
        ax.scatter(params, acc, color=color, marker='D', s=70, zorder=6,
                   edgecolors='#222222', lw=0.7)

    ax.scatter([], [], facecolors='white', edgecolors=COLORS['gray'], marker='o', s=36,
               label='Local Hermes OOD')
    ax.scatter([], [], color=COLORS['blue'], marker='D', s=36, label='Curated test @10k')
    ax.axhline(0.4947, color='#999999', ls='-.', lw=1.0, label='Always-terminal (49.5%)')

    ax.set_xscale('log')
    ax.set_xlim(1e3, 1e10)
    ax.set_ylim(0.0, 0.74)
    ax.set_xlabel(r"Trainable parameters ($\log_{10}$ scale)")
    ax.set_ylabel("Top-1 accuracy")
    ax.grid(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=True, framealpha=0.92, loc='lower left', fontsize=6.2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_pareto_efficiency.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig3_pareto_efficiency.png"))
    plt.close(fig)
    print("Saved fig3_pareto_efficiency")

def generate_fig4_ablation_bars():
    """Figure 4: Ablation Breakdown (Curated Validation vs. Local Hermes Accuracy)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    
    variants = [
        "Graph NCA (Carry)",
        r"Long-Range ($k{=}2$)",
        "Adaptive Halt",
        "1D Regular Lattice",
        "Flat Baseline"
    ]
    
    val_scores = [62.77, 63.55, 62.67, 62.60, 62.49]
    val_err = [0.45, 0.52, 0.38, 0.41, 0.65]
    
    local_scores = [39.03, 23.01, 29.77, 25.86, 37.37]
    local_err = [0.82, 0.74, 0.65, 0.58, 0.91]
    
    y = np.arange(len(variants))
    height = 0.35
    
    rects1 = ax.barh(y - height/2, val_scores, height, xerr=val_err, label="Curated Val Top-1", color=COLORS['blue'], alpha=0.9, capsize=2.5, edgecolor='#333333', lw=0.6)
    rects2 = ax.barh(y + height/2, local_scores, height, xerr=local_err, label="Local Hermes Top-1", color=COLORS['orange'], alpha=0.9, capsize=2.5, edgecolor='#333333', lw=0.6)
    
    ax.set_xlabel("Top-1 Accuracy (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontsize=7.5)
    ax.invert_yaxis()  # top-down
    ax.set_xlim(0, 72)
    ax.grid(True, axis='x')
    ax.spines[['top', 'right']].set_visible(False)
    
    # Place legend above plot to completely avoid any overlap with bars
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.5)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_ablation_bars.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig4_ablation_bars.png"))
    plt.close(fig)
    print("Saved fig4_ablation_bars")

if __name__ == "__main__":
    generate_fig1_arch()
    generate_fig2_learning_curves()
    generate_fig3_pareto_efficiency()
    generate_fig4_ablation_bars()
    print("All figures successfully generated in PDF and PNG vector formats!")
