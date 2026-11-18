#!/usr/bin/env python3
"""Figure 2 of the revised paper: test top-1 per benchmark (mean +/- SD over 3 seeds), read from
stage0/stage0_results/review_1110/multibench.json. Okabe-Ito palette, vector PDF."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
mb = json.loads((REPO / "stage0/stage0_results/review_1110/multibench.json").read_text())["benchmarks"]
BENCH = [("curated", "Hermes"), ("bfcl_e1v2", "BFCL"), ("toolbench", "ToolBench"), ("apibank", "API-Bank")]
MODELS = [("RandomForest", "Random Forest", "#E69F00"), ("HistGradientBoosting", "HistGradBoost", "#56B4E9"),
          ("MLP", "MLP", "#009E73"), ("GraphNCA_tree_LS", "GNCA tree", "#D55E00"), ("GraphNCA_star_LS", "GNCA star", "#CC79A7")]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
x = np.arange(len(BENCH)); w = 0.16
for i, (m, lab, c) in enumerate(MODELS):
    mu = [mb[b]["models"][m]["top1_mean"] for b, _ in BENCH]
    sd = [mb[b]["models"][m]["top1_sd"] for b, _ in BENCH]
    ax.bar(x + (i - 2) * w, mu, w, yerr=sd, color=c, label=lab, capsize=1.5, error_kw={"lw": 0.6})
maj = [mb[b]["models"]["Majority"]["top1_mean"] for b, _ in BENCH]
for xi, mj in zip(x, maj):
    ax.hlines(mj, xi - 2.6 * w, xi + 2.6 * w, colors="k", linestyles=":", lw=0.8)
ax.set_xticks(x, [n for _, n in BENCH], fontsize=7)
ax.set_ylabel("Test top-1 accuracy (%)", fontsize=7)
ax.set_ylim(30, 92); ax.tick_params(labelsize=7)
ax.legend(fontsize=5.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.0, columnspacing=0.8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = REPO / "paper/figures/fig2_multibench_1110.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
print(out)
