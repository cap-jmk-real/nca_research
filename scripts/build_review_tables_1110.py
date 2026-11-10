#!/usr/bin/env python3
"""Build every number used in the revised paper from result files (no hand-typed numbers).

Inputs : stage0/stage0_results/review_1110/{multibench.json,correct_by_seed.npz,jacobian_audit.json,cpu_cost.json}
         stage0/stage0_results/reviewer_suite_summary.json (escalation calibration, factorial, contractive audit)
         stage0/stage0_results/full_history_benchmarks.json (earlier full-history Hermes protocol)
Outputs: stage0/stage0_results/review_1110/stats.json, paper/generated/{numbers.tex,tab_multibench.tex,tab_paired.tex}
Paired tests: example-level bootstrap with the 3 seeds kept together (per-example accuracy averaged
over seeds), so n = number of test examples, not seeds x examples. Holm correction over the full
family of pre-specified comparisons. Pooled effect: DerSimonian-Laird random-effects over benchmarks.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "stage0" / "stage0_results"
R11 = RES / "review_1110"
GEN = REPO / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
BENCH = [("curated", "Hermes"), ("bfcl_e1v2", "BFCL (E1v2 split)"), ("toolbench", "ToolBench"), ("apibank", "API-Bank")]
PRIMARY = "GraphNCA_tree_LS"           # pre-specified: the K=4 tree configuration described in Sec. 3
BASELINES = ["RandomForest", "HistGradientBoosting", "MLP"]
SHOW = ["Majority", "Linear", "RandomForest", "HistGradientBoosting", "MLP",
        "GraphNCA_1d_LS", "GraphNCA_star_LS", "GraphNCA_tree_LS", "GraphNCA_meta_LS"]
SHORT = {"RandomForest": "RF", "HistGradientBoosting": "HGB", "MLP": "MLP"}
NICE = {"Majority": "Majority class", "Linear": "Logistic regression", "RandomForest": "Random Forest",
        "HistGradientBoosting": "HistGradientBoosting", "MLP": "MLP (76 hidden)",
        "GraphNCA_1d_LS": "GNCA 1D lattice", "GraphNCA_star_LS": "GNCA star", "GraphNCA_tree_LS": "GNCA tree (primary)",
        "GraphNCA_meta_LS": "GNCA meta-learned"}


def cluster_boot(a: np.ndarray, b: np.ndarray, n_boot=10000, seed=0) -> dict:
    """a, b: (seeds, n) 0/1 correctness. Resample examples; seeds stay together."""
    d = a.mean(0) - b.mean(0)
    obs = d.mean() * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1) * 100
    p = (1 + np.sum(np.abs(boots - obs) >= abs(obs))) / (n_boot + 1)
    return {"diff_pp": float(obs), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "se_pp": float(boots.std(ddof=1)), "p": float(p), "n_examples": int(len(d)), "n_seeds": int(a.shape[0])}


def holm(ps: list[float]) -> list[float]:
    m = len(ps); order = np.argsort(ps); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def dersimonian_laird(est, se) -> dict:
    est, se = np.array(est), np.array(se); w = 1 / se**2
    fe = np.sum(w * est) / w.sum(); q = np.sum(w * (est - fe) ** 2); k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - np.sum(w**2) / w.sum()))
    wr = 1 / (se**2 + tau2); re = np.sum(wr * est) / wr.sum(); sre = math.sqrt(1 / wr.sum())
    z = re / sre; p = math.erfc(abs(z) / math.sqrt(2))
    return {"estimate_pp": re, "ci95": [re - 1.96 * sre, re + 1.96 * sre], "p": p, "tau2": tau2, "Q": q, "k": k}


def f(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    mb = json.loads((R11 / "multibench.json").read_text())
    cz = np.load(R11 / "correct_by_seed.npz")
    stats = {"source": "stage0/stage0_results/review_1110", "paired": {}, "pooled": {}}
    fam = []
    for b, _ in BENCH:
        for base in BASELINES:
            r = cluster_boot(cz[f"{b}/{PRIMARY}"], cz[f"{b}/{base}"])
            stats["paired"][f"{b}/{PRIMARY}-{base}"] = r; fam.append(f"{b}/{PRIMARY}-{base}")
    adj = holm([stats["paired"][k]["p"] for k in fam])
    for k, a in zip(fam, adj):
        stats["paired"][k]["p_holm"] = a
    for base in BASELINES:
        ks = [f"{b}/{PRIMARY}-{base}" for b, _ in BENCH]
        stats["pooled"][base] = dersimonian_laird([stats["paired"][k]["diff_pp"] for k in ks],
                                                  [stats["paired"][k]["se_pp"] for k in ks])
    # best GNCA variant chosen on validation (not test), per benchmark
    stats["val_selected"] = {}
    for b, _ in BENCH:
        ms = mb["benchmarks"][b]["models"]
        cands = [m for m in ms if m.startswith("GraphNCA_")]
        best = max(cands, key=lambda m: ms[m]["val_top1_mean"])
        stats["val_selected"][b] = {"model": best, "val": ms[best]["val_top1_mean"], "test": ms[best]["top1_mean"]}
    jac = json.loads((R11 / "jacobian_audit.json").read_text())
    cpu = json.loads((R11 / "cpu_cost.json").read_text())
    rss = json.loads((RES / "reviewer_suite_summary.json").read_text())["hermes_factorial_and_contractive"]
    esc = rss["escalation_calibration_3seed"]
    fh = json.loads((RES / "full_history_benchmarks.json").read_text())
    stats["escalation"] = {k: [e[k] for e in esc] for k in ("auroc_composite_tau", "auroc_entropy_only", "auroc_step_norm_only")}
    (R11 / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # ---------------- macros ----------------
    M = {}
    def put(name, val): M[name] = val
    nb = mb["benchmarks"]
    for b, short in [("curated", "Her"), ("bfcl_e1v2", "Bfcl"), ("toolbench", "Tb"), ("apibank", "Ab")]:
        ms = nb[b]["models"]
        put(f"N{short}Test", f"{nb[b]['n']['test']:,}".replace(",", "{,}"))
        put(f"N{short}Train", f"{nb[b]['n']['train']:,}".replace(",", "{,}"))
        for m, mm in [("GraphNCA_tree_LS", "Gnca"), ("RandomForest", "Rf"), ("HistGradientBoosting", "Hgb"), ("MLP", "Mlp"),
                      ("Majority", "Maj"), ("GraphNCA_star_LS", "Star")]:
            put(f"{short}{mm}", f(ms[m]["top1_mean"])); put(f"{short}{mm}Sd", f(ms[m]["top1_sd"]))
        for base, bb in [("RandomForest", "Rf"), ("HistGradientBoosting", "Hgb"), ("MLP", "Mlp")]:
            r = stats["paired"][f"{b}/{PRIMARY}-{base}"]
            put(f"{short}D{bb}", f"{r['diff_pp']:+.2f}"); put(f"{short}D{bb}Lo", f"{r['ci95'][0]:+.2f}")
            put(f"{short}D{bb}Hi", f"{r['ci95'][1]:+.2f}"); put(f"{short}D{bb}P", f(r['p_holm'], 3))
    for base, bb in [("RandomForest", "Rf"), ("HistGradientBoosting", "Hgb"), ("MLP", "Mlp")]:
        p = stats["pooled"][base]
        put(f"Pool{bb}", f"{p['estimate_pp']:+.2f}"); put(f"Pool{bb}Lo", f"{p['ci95'][0]:+.2f}")
        put(f"Pool{bb}Hi", f"{p['ci95'][1]:+.2f}"); put(f"Pool{bb}P", f(p['p'], 3)); put(f"Pool{bb}TauSq", f(p['tau2'], 2))
    put("NTotalTest", f"{sum(nb[b]['n']['test'] for b, _ in BENCH):,}".replace(",", "{,}"))
    put("GncaParams", f"{nb['curated']['models'][PRIMARY]['n_params']:,}".replace(",", "{,}"))
    js = jac["per_seed"]
    put("JacMax", f(max(s["spectral_norm_max"] for s in js), 3)); put("JacMean", f(np.mean([s["spectral_norm_mean"] for s in js]), 3))
    put("JacPNinetyFive", f(max(s["spectral_norm_p95"] for s in js), 3))
    put("JacFracBelowOne", f(100 * np.mean([s["fraction_below_1"] for s in js]), 1))
    put("JacN", f"{sum(s['n_jacobians'] for s in js):,}".replace(",", "{,}"))
    put("CpuName", cpu["cpu"].replace("_", " "))
    for k, kk in [("RandomForest", "Rf"), ("HistGradientBoosting", "Hgb"), ("GraphNCA_tree_LS", "Gnca")]:
        put(f"Lat{kk}", f(cpu[k]["latency"]["p50_ms"], 3)); put(f"LatPNF{kk}", f(cpu[k]["latency"]["p95_ms"], 3))
    put("SizeRfMB", f(cpu["RandomForest"]["pickle_bytes"] / 1e6, 2)); put("SizeHgbMB", f(cpu["HistGradientBoosting"]["pickle_bytes"] / 1e6, 2))
    put("SizeGnca", f(cpu["GraphNCA_tree_LS"]["state_dict_bytes"] / 1e3, 0))
    put("RfNodes", f"{cpu['RandomForest']['n_tree_nodes']:,}".replace(",", "{,}"))
    put("SpeedupRf", f(cpu["RandomForest"]["latency"]["p50_ms"] / cpu["GraphNCA_tree_LS"]["latency"]["p50_ms"], 1))
    put("SpeedupHgb", f(cpu["HistGradientBoosting"]["latency"]["p50_ms"] / cpu["GraphNCA_tree_LS"]["latency"]["p50_ms"], 1))
    e = stats["escalation"]
    for k, kk in [("auroc_composite_tau", "Comp"), ("auroc_entropy_only", "Ent"), ("auroc_step_norm_only", "Step")]:
        put(f"Auroc{kk}", f(np.mean(e[k]), 3)); put(f"Auroc{kk}Sd", f(np.std(e[k], ddof=1), 3))
    ca = rss["contractive_audit_3seed"]
    put("ContrRatioMax", f(max(c["perturbation_contraction_ratio_max"] for c in ca), 3))
    put("ContrTop", f(np.mean([c["top1"] for c in ca])))
    put("UnconRatio", f(np.mean([e["unconstrained_step_ratio_mean"] for e in esc]), 2))
    fs = rss["factorial_summary"]
    put("FactKFourLS", f(fs["Fact_tree_K4_LS"]["top1_mean"])); put("FactKFourLSSd", f(fs["Fact_tree_K4_LS"]["top1_sd"]))
    put("FactKFourCW", f(fs["Fact_tree_K4_CW"]["top1_mean"]))
    def fhm(m): return f(np.mean([r["test"]["top1"] for r in fh["models"][m]]))
    put("FhHgb", fhm("HistGradientBoosting")); put("FhRf", fhm("RandomForest")); put("FhStar", fhm("GraphNCA_star"))
    put("FhTree", fhm("GraphNCA_tree"))
    for fn, kk in [("teacher_qwen3.5_9b_curated_test.json", "Qwen"), ("teacher_gemma4_12b-it-q4_K_M_curated_test.json", "Gemma")]:
        t = json.loads((RES / "baselines_20261026" / fn).read_text())
        put(f"Llm{kk}", f(t["top1"])); put(f"Llm{kk}Lat", f(t["latency_ms_p50"], 0))
    e1 = json.loads((RES / "e1v2" / "run" / "e1v2_benchmark.json").read_text())["models"]
    def e1m(m): return float(np.mean([r["test"]["top1"] for r in e1[m]]))
    gn = [e1m(m) for m in e1 if m.startswith("GraphNCA_")]
    put("EoneGncaLo", f(min(gn))); put("EoneGncaHi", f(max(gn))); put("EoneHgb", f(e1m("HistGradientBoosting")))
    put("EoneRf", f(e1m("RandomForest"))); put("EoneMaj", f(e1m("Majority")))
    put("EoneSteps", f"{json.loads((RES / 'e1v2' / 'run' / 'e1v2_benchmark.json').read_text())['steps']:,}".replace(",", "{,}"))
    ax = (R11 / "lean_axioms.txt").read_text().splitlines()
    ax = [l for l in ax if "axioms" in l]
    put("LeanThms", str(len(ax))); put("LeanAxFree", str(sum("does not depend" in l for l in ax)))
    put("LeanSorry", str(sum("sorryAx" in l for l in ax)))
    lines = ["% AUTO-GENERATED by scripts/build_review_tables_1110.py -- do not edit by hand"]
    for k, v in M.items():
        lines.append(f"\\newcommand{{\\num{k}}}{{{v}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")

    # ---------------- tables ----------------
    rows = []
    for m in SHOW:
        cells = []
        for b, _ in BENCH:
            mm = nb[b]["models"][m]
            cells.append(f"{mm['top1_mean']:.2f} $\\pm$ {mm['top1_sd']:.2f}")
        params = nb["curated"]["models"][m]["n_params"]
        rows.append(f"{NICE[m]} & {params:,} & " + " & ".join(cells) + r" \\")
    t = [r"% AUTO-GENERATED by scripts/build_review_tables_1110.py", r"\begin{tabular}{@{}lr" + "c" * len(BENCH) + r"@{}}", r"\toprule",
         "Model & Params (Hermes) & " + " & ".join(n for _, n in BENCH) + r" \\",
         " & & " + " & ".join(f"$n_{{\\text{{test}}}}={nb[b]['n']['test']}$" for b, _ in BENCH) + r" \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_multibench.tex").write_text("\n".join(t).replace(",", "{,}").replace("{{,}}", "{,}") + "\n")
    rows = []
    for b, name in BENCH:
        for base in BASELINES:
            r = stats["paired"][f"{b}/{PRIMARY}-{base}"]
            rows.append(f"{name.split(' (')[0]} & {SHORT[base]} & {r['diff_pp']:+.2f} & [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] & {r['p']:.4f} & {r['p_holm']:.4f} \\\\")
    for base in BASELINES:
        p = stats["pooled"][base]
        rows.append(f"Pooled & {SHORT[base]} & {p['estimate_pp']:+.2f} & [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] & {p['p']:.4f} & -- \\\\")
    t = [r"% AUTO-GENERATED by scripts/build_review_tables_1110.py", r"\begin{tabular}{@{}llrcrr@{}}", r"\toprule",
         r"Benchmark & Baseline & $\Delta$ (pp) & 95\% CI & $p$ & $p_{\text{Holm}}$ \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_paired.tex").write_text("\n".join(t) + "\n")
    print(json.dumps({k: v for k, v in M.items()}, indent=0)[:4000])


if __name__ == "__main__":
    main()
