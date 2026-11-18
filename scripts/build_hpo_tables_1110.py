#!/usr/bin/env python3
"""Tables, macros and figure for the per-benchmark hyperparameter search (stage0/run_hpo_1110.py).

Inputs : stage0/stage0_results/hpo_1110/{hpo_summary.json,correct_by_seed_hpo.npz,run_meta.json}
         stage0/stage0_results/review_1110/{multibench.json,correct_by_seed.npz,correct_by_seed_trees.npz,
                                             input_matched_trees.json}  (untuned defaults)
Outputs: stage0/stage0_results/hpo_1110/stats_hpo.json,
         paper/generated/{numbers_hpo.tex,tab_hpo.tex,tab_hpo_paired.tex,tab_hpo_configs.tex},
         paper/figures/fig4_hpo.{pdf,png}
Same statistics as scripts/build_review_tables_1110.py: seed-clustered paired bootstrap over test turns
(10,000 resamples), Holm within each family, DerSimonian-Laird random effects over the four benchmarks.
Families: tuned primary (tree GNCA vs tuned RF/HGB/MLP, 12), tuned secondary (vs tuned RF*/HGB*, 8),
tuning gain (tuned minus untuned, every model x benchmark, 40).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review_tables_1110 import cluster_boot, holm, dersimonian_laird  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "stage0" / "stage0_results"
H = RES / "hpo_1110"; R11 = RES / "review_1110"
GEN = REPO / "paper" / "generated"; FIG = REPO / "paper" / "figures"
BENCH = [("curated", "Hermes", "Her"), ("bfcl_e1v2", "BFCL", "Bfcl"), ("toolbench", "ToolBench", "Tb"), ("apibank", "API-Bank", "Ab")]
# model id -> (untuned key, untuned npz, macro short, display name)
MODELS = {
    "logreg": ("Linear", "m", "Lr", "Logistic regression"),
    "rf": ("RandomForest", "m", "Rf", "Random Forest"),
    "hgb": ("HistGradientBoosting", "m", "Hgb", "HistGradientBoosting"),
    "rf_cell": ("RandomForest_packed", "t", "Rfp", "Random Forest, cell features"),
    "hgb_cell": ("HistGradientBoosting_packed", "t", "Hgbp", "HistGradientBoosting, cell features"),
    "mlp": ("MLP", "m", "Mlp", "MLP"),
    "gnca_chain": ("GraphNCA_1d_LS", "m", "Chain", "GNCA chain"),
    "gnca_star": ("GraphNCA_star_LS", "m", "Star", "GNCA star"),
    "gnca_tree": ("GraphNCA_tree_LS", "m", "Gnca", "GNCA tree (primary)"),
    "gnca_learned": ("GraphNCA_meta_LS", "m", "Meta", "GNCA learned"),
}
PRIMARY = "gnca_tree"
PRIM_BASE = ["rf", "hgb", "mlp"]; SEC_BASE = ["rf_cell", "hgb_cell"]
SHORT = {"rf": "RF", "hgb": "HGB", "mlp": "MLP", "rf_cell": "RF$^\\ast$", "hgb_cell": "HGB$^\\ast$"}


def f(x, nd=2):
    return f"{x:.{nd}f}"


def sgn(x):
    return f"{x:+.2f}"


def thou(n):
    return f"{int(n):,}".replace(",", "{,}")


def family(cz, pairs):
    out = {k: cluster_boot(a, b) for k, (a, b) in pairs.items()}
    for k, p in zip(out, holm([out[k]["p"] for k in out])):
        out[k]["p_holm"] = p
    return out


def main():
    hs = json.loads((H / "hpo_summary.json").read_text())
    ct = np.load(H / "correct_by_seed_hpo.npz")
    cu = {"m": np.load(R11 / "correct_by_seed.npz"), "t": np.load(R11 / "correct_by_seed_trees.npz")}
    mb = json.loads((R11 / "multibench.json").read_text())["benchmarks"]
    imt = json.loads((R11 / "input_matched_trees.json").read_text())["benchmarks"]
    meta = json.loads((H / "run_meta.json").read_text()) if (H / "run_meta.json").exists() else {}
    hb = hs["benchmarks"]
    for b, *_ in BENCH:
        for m in MODELS:
            assert f"{b}/{m}" in ct, f"missing tuned result {b}/{m}"
            assert ct[f"{b}/{m}"].shape == cu[MODELS[m][1]][f"{b}/{MODELS[m][0]}"].shape, (b, m)

    def untuned(b, m):
        k, src = MODELS[m][0], MODELS[m][1]
        if src == "m":
            mm = mb[b]["models"][k]; return mm["top1_mean"], mm["top1_sd"], mm["per_seed"][0]["val_top1"]
        mm = imt[b]["models"][k]; return mm["test_top1_mean"], mm["test_top1_sd"], mm["runs"][0]["val_top1"]

    S = {"source": "stage0/stage0_results/hpo_1110", "trials_per_model": hs["trials_per_model"], "seeds": hs["seeds"]}
    S["primary"] = family(ct, {f"{b}/{PRIMARY}-{m}": (ct[f"{b}/{PRIMARY}"], ct[f"{b}/{m}"]) for b, *_ in BENCH for m in PRIM_BASE})
    S["secondary"] = family(ct, {f"{b}/{PRIMARY}-{m}": (ct[f"{b}/{PRIMARY}"], ct[f"{b}/{m}"]) for b, *_ in BENCH for m in SEC_BASE})
    S["gain"] = family(ct, {f"{b}/{m}": (ct[f"{b}/{m}"], cu[MODELS[m][1]][f"{b}/{MODELS[m][0]}"]) for b, *_ in BENCH for m in MODELS})
    S["pooled_primary"] = {m: dersimonian_laird([S["primary"][f"{b}/{PRIMARY}-{m}"]["diff_pp"] for b, *_ in BENCH],
                                                [S["primary"][f"{b}/{PRIMARY}-{m}"]["se_pp"] for b, *_ in BENCH]) for m in PRIM_BASE}
    S["pooled_secondary"] = {m: dersimonian_laird([S["secondary"][f"{b}/{PRIMARY}-{m}"]["diff_pp"] for b, *_ in BENCH],
                                                  [S["secondary"][f"{b}/{PRIMARY}-{m}"]["se_pp"] for b, *_ in BENCH]) for m in SEC_BASE}
    S["pooled_gain"] = {m: dersimonian_laird([S["gain"][f"{b}/{m}"]["diff_pp"] for b, *_ in BENCH],
                                             [S["gain"][f"{b}/{m}"]["se_pp"] for b, *_ in BENCH]) for m in MODELS}
    # trial 0 reproduces the untuned default: compare its seed-0 validation score with the review_1110 run
    S["default_reproduction"] = {f"{b}/{m}": {"trial0_val": hb[b][m]["default_val_top1"], "review_val_seed0": untuned(b, m)[2]}
                                 for b, *_ in BENCH for m in MODELS}
    dev = [abs(v["trial0_val"] - v["review_val_seed0"]) for v in S["default_reproduction"].values()]
    S["default_reproduction_max_abs_pp"] = max(dev)
    # HGB on Hermes: the untuned run used scikit-learn's automatic early stopping (on for n > 10,000);
    # the search fixes early_stopping=False, so its trial 0 is not the same model there.
    es = {"curated/hgb", "curated/hgb_cell"}
    rest = {k: abs(v["trial0_val"] - v["review_val_seed0"]) for k, v in S["default_reproduction"].items() if k not in es}
    S["default_reproduction_max_abs_pp_excl_hermes_hgb"] = max(rest.values())
    S["default_reproduction_n_exact"] = sum(d < 0.005 for d in rest.values())
    S["default_reproduction_hermes_hgb"] = {k: S["default_reproduction"][k] for k in sorted(es)}
    S["n_default_selected"] = sum(hb[b][m]["best_trial"] == 0 for b, *_ in BENCH for m in MODELS)
    vs = {}
    for b, *_ in BENCH:
        g = [m for m in MODELS if m.startswith("gnca_")]
        best = max(g, key=lambda m: np.mean([x["val_top1"] for x in hb[b][m]["final"]]))
        vs[b] = {"model": best, "test": hb[b][best]["test_top1_mean"]}
    S["val_selected_gnca"] = vs
    cpu_s = sum(hb[b][m]["search_wall_s"] for b, *_ in BENCH for m in MODELS if not m.startswith(("mlp", "gnca")))
    gpu_s = sum(hb[b][m]["search_wall_s"] for b, *_ in BENCH for m in MODELS if m.startswith(("mlp", "gnca")))
    S["search_wall_s"] = {"cpu_models": cpu_s, "gpu_models": gpu_s}
    S["run_meta"] = meta
    (H / "stats_hpo.json").write_text(json.dumps(S, indent=2) + "\n")

    # ---------------- macros ----------------
    M = {}
    def put(k, v): M[k] = v
    put("HpoTrials", str(hs["trials_per_model"])); put("HpoNModels", str(len(MODELS)))
    put("HpoNTrialsTotal", thou(hs["trials_per_model"] * len(MODELS) * len(BENCH)))
    put("HpoNTrialsSpent", thou(sum(hb[b][m]["n_trials"] for b, *_ in BENCH for m in MODELS)))
    put("HpoCpuH", f(cpu_s / 3600, 1)); put("HpoGpuH", f(gpu_s / 3600, 1))
    if meta.get("wall_h") is not None:
        put("HpoWallH", f(meta["wall_h"], 1))
    put("HpoRepro", f(S["default_reproduction_max_abs_pp_excl_hermes_hgb"], 2)); put("HpoReproExact", str(S["default_reproduction_n_exact"]))
    put("HpoReproHgbHer", f(S["default_reproduction_max_abs_pp"], 2)); put("HpoNDefault", str(S["n_default_selected"]))
    for b, _, sh in BENCH:
        for m, (_, _, mm, _) in MODELS.items():
            r = hb[b][m]
            put(f"T{sh}{mm}", f(r["test_top1_mean"])); put(f"T{sh}{mm}Sd", f(r["test_top1_sd"]))
            g = S["gain"][f"{b}/{m}"]
            put(f"G{sh}{mm}", sgn(g["diff_pp"])); put(f"G{sh}{mm}P", f(g["p_holm"], 3))
        for m in PRIM_BASE + SEC_BASE:
            r = (S["primary"] if m in PRIM_BASE else S["secondary"])[f"{b}/{PRIMARY}-{m}"]; mm = MODELS[m][2]
            put(f"T{sh}D{mm}", sgn(r["diff_pp"])); put(f"T{sh}D{mm}Lo", sgn(r["ci95"][0]))
            put(f"T{sh}D{mm}Hi", sgn(r["ci95"][1])); put(f"T{sh}D{mm}P", f(r["p_holm"], 3))
        nice = {"gnca_chain": "chain", "gnca_star": "star", "gnca_tree": "tree", "gnca_learned": "learned"}
        put(f"TVs{sh}Model", nice[vs[b]["model"]]); put(f"TVs{sh}Test", f(vs[b]["test"]))
    for m in PRIM_BASE + SEC_BASE:
        p = (S["pooled_primary"] if m in PRIM_BASE else S["pooled_secondary"])[m]; mm = MODELS[m][2]
        put(f"TPool{mm}", sgn(p["estimate_pp"])); put(f"TPool{mm}Lo", sgn(p["ci95"][0]))
        put(f"TPool{mm}Hi", sgn(p["ci95"][1])); put(f"TPool{mm}P", f(p["p"], 3)); put(f"TPool{mm}TauSq", f(p["tau2"], 2))
    for m, (_, _, mm, _) in MODELS.items():
        p = S["pooled_gain"][m]
        put(f"GPool{mm}", sgn(p["estimate_pp"])); put(f"GPool{mm}Lo", sgn(p["ci95"][0]))
        put(f"GPool{mm}Hi", sgn(p["ci95"][1])); put(f"GPool{mm}P", f(p["p"], 3))
    put("TNSigPrimary", str(sum(v["p_holm"] < 0.05 for v in S["primary"].values())))
    put("TNSigSecondary", str(sum(v["p_holm"] < 0.05 for v in S["secondary"].values())))
    put("GNSig", str(sum(v["p_holm"] < 0.05 for v in S["gain"].values())))
    put("GNSigPos", str(sum(v["p_holm"] < 0.05 and v["diff_pp"] > 0 for v in S["gain"].values())))
    put("GNSigNeg", str(sum(v["p_holm"] < 0.05 and v["diff_pp"] < 0 for v in S["gain"].values())))
    tp = [hb[b][PRIMARY]["final"][0]["n_params"] for b, *_ in BENCH]
    put("TGncaParamsLo", thou(min(tp))); put("TGncaParamsHi", thou(max(tp)))
    # ---- power, CI widths (default-run primary family, S1), environment, earlier OOD pilot (S7) ----
    s1 = json.loads((R11 / "stats.json").read_text())["paired"]
    se = [v["se_pp"] for v in s1.values()]
    put("MdeLo", f(2.8 * min(se), 1)); put("MdeHi", f(2.8 * max(se), 1))
    small = [v["ci95"][1] - v["ci95"][0] for k, v in s1.items() if not k.startswith("curated/")]
    put("CiWidthLo", f(min(small), 1)); put("CiWidthHi", f(max(small), 1))
    nsm = [json.loads((R11 / "multibench.json").read_text())["benchmarks"][b]["n"]["test"] for b in ("bfcl_e1v2", "toolbench", "apibank")]
    put("NSmallTestLo", thou(min(nsm))); put("NSmallTestHi", thou(max(nsm)))
    env = json.loads((H / "environment.json").read_text())
    put("EnvPython", env["python"]); put("EnvTorch", env["torch"].split("+")[0]); put("EnvCuda", env["cuda"])
    put("EnvSklearn", env["scikit_learn"]); put("EnvNumpy", env["numpy"])
    ood = json.loads((RES / "hermes_ood_20261026" / "summary.json").read_text())
    om = ood["models"]
    best = max(om, key=lambda k: om[k]["diff_vs_floor_pp"])
    put("OodN", thou(ood["n_rows"])); put("OodSessions", str(ood["n_sessions"])); put("OodFloor", f(ood["floor_terminal"]["top1"]))
    put("OodNModels", str(len(om)))
    put("OodBestDiff", sgn(om[best]["diff_vs_floor_pp"])); put("OodBestLo", sgn(om[best]["diff_vs_floor_ci_session"][0]))
    put("OodBestHi", sgn(om[best]["diff_vs_floor_ci_session"][1]))
    put("OodNAbove", str(sum(v["diff_vs_floor_ci_session"][0] > 0 for v in om.values())))
    put("OodNBelow", str(sum(v["diff_vs_floor_ci_session"][1] < 0 for v in om.values())))
    S["ood_pilot"] = {"best": best, "n_above": M["OodNAbove"], "n_below": M["OodNBelow"]}
    (H / "stats_hpo.json").write_text(json.dumps(S, indent=2) + "\n")
    lines = ["% AUTO-GENERATED by scripts/build_hpo_tables_1110.py -- do not edit by hand"]
    lines += [f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in M.items()]
    (GEN / "numbers_hpo.tex").write_text("\n".join(lines) + "\n")

    # ---------------- tables ----------------
    def star(p):
        return "$^{\\dagger}$" if p < 0.05 else ""
    rows = []
    for m, (_, _, _, nm) in MODELS.items():
        cells = []
        for b, *_ in BENCH:
            r = hb[b][m]; g = S["gain"][f"{b}/{m}"]
            cells.append(f"{r['test_top1_mean']:.2f} $\\pm$ {r['test_top1_sd']:.2f} & {g['diff_pp']:+.2f}{star(g['p_holm'])}")
        p = S["pooled_gain"][m]
        rows.append(f"{nm} & " + " & ".join(cells) + f" & {p['estimate_pp']:+.2f} \\\\")
    t = ["% AUTO-GENERATED by scripts/build_hpo_tables_1110.py", r"\begin{tabular}{@{}l" + "rr" * len(BENCH) + r"r@{}}", r"\toprule",
         "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n, _ in BENCH) + r" & Pooled \\",
         " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BENCH))),
         "Model & " + " & ".join(["Tuned & $\\Delta_{\\text{tune}}$"] * len(BENCH)) + r" & $\Delta_{\text{tune}}$ \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_hpo.tex").write_text("\n".join(t) + "\n")
    rows = []
    for fam, pool, bases, title in [(S["primary"], S["pooled_primary"], PRIM_BASE, "Primary family (Holm over 12)"),
                                    (S["secondary"], S["pooled_secondary"], SEC_BASE, "Secondary: tuned trees on cell features (Holm over 8)")]:
        if bases is SEC_BASE:
            rows.append(r"\midrule")
        rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for b, nm, _ in BENCH:
            for m in bases:
                r = fam[f"{b}/{PRIMARY}-{m}"]
                rows.append(f"{nm} & {SHORT[m]} & {r['diff_pp']:+.2f} & [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] & {r['p']:.4f} & {r['p_holm']:.4f} \\\\")
        for m in bases:
            p = pool[m]
            rows.append(f"Pooled & {SHORT[m]} & {p['estimate_pp']:+.2f} & [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] & {p['p']:.4f} & -- \\\\")
    t = ["% AUTO-GENERATED by scripts/build_hpo_tables_1110.py", r"\begin{tabular}{@{}llrcrr@{}}", r"\toprule",
         r"Benchmark & Baseline & $\Delta$ (pp) & 95\% CI & $p$ & $p_{\text{Holm}}$ \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_hpo_paired.tex").write_text("\n".join(t) + "\n")
    # selected configurations (appendix)
    def cfgstr(m, c):
        if m == "logreg":
            return f"$C$={c['C']:.3g}, cw={c['class_weight'] or 'none'}"
        if m.startswith("rf"):
            dep = c['max_depth'] if c['max_depth'] else "none"
            return (f"{c['n_estimators']} trees, depth {dep}, leaf {c['min_samples_leaf']}, "
                    f"feat {c['max_features']}, cw={'bal.' if c['class_weight'] else 'none'}")
        if m.startswith("hgb"):
            return (f"lr {c['learning_rate']:.3g}, {c['max_iter']} it., {c['max_leaf_nodes']} leaves, leaf {c['min_samples_leaf']}, "
                    f"$\\ell_2$ {c['l2_regularization']:g}, cw={'bal.' if c['class_weight'] else 'none'}")
        s = f"lr {c['lr']:.2g}, wd {c['weight_decay']:.2g}, LS {c['label_smoothing']:g}, bs {thou(c['batch_size'])}, {thou(c['train_steps'])} st."
        if m == "mlp":
            return s + f", {c['hidden']} hid., drop {c['dropout']:g}"
        return s + f", $K$={c['K']}, $H$={c['hidden_c']}, upd {c['update_hidden']}"
    rows = []
    for b, nm, _ in BENCH:
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{nm}}}}} \\\\")
        for m, (_, _, _, mn) in MODELS.items():
            r = hb[b][m]
            rows.append(f"{mn.replace(' (primary)', '')} & {r['best_trial']} & {cfgstr(m, r['best_config'])} \\\\")
    t = ["% AUTO-GENERATED by scripts/build_hpo_tables_1110.py", r"\begin{tabularx}{\textwidth}{@{}lr>{\raggedright\arraybackslash}X@{}}", r"\toprule",
         r"Model & Trial & Selected configuration (highest seed-0 validation top-1; trial 0 = default) \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabularx}"]
    (GEN / "tab_hpo_configs.tex").write_text("\n".join(t) + "\n")
    figure(S)
    print(json.dumps(M, indent=0))


def figure(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    cols = {"curated": "#0072B2", "bfcl_e1v2": "#E69F00", "toolbench": "#009E73", "apibank": "#CC79A7"}
    mk = {"curated": "o", "bfcl_e1v2": "s", "toolbench": "^", "apibank": "v"}
    ms = list(MODELS)
    for i, m in enumerate(ms):
        for j, (b, nm, _) in enumerate(BENCH):
            g = S["gain"][f"{b}/{m}"]
            a1.plot(g["diff_pp"], i + (j - 1.5) * 0.09, mk[b], ms=2.6, color=cols[b], mfc=cols[b] if g["p_holm"] < 0.05 else "white",
                    mew=0.6, label=nm if i == 0 else None)
        p = S["pooled_gain"][m]
        a1.errorbar(p["estimate_pp"], i, xerr=[[p["estimate_pp"] - p["ci95"][0]], [p["ci95"][1] - p["estimate_pp"]]], fmt="D", ms=3.2,
                    color="k", lw=0.9, capsize=1.5, zorder=3, label="Pooled (95% CI)" if i == 0 else None)
    a1.axvline(0, color="k", lw=0.6, ls=":"); a1.set_yticks(range(len(ms)), [MODELS[m][3].replace(" (primary)", "").replace("HistGradientBoosting", "HGB").replace("Random Forest", "RF").replace(", cell features", "$^\\ast$").replace("Logistic regression", "Logistic reg.") for m in ms])
    a1.invert_yaxis(); a1.set_xlabel("Tuned $-$ untuned top-1 (pp)"); a1.set_title("(a) Effect of tuning (filled: Holm $p<0.05$)", fontsize=7)
    a1.spines[["top", "right"]].set_visible(False); a1.legend(fontsize=5.5, loc="lower right", frameon=False, handletextpad=0.3)
    bases = PRIM_BASE + SEC_BASE
    for i, m in enumerate(bases):
        fam = S["primary"] if m in PRIM_BASE else S["secondary"]; pool = S["pooled_primary"] if m in PRIM_BASE else S["pooled_secondary"]
        for j, (b, nm, _) in enumerate(BENCH):
            r = fam[f"{b}/{PRIMARY}-{m}"]; y = i + (j - 1.5) * 0.13
            a2.errorbar(r["diff_pp"], y, xerr=[[r["diff_pp"] - r["ci95"][0]], [r["ci95"][1] - r["diff_pp"]]], fmt=mk[b], ms=2.6, color=cols[b],
                        mfc=cols[b] if r["p_holm"] < 0.05 else "white", mew=0.6, lw=0.5, capsize=0.8)
        p = pool[m]
        a2.errorbar(p["estimate_pp"], i + 0.38, xerr=[[p["estimate_pp"] - p["ci95"][0]], [p["ci95"][1] - p["estimate_pp"]]], fmt="D", ms=3.2,
                    color="k", lw=0.9, capsize=1.5, zorder=3)
    a2.axvline(0, color="k", lw=0.6, ls=":"); a2.set_yticks(range(len(bases)), ["GNCA $-$ " + SHORT[m] for m in bases]); a2.invert_yaxis()
    a2.set_xlabel("$\\Delta$ top-1 (pp), both tuned"); a2.set_title("(b) Tuned tree GNCA vs. tuned baselines", fontsize=7)
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "fig4_hpo.pdf"); fig.savefig(FIG / "fig4_hpo.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
