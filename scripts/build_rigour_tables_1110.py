#!/usr/bin/env python3
"""Tables, macros and figure for the rigour pass (stage0/run_rigour_1110.py, run_ablation_1110.py,
run_latency_1110.py).

Step 1 (--collect RAW): compact the engine output directory RAW (finals with per-turn arrays) into
   stage0/stage0_results/rigour_1110/{rigour_summary.json, per_turn_rigour.npz, latency.json,
   ablation_summary.json, select/, trials/ (new models)}.   Per-turn arrays kept: test/ood correctness,
   top-1 probability, normalised entropy, step-norm (GNCA/GNN). No text, session ids are hashed.
Step 2 (default): statistics + paper/generated/{numbers_rig.tex, tab_rig_main.tex, tab_rig_deploy.tex,
   tab_rig_ablation.tex} + paper/figures/fig5_rig.{pdf,png} + rigour_1110/stats_rigour.json.
Statistics as in build_review_tables_1110.py: paired bootstrap over test turns with seeds kept together
(10,000 resamples), Holm within each family, DerSimonian-Laird pooling over the four benchmarks.
OOD and session-level statistics resample whole sessions (cluster bootstrap).
"""
from __future__ import annotations
import json, math, shutil, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review_tables_1110 import cluster_boot, holm, dersimonian_laird  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "stage0" / "stage0_results"; RG = RES / "rigour_1110"; HPO = RES / "hpo_1110"
GEN = REPO / "paper" / "generated"; FIG = REPO / "paper" / "figures"
BENCH = [("curated", "Hermes", "Her"), ("bfcl_e1v2", "BFCL", "Bfcl"), ("toolbench", "ToolBench", "Tb"), ("apibank", "API-Bank", "Ab")]
MODELS = {  # id -> (macro short, display)
    "logreg": ("Lr", "Logistic regression"), "rf": ("Rf", "Random Forest"), "hgb": ("Hgb", "HistGradientBoosting"),
    "rf_cell": ("Rfp", "Random Forest$^\\ast$"), "hgb_cell": ("Hgbp", "HistGradientBoosting$^\\ast$"),
    "knn": ("Knn", "$k$-NN retrieval"), "mlp": ("Mlp", "MLP"), "transformer": ("Tf", "Transformer"),
    "gnn_untied": ("Gnn", "GNN (untied steps)"), "gnca_chain": ("Chain", "GNCA chain"), "gnca_star": ("Star", "GNCA star"),
    "gnca_tree": ("Gnca", "GNCA tree (primary)"), "gnca_learned": ("Meta", "GNCA learned"),
}
PRIMARY = "gnca_tree"
FAMILIES = {"primary": ["rf", "hgb", "mlp"], "secondary": ["rf_cell", "hgb_cell"], "new": ["transformer", "gnn_untied", "knn"]}
SEEDS = [0, 1, 2, 3, 4]
ABL = {"sym": "symmetric normalisation", "no_lap": "no $(A{-}I)x$ block", "no_gate": "no gate", "untied": "untied weights"}
SESSION_BENCH = ["curated"]  # ToolBench/API-Bank test splits have one turn per session, BFCL is single-turn
B = 10000


def f(x, nd=2):
    return f"{x:.{nd}f}"


def sgn(x, nd=2):
    return f"{x:+.{nd}f}"


def thou(n):
    return f"{int(n):,}".replace(",", "{,}")


# ----------------------------------------------------------------------------- collect
def collect(raw: Path):
    RG.mkdir(parents=True, exist_ok=True)
    summ = {"source": "engine ~/nca_e_logs/rigour_1110 (stage0/run_rigour_1110.py)", "seeds": SEEDS, "benchmarks": {}}
    arr = {}
    for b, *_ in BENCH:
        meta = json.loads((raw / "meta" / f"{b}.json").read_text())
        arr[f"{b}/y_test"] = np.array(meta["y_test"], np.int16)
        arr[f"{b}/test_session"] = np.unique(meta["test_session"], return_inverse=True)[1].astype(np.int32)
        if "y_ood" in meta:  # private sessions: store only whether each turn's label is the majority tool, not the labels
            maj = int(np.bincount(np.array(meta["y_test"])).argmax())
            arr[f"{b}/ood_is_majority"] = (np.array(meta["y_ood"]) == maj).astype(np.int8)
            arr[f"{b}/ood_session"] = np.unique(meta["ood_session"], return_inverse=True)[1].astype(np.int32)
        sb = {}
        for m in MODELS:
            recs = []
            for s in SEEDS:
                fd = raw / "final" / b / m
                rec = json.loads((fd / f"seed{s}.json").read_text()); z = np.load(fd / f"seed{s}.npz")
                recs.append(rec)
                for sp in ("test", "ood"):
                    if f"{sp}_correct" not in z:
                        continue
                    P = z[f"{sp}_proba"].astype(np.float64); P = P / P.sum(1, keepdims=True)
                    ent = -(np.where(P > 0, P * np.log(np.clip(P, 1e-12, 1)), 0)).sum(1) / math.log(P.shape[1])
                    arr.setdefault(f"{b}/{m}/{sp}_correct", []).append(z[f"{sp}_correct"])
                    arr.setdefault(f"{b}/{m}/{sp}_pmax", []).append(P.max(1).astype(np.float32))
                    arr.setdefault(f"{b}/{m}/{sp}_entropy", []).append(ent.astype(np.float32))
                    if f"{sp}_stepnorm" in z:
                        arr.setdefault(f"{b}/{m}/{sp}_stepnorm", []).append(z[f"{sp}_stepnorm"])
            sb[m] = {"selected_trial": recs[0]["selected_trial"], "config": recs[0]["config"], "selection": recs[0]["selection"],
                     "n_params": recs[0].get("n_params"), "final": [{k: v for k, v in r.items() if k not in ("selection", "config")} for r in recs]}
            hp = HPO / "hpo_summary.json"
            if hp.exists() and m in json.loads(hp.read_text())["benchmarks"][b]:
                sb[m]["seed0_selected_trial"] = json.loads(hp.read_text())["benchmarks"][b][m]["best_trial"]
            else:
                sb[m]["seed0_selected_trial"] = max(recs[0]["selection"], key=lambda x: (x["val_seed0"], -x["trial"]))["trial"]
        summ["benchmarks"][b] = sb
    arr = {k: (np.stack(v) if isinstance(v, list) else v) for k, v in arr.items()}
    np.savez_compressed(RG / "per_turn_rigour.npz", **arr)
    (RG / "rigour_summary.json").write_text(json.dumps(summ, indent=1, default=str) + "\n")
    ab = {}
    for b, *_ in BENCH:
        for v in ABL:
            fd = raw / "ablation" / b / v
            ab[f"{b}/{v}"] = {"runs": [json.loads((fd / f"seed{s}.json").read_text()) for s in SEEDS],
                              "correct": np.stack([np.load(fd / f"seed{s}.npz")["test_correct"] for s in SEEDS])}
    np.savez_compressed(RG / "ablation_correct.npz", **{k: v["correct"] for k, v in ab.items()})
    (RG / "ablation_summary.json").write_text(json.dumps({k: v["runs"] for k, v in ab.items()}, indent=1) + "\n")
    if (raw / "latency.json").exists() and json.loads((raw / "latency.json").read_text())["models"]:
        shutil.copy(raw / "latency.json", RG / "latency.json")
    for extra in ("artifacts/refit_check.jsonl",):
        if (raw / extra).exists():
            (RG / "artifacts").mkdir(exist_ok=True); shutil.copy(raw / extra, RG / extra)
    for sub in ("select",):
        shutil.copytree(raw / sub, RG / sub, dirs_exist_ok=True)
    for b, *_ in BENCH:
        for m in ("knn", "transformer", "gnn_untied"):
            (RG / "trials" / b).mkdir(parents=True, exist_ok=True)
            shutil.copy(raw / "trials" / b / f"{m}.jsonl", RG / "trials" / b / f"{m}.jsonl")
    print("collected", len(arr), "arrays")


# ----------------------------------------------------------------------------- statistics helpers
def session_boot(v: np.ndarray, sess: np.ndarray, rng, n=B):
    """v: per-row values (already averaged over seeds). Cluster bootstrap over sessions -> mean and 95% CI (in %)."""
    us = np.unique(sess); members = [np.where(sess == s)[0] for s in us]
    sums = np.array([v[m].sum() for m in members]); cnt = np.array([len(m) for m in members])
    pick = rng.integers(0, len(us), size=(n, len(us)))
    boots = sums[pick].sum(1) / cnt[pick].sum(1)
    return {"mean": 100 * float(v.mean()), "ci95": [100 * float(np.percentile(boots, 2.5)), 100 * float(np.percentile(boots, 97.5))],
            "se": 100 * float(boots.std(ddof=1)),
            "p_two_sided_vs0": float(min(1.0, 2 * (1 + min(np.sum(boots <= 0), np.sum(boots >= 0))) / (n + 1)))}


def session_success(corr: np.ndarray, sess: np.ndarray, keep: np.ndarray | None = None):
    """corr: (seeds, n). Fraction of sessions whose (kept) turns are all correct, per seed."""
    us = np.unique(sess); out = []
    for s in range(corr.shape[0]):
        ok = corr[s].astype(bool) | (~keep[s] if keep is not None else False)
        out.append(np.mean([ok[sess == u].all() for u in us]))
    return np.array(out)


def auroc(err, score):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = err.sum(); n0 = len(err) - n1
    return float((r[err == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def risk_coverage(corr, unc):
    o = np.argsort(unc, kind="stable"); c = corr[o].astype(float); n = len(c)
    sel_acc = np.cumsum(c) / np.arange(1, n + 1)
    return sel_acc, float(np.mean(1 - sel_acc))  # AURC (risk averaged over coverage)


def ece(corr, conf, bins=15):
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


# ----------------------------------------------------------------------------- main
def main():
    S = json.loads((RG / "rigour_summary.json").read_text()); Z = np.load(RG / "per_turn_rigour.npz")
    rng = np.random.default_rng(20261110)
    hb = S["benchmarks"]; M = {}
    put = M.__setitem__
    C = lambda b, m, sp="test": Z[f"{b}/{m}/{sp}_correct"]
    st = {"source": "stage0/stage0_results/rigour_1110", "seeds": SEEDS}

    # (a) accuracy with 5 seeds and multi-seed selection
    acc = {b: {m: {"mean": float(100 * C(b, m).mean()), "sd": float(np.std(100 * C(b, m).mean(1), ddof=1))} for m in MODELS} for b, *_ in BENCH}
    st["accuracy"] = acc
    fam = {}
    for name, bases in FAMILIES.items():
        pairs = {f"{b}/{PRIMARY}-{m}": cluster_boot(C(b, PRIMARY), C(b, m)) for b, *_ in BENCH for m in bases}
        for k, p in zip(pairs, holm([v["p"] for v in pairs.values()])):
            pairs[k]["p_holm"] = p
        fam[name] = pairs
        st[f"pooled_{name}"] = {m: dersimonian_laird([pairs[f"{b}/{PRIMARY}-{m}"]["diff_pp"] for b, *_ in BENCH],
                                                     [pairs[f"{b}/{PRIMARY}-{m}"]["se_pp"] for b, *_ in BENCH]) for m in bases}
    st["paired"] = fam
    nchg = sum(hb[b][m]["selected_trial"] != hb[b][m]["seed0_selected_trial"] for b, *_ in BENCH for m in MODELS)
    put("RigNSelChanged", str(nchg)); put("RigNSelTotal", str(len(BENCH) * len(MODELS)))
    put("RigTopK", "5"); put("RigNSeeds", str(len(SEEDS))); put("RigNModels", str(len(MODELS)))
    for b, _, sh in BENCH:
        for m, (mm, _) in MODELS.items():
            put(f"R{sh}{mm}", f(acc[b][m]["mean"])); put(f"R{sh}{mm}Sd", f(acc[b][m]["sd"]))
        for name, bases in FAMILIES.items():
            for m in bases:
                r = fam[name][f"{b}/{PRIMARY}-{m}"]; mm = MODELS[m][0]
                put(f"R{sh}D{mm}", sgn(r["diff_pp"])); put(f"R{sh}D{mm}Lo", sgn(r["ci95"][0])); put(f"R{sh}D{mm}Hi", sgn(r["ci95"][1]))
                put(f"R{sh}D{mm}P", f(r["p_holm"], 3))
    for name, bases in FAMILIES.items():
        for m in bases:
            p = st[f"pooled_{name}"][m]; mm = MODELS[m][0]
            put(f"RPool{mm}", sgn(p["estimate_pp"])); put(f"RPool{mm}Lo", sgn(p["ci95"][0])); put(f"RPool{mm}Hi", sgn(p["ci95"][1]))
            put(f"RPool{mm}P", f(p["p"], 3)); put(f"RPool{mm}TauSq", f(p["tau2"], 2))
        put(f"RNSig{name.capitalize()}", str(sum(v["p_holm"] < 0.05 for v in fam[name].values())))
        put(f"RNSigPos{name.capitalize()}", str(sum(v["p_holm"] < 0.05 and v["diff_pp"] > 0 for v in fam[name].values())))
        put(f"RNSigNeg{name.capitalize()}", str(sum(v["p_holm"] < 0.05 and v["diff_pp"] < 0 for v in fam[name].values())))
        put(f"RNFam{name.capitalize()}", str(len(fam[name])))
    se = [v["se_pp"] for fm in fam.values() for v in fm.values()]
    put("RMdeLo", f(2.8 * min(se), 1)); put("RMdeHi", f(2.8 * max(se), 1))
    # best model per benchmark (by test mean, reported only descriptively)
    for b, _, sh in BENCH:
        best = max(MODELS, key=lambda m: acc[b][m]["mean"]); put(f"R{sh}Best", MODELS[best][1].replace("$^\\ast$", "$^{\\ast}$")); put(f"R{sh}BestAcc", f(acc[b][best]["mean"]))
    # parameter counts of new neural baselines (Hermes)
    for m in ("transformer", "gnn_untied", "mlp", "gnca_tree"):
        ps = [hb[b][m]["n_params"] for b, *_ in BENCH]
        put(f"RParams{MODELS[m][0]}Lo", thou(min(ps))); put(f"RParams{MODELS[m][0]}Hi", thou(max(ps)))

    # FLOPs per turn from Proposition (iv), default and tuned Hermes configuration of the tree GNCA
    def flops(K, H, Dm, Dobs=146, V=5, Cn=13):
        Dc = Dobs + H
        return K * V * (6 * Dc * Dm + 4 * H * Dm) + 2 * K * V * V * Dc + 2 * V * H * Cn
    put("FlopsGnca", f(flops(4, 16, 96) / 1e6, 2) + "\\,M")
    tc = hb["curated"][PRIMARY]["config"]
    put("FlopsGncaTunedHer", f(flops(tc["K"], tc["hidden_c"], tc["update_hidden"]) / 1e6, 2) + "\\,M")

    # (b) OOD on held-out private Hermes sessions
    so = Z["curated/ood_session"]
    floor = Z["curated/ood_is_majority"].astype(float)  # constant predictor: majority tool of the in-distribution test split
    yo = floor
    ood = {"n_rows": int(len(yo)), "n_sessions": int(len(np.unique(so))), "floor": session_boot(floor, so, rng), "models": {}}
    for m in MODELS:
        c = C("curated", m, "ood"); v = c.mean(0)
        ood["models"][m] = {**session_boot(v, so, rng), "sd_seeds": float(np.std(100 * c.mean(1), ddof=1)),
                            "id_test": acc["curated"][m]["mean"], "vs_floor": session_boot(v - floor, so, rng)}
    oo = {m: session_boot(C("curated", PRIMARY, "ood").mean(0) - C("curated", m, "ood").mean(0), so, rng) for m in MODELS if m != PRIMARY}
    for k, p in zip(oo, holm([v["p_two_sided_vs0"] for v in oo.values()])):
        oo[k]["p_holm"] = p
    ood["paired_vs_primary"] = oo; st["ood"] = ood
    put("ROodN", thou(ood["n_rows"])); put("ROodSess", str(ood["n_sessions"])); put("ROodFloor", f(ood["floor"]["mean"]))
    for m, (mm, _) in MODELS.items():
        o = ood["models"][m]
        put(f"ROod{mm}", f(o["mean"])); put(f"ROod{mm}Lo", f(o["ci95"][0])); put(f"ROod{mm}Hi", f(o["ci95"][1]))
        put(f"ROodDrop{mm}", f(o["id_test"] - o["mean"]))
        if m != PRIMARY:
            put(f"ROodD{mm}", sgn(oo[m]["mean"])); put(f"ROodD{mm}Lo", sgn(oo[m]["ci95"][0])); put(f"ROodD{mm}Hi", sgn(oo[m]["ci95"][1])); put(f"ROodD{mm}P", f(oo[m]["p_holm"], 3))
    put("ROodNSig", str(sum(v["p_holm"] < 0.05 for v in oo.values())))
    put("ROodNAboveFloor", str(sum(o["vs_floor"]["ci95"][0] > 0 for o in ood["models"].values())))
    drops = [o["id_test"] - o["mean"] for o in ood["models"].values()]
    put("ROodDropLo", f(min(drops))); put("ROodDropHi", f(max(drops)))
    put("ROodMaxAbsDNs", f(max(abs(v["mean"]) for v in oo.values() if v["p_holm"] >= 0.05)))
    put("ROodLo", f(min(o["mean"] for o in ood["models"].values())))
    bo = max(MODELS, key=lambda m: ood["models"][m]["mean"]); put("ROodBest", MODELS[bo][1]); put("ROodBestAcc", f(ood["models"][bo]["mean"]))

    # (d) escalation / selective prediction on the primary tuned model, all benchmarks, 5 seeds
    esc = {}
    covs = (0.9, 0.8, 0.7)
    for b, _, sh in BENCH:
        e = {}
        for m in MODELS:
            cc = C(b, m); ent = Z[f"{b}/{m}/test_entropy"]; pm = Z[f"{b}/{m}/test_pmax"]
            rows = []
            for s in range(cc.shape[0]):
                err = 1 - cc[s]; r = {"auroc_entropy": auroc(err, ent[s]), "auroc_msp": auroc(err, -pm[s]), "ece": 100 * ece(cc[s], pm[s])}
                sel, aurc = risk_coverage(cc[s], ent[s]); r["aurc"] = 100 * aurc
                for cv in covs:
                    r[f"selacc_{int(cv * 100)}"] = 100 * float(sel[int(round(cv * len(sel))) - 1])
                if f"{b}/{m}/test_stepnorm" in Z:
                    sn = Z[f"{b}/{m}/test_stepnorm"][s]
                    r["auroc_step"] = auroc(err, sn); r["auroc_composite"] = auroc(err, ent[s] + sn)
                rows.append(r)
            e[m] = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1))} for k in rows[0]}
        esc[b] = e
        g = e[PRIMARY]
        put(f"REsc{sh}AucEnt", f(g["auroc_entropy"]["mean"], 3)); put(f"REsc{sh}AucEntSd", f(g["auroc_entropy"]["sd"], 3))
        put(f"REsc{sh}AucComp", f(g["auroc_composite"]["mean"], 3)); put(f"REsc{sh}AucStep", f(g["auroc_step"]["mean"], 3))
        put(f"REsc{sh}Ece", f(g["ece"]["mean"], 1)); put(f"REsc{sh}Aurc", f(g["aurc"]["mean"], 1))
        for cv in covs:
            put(f"REsc{sh}Sel{ {90: 'Ninety', 80: 'Eighty', 70: 'Seventy'}[int(round(cv * 100))]}", f(g[f"selacc_{int(cv * 100)}"]["mean"]))
        put(f"REsc{sh}Full", f(acc[b][PRIMARY]["mean"]))
        for m in ("hgb", "mlp"):
            put(f"REsc{sh}AucEnt{MODELS[m][0]}", f(e[m]["auroc_entropy"]["mean"], 3)); put(f"REsc{sh}Ece{MODELS[m][0]}", f(e[m]["ece"]["mean"], 1))
            put(f"REsc{sh}SelEighty{MODELS[m][0]}", f(e[m]["selacc_80"]["mean"]))
    st["escalation"] = esc
    put("REscNStepBelowHalf", str(sum(esc[b][PRIMARY]["auroc_step"]["mean"] < 0.5 for b, *_ in BENCH)))
    put("REscNCompWorse", str(sum(esc[b][PRIMARY]["auroc_composite"]["mean"] < esc[b][PRIMARY]["auroc_entropy"]["mean"] for b, *_ in BENCH)))

    # (f) session-level routing success (offline proxy for task success)
    ses = {}
    for b in SESSION_BENCH + ["curated_ood"]:
        bb, sp = ("curated", "ood") if b == "curated_ood" else (b, "test")
        sess = Z[f"{bb}/{sp}_session"]; r = {"n_sessions": int(len(np.unique(sess)))}
        for m in MODELS:
            c = C(bb, m, sp); ent = Z[f"{bb}/{m}/{sp}_entropy"]
            thr = np.quantile(ent, 0.8, axis=1, keepdims=True); keep = ent <= thr  # keep the 80% most confident turns per seed
            full = session_success(c, sess); kept = session_success(c, sess, keep)
            r[m] = {"all_correct": 100 * float(full.mean()), "all_correct_sd": 100 * float(full.std(ddof=1)),
                    "all_kept_correct_cov80": 100 * float(kept.mean()), "turn_acc": 100 * float(c.mean())}
        ses[b] = r
    st["session"] = ses
    shs = {"curated": "Her", "curated_ood": "Ood"}
    put("RSesHerTurns", f(len(Z["curated/test_session"]) / ses["curated"]["n_sessions"], 1)); put("RSesOodTurns", f(len(Z["curated/ood_session"]) / ses["curated_ood"]["n_sessions"], 1))
    put("RSesTbOne", str(int(len(np.unique(Z["toolbench/test_session"])) == len(Z["toolbench/test_session"]) and len(np.unique(Z["apibank/test_session"])) == len(Z["apibank/test_session"]))))
    for b, sh in shs.items():
        put(f"RSes{sh}N", str(ses[b]["n_sessions"]))
        for m, (mm, _) in MODELS.items():
            put(f"RSes{sh}{mm}", f(ses[b][m]["all_correct"], 1)); put(f"RSes{sh}{mm}Esc", f(ses[b][m]["all_kept_correct_cov80"], 1))
        put(f"RSes{sh}Lo", f(min(ses[b][m]["all_correct"] for m in MODELS), 1)); put(f"RSes{sh}Hi", f(max(ses[b][m]["all_correct"] for m in MODELS), 1))
        put(f"RSes{sh}EscLo", f(min(ses[b][m]["all_kept_correct_cov80"] for m in MODELS), 1)); put(f"RSes{sh}EscHi", f(max(ses[b][m]["all_kept_correct_cov80"] for m in MODELS), 1))

    # (c) latency
    if not (RG / "latency.json").exists():  # only while the latency job is still running; the paper build asserts it exists
        print("WARNING: latency.json missing -- latency columns are NaN"); L = {"n_turns": 0, "models": {m: {"seed0": {k: {"median": float("nan"), "p95": float("nan")} for k in ("featurise_ms", "predict_ms", "total_ms")} | {"size_bytes": 0}} for m in MODELS}}
    else:
        L = json.loads((RG / "latency.json").read_text())
    lat = {}
    for m, (mm, _) in MODELS.items():
        sd = L["models"][m]
        g = lambda key, q: float(np.mean([sd[s][key][q] for s in sd]))
        lat[m] = {"feat_med": g("featurise_ms", "median"), "pred_med": g("predict_ms", "median"), "tot_med": g("total_ms", "median"),
                  "tot_p95": g("total_ms", "p95"), "size_kb": float(np.mean([sd[s]["size_bytes"] for s in sd])) / 1024, "n_seeds": len(sd)}
        put(f"RLat{mm}", f(lat[m]["tot_med"])); put(f"RLat{mm}Pred", f(lat[m]["pred_med"], 3)); put(f"RLat{mm}Feat", f(lat[m]["feat_med"]))
        put(f"RLat{mm}PNF", f(lat[m]["tot_p95"])); put(f"RSize{mm}", f(lat[m]["size_kb"], 0) if lat[m]["size_kb"] >= 10 else f(lat[m]["size_kb"], 1))
    st["latency"] = lat; put("RLatN", thou(L["n_turns"]))
    put("RLatFeatLo", f(min(v["feat_med"] for v in lat.values()))); put("RLatFeatHi", f(max(v["feat_med"] for v in lat.values())))
    put("RLatPredLo", f(min(v["pred_med"] for v in lat.values()), 3)); put("RLatPredHi", f(max(v["pred_med"] for v in lat.values()), 2))
    put("RLatTotLo", f(min(v["tot_med"] for v in lat.values()))); put("RLatTotHi", f(max(v["tot_med"] for v in lat.values())))
    put("RLatFeatShareGnca", f(100 * lat[PRIMARY]["feat_med"] / lat[PRIMARY]["tot_med"], 0))

    # refit check (resume-only artifact refits vs. the reported finals; Hermes, seeds 0-2)
    rc = [json.loads(l) for l in (RG / "artifacts/refit_check.jsonl").read_text().splitlines() if l.strip()]
    dd = [abs(r["refit_test_top1"] - r["final_test_top1"]) for r in rc]
    st["refit_check"] = rc
    put("RRefitN", str(len(rc))); put("RRefitNExact", str(sum(d < 0.005 for d in dd))); put("RRefitMaxDiff", f(max(dd)))
    put("RRefitMaxModel", MODELS[max(rc, key=lambda r: abs(r["refit_test_top1"] - r["final_test_top1"]))["model"]][1])

    # ablation
    AZ = np.load(RG / "ablation_correct.npz"); abl = {}
    pairs = {f"{b}/{v}": cluster_boot(AZ[f"{b}/{v}"], C(b, PRIMARY)) for b, *_ in BENCH for v in ABL}
    for k, p in zip(pairs, holm([x["p"] for x in pairs.values()])):
        pairs[k]["p_holm"] = p
    for v in ABL:
        abl[v] = dersimonian_laird([pairs[f"{b}/{v}"]["diff_pp"] for b, *_ in BENCH], [pairs[f"{b}/{v}"]["se_pp"] for b, *_ in BENCH])
        mm = {"sym": "Sym", "no_lap": "NoLap", "no_gate": "NoGate", "untied": "Untied"}[v]
        put(f"RAbl{mm}", sgn(abl[v]["estimate_pp"])); put(f"RAbl{mm}Lo", sgn(abl[v]["ci95"][0])); put(f"RAbl{mm}Hi", sgn(abl[v]["ci95"][1]))
        for b, _, sh in BENCH:
            put(f"RAbl{mm}{sh}", sgn(pairs[f"{b}/{v}"]["diff_pp"]))
    st["ablation"] = {"paired": pairs, "pooled": abl}
    put("RAblNSig", str(sum(x["p_holm"] < 0.05 for x in pairs.values()))); put("RAblNFam", str(len(pairs)))

    (RG / "stats_rigour.json").write_text(json.dumps(st, indent=1, default=float) + "\n")
    lines = ["% AUTO-GENERATED by scripts/build_rigour_tables_1110.py -- do not edit by hand"] + [f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in M.items()]
    (GEN / "numbers_rig.tex").write_text("\n".join(lines) + "\n")

    # ---------------- tables ----------------
    def mark(p):
        return "$^{\\dagger}$" if p < 0.05 else ""
    pairmap = {m: n for n, bs in FAMILIES.items() for m in bs}
    rows = []
    order = ["logreg", "rf", "hgb", "rf_cell", "hgb_cell", "knn", "mlp", "transformer", "gnn_untied", "gnca_chain", "gnca_star", "gnca_learned", "gnca_tree"]
    for m in order:
        cells = []
        for b, *_ in BENCH:
            a = acc[b][m]; s = f"{a['mean']:.2f}\\,$\\pm$\\,{a['sd']:.2f}"
            if m in pairmap:
                r = fam[pairmap[m]][f"{b}/{PRIMARY}-{m}"]; s += f" & {r['diff_pp']:+.2f}{mark(r['p_holm'])}"
            else:
                s += " & --"
            cells.append(s)
        if m in pairmap:
            p = st[f"pooled_{pairmap[m]}"][m]; pool = f"{p['estimate_pp']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"
        else:
            pool = "--"
        rows.append(f"{MODELS[m][1]} & " + " & ".join(cells) + f" & {pool} \\\\")
        if m in ("hgb_cell", "gnn_untied"):
            rows.append(r"\midrule")
    t = ["% AUTO-GENERATED by scripts/build_rigour_tables_1110.py", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}l" + "rr" * len(BENCH) + r"r@{}}", r"\toprule",
         "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n, _ in BENCH) + r" & Pooled $\Delta$ \\",
         " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(BENCH))),
         "Model & " + " & ".join(["Top-1 & $\\Delta$"] * len(BENCH)) + r" & [95\% CI] \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}}"]
    (GEN / "tab_rig_main.tex").write_text("\n".join(t) + "\n")

    rows = []
    for m in order:
        o = ood["models"][m]; l = lat[m]; sh = ses["curated"][m]
        d = "--" if m == PRIMARY else f"{oo[m]['mean']:+.2f}{mark(oo[m]['p_holm'])}"
        rows.append(f"{MODELS[m][1]} & {o['mean']:.2f} [{o['ci95'][0]:.1f}, {o['ci95'][1]:.1f}] & {d} & {sh['all_correct']:.1f} & {ses['curated_ood'][m]['all_correct']:.1f} & "
                    f"{l['feat_med']:.2f} & {l['pred_med']:.3f} & {l['tot_med']:.2f} & {l['size_kb']:.0f} \\\\")
        if m in ("hgb_cell", "gnn_untied"):
            rows.append(r"\midrule")
    t = ["% AUTO-GENERATED by scripts/build_rigour_tables_1110.py", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{OOD (held-out sessions)} & \multicolumn{2}{c}{Sessions all correct (\%)} & \multicolumn{3}{c}{CPU latency, 1 thread (ms, median)} & Size \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}",
         r"Model & Top-1 [95\% CI] & $\Delta$ vs.\ GNCA & Hermes test & OOD & Featurise & Predict & Total & kB \\", r"\midrule"] + rows + \
        [r"\midrule", f"Majority class & {ood['floor']['mean']:.2f} [{ood['floor']['ci95'][0]:.1f}, {ood['floor']['ci95'][1]:.1f}] & & & & & & & \\\\", r"\bottomrule", r"\end{tabular}}"]
    (GEN / "tab_rig_deploy.tex").write_text("\n".join(t) + "\n")

    rows = []
    for v, nm in ABL.items():
        cells = " & ".join(f"{pairs[f'{b}/{v}']['diff_pp']:+.2f}{mark(pairs[f'{b}/{v}']['p_holm'])}" for b, *_ in BENCH)
        p = abl[v]; rows.append(f"{nm} & {cells} & {p['estimate_pp']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] \\\\")
    rows.append(r"\midrule")
    for key, nm in (("auroc_entropy", "AUROC, entropy"), ("auroc_composite", "AUROC, $s_{\\text{esc}}$ ($\\beta{=}1$)"), ("auroc_step", "AUROC, step norm only"),
                    ("ece", "ECE (\\%)"), ("selacc_80", "Top-1 on 80\\% most confident")):
        nd = 3 if key.startswith("auroc") else 2
        rows.append(f"{nm} & " + " & ".join(f"{esc[b][PRIMARY][key]['mean']:.{nd}f}\\,$\\pm$\\,{esc[b][PRIMARY][key]['sd']:.{nd}f}" for b, *_ in BENCH) + " & \\\\")
    t = ["% AUTO-GENERATED by scripts/build_rigour_tables_1110.py", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         "& " + " & ".join(n for _, n, _ in BENCH) + r" & Pooled [95\% CI] \\", r"\midrule",
         r"\multicolumn{6}{@{}l}{\emph{Ablation: change in top-1 (pp) vs.\ the tuned tree GNCA, same configuration}} \\"] + rows[:len(ABL)] + \
        [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Escalation of the tuned tree GNCA (5 seeds, mean $\pm$ SD)}} \\"] + rows[len(ABL) + 1:] + [r"\bottomrule", r"\end{tabular}"]
    (GEN / "tab_rig_ablation.tex").write_text("\n".join(t) + "\n")

    # ---------------- figure: accuracy vs latency (Hermes) + OOD ----------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.0, 1.1]})
    cols = {"gnca": "#009E73", "tree": "#D55E00", "nn": "#0072B2", "other": "#555555"}  # Okabe-Ito
    marks = {"gnca": "o", "tree": "s", "nn": "^", "other": "D"}
    def fam(m):
        return "gnca" if m.startswith("gnca") else "tree" if m.startswith(("rf", "hgb")) else "nn" if m in ("mlp", "transformer", "gnn_untied") else "other"
    def col(m):
        return cols[fam(m)]
    short = lambda m: MODELS[m][1].replace("$^\\ast$", "*").replace("$k$", "k").replace(" (primary)", "").replace("HistGradientBoosting", "HGB").replace("Random Forest", "RF").replace("Logistic regression", "LogReg").replace(" (untied steps)", " untied")
    label_these = {PRIMARY: (4, -9), "hgb_cell": (4, 2), "hgb": (4, -3), "rf": (-14, -9), "knn": (-34, 5), "logreg": (4, 2), "mlp": (-22, 4)}
    for m in order:
        x, y = lat[m]["tot_med"], acc["curated"][m]["mean"]
        ax[0].scatter(x, y, s=16, color=col(m), marker=marks[fam(m)], zorder=3, edgecolor="k" if m == PRIMARY else "none", linewidth=0.6)
        if m in label_these:
            ax[0].annotate(short(m), (x, y), fontsize=5.6, xytext=label_these[m], textcoords="offset points")
    ax[0].set_xscale("log"); ax[0].xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50])); ax[0].xaxis.set_major_formatter(FixedFormatter(["3", "5", "10", "20", "50"])); ax[0].xaxis.set_minor_locator(NullLocator())
    ax[0].set_xlabel("End-to-end CPU latency per turn (ms, median, 1 thread)"); ax[0].set_ylabel("Hermes test top-1 (%)")
    ax[0].set_title("(a) Accuracy vs. end-to-end latency")
    ax[0].legend(handles=[Line2D([], [], ls="", marker=marks[k], color=cols[k], label=l, ms=4) for k, l in (("gnca", "GNCA"), ("tree", "tree ensembles"), ("nn", "MLP/Transformer/GNN"), ("other", "LogReg, k-NN"))], fontsize=5.4, loc="lower right", frameon=False)
    ys = np.arange(len(order))
    for i, m in enumerate(order):
        o = ood["models"][m]
        ax[1].errorbar(o["mean"], i, xerr=[[o["mean"] - o["ci95"][0]], [o["ci95"][1] - o["mean"]]], fmt=marks[fam(m)], ms=3, color=col(m), capsize=1.5, lw=0.8)
        ax[1].scatter(acc["curated"][m]["mean"], i, marker="|", s=30, color="k")
    ax[1].axvline(ood["floor"]["mean"], ls=":", color="grey", lw=0.8)
    ax[1].set_yticks(ys); ax[1].set_yticklabels([short(m) for m in order], fontsize=5.6)
    ax[1].legend(handles=[Line2D([], [], ls="", marker="o", color="#555555", ms=3, label="OOD top-1, 95% session CI"), Line2D([], [], ls="", marker="|", color="k", ms=6, label="in-distribution test"), Line2D([], [], ls=":", color="grey", label="OOD majority class")], fontsize=5.4, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False)
    ax[1].set_xlabel("Top-1 (%)", labelpad=1); ax[1].set_title("(b) Held-out private sessions")
    fig.tight_layout(); FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig5_rig.pdf"); fig.savefig(FIG / "fig5_rig.png", dpi=200)
    print("ok", len(M), "macros")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--collect":
        collect(Path(sys.argv[2]))
    else:
        main()
