# v7.6 → v7.6-concise ledger (2026-11-10)

Base: v7.6 + Julian's 04:35 main.tex edit (commit 1a7186c, branch sync/engine-wip-0435-20261110).
All numbers in `paper/main.tex` are macros from `paper/generated/numbers.tex`
(`scripts/build_review_tables_1110.py`) and `paper/generated/numbers_v76c.tex`
(`scripts/build_v76c_extras_1110.py`). Paths below are relative to `stage0/stage0_results/`.

## Result sources (paper Table 5)

| Id | Files | Used for |
|---|---|---|
| S1 | `review_1110/multibench.json`, `review_1110/stats.json`, `review_1110/correct_by_seed*.npz`, `review_1110/input_matched_trees.json`, `review_1110/cpu_cost.json`, `review_1110/jacobian_audit.json`, `review_1110/lean_axioms.txt` | Table 2, Table 3, Fig. 2, Fig. 3, latency/size, Jacobian audit, Lean counts, parameter counts, validation-selected topologies, Hermes macro-F1 |
| S2 | `reviewer_suite_summary.json` → `hermes_factorial_and_contractive` | Table 4 (factorial), contractive 61.99, learning curve 68.15→70.79 (`grokking_curves_3seed.Tree_K4_LS`), escalation AUROC 0.730/0.732/0.441, θ=0.76 (11.4% / 72.45%), θ=0.50 (72.6% / 90.48%), step ratio 1.20 |
| S3 | `full_history_benchmarks.json` | Protocol sensitivity: HGB 69.47, RF 67.88, star 66.19, tree 65.27 |
| S4 | `e1v2/run/e1v2_benchmark.json` | BFCL 150,000-step run 75.29–76.40 vs HGB 78.18 |
| S5 | `baselines_20261026/teacher_qwen3.5_9b_curated_test.json`, `teacher_gemma4_12b-it-q4_K_M_curated_test.json`, `sota_followup_summary.json`, `B_tx_curated_s*.json` | Qwen 35.01%/378 ms, Gemma 38.02%/848 ms; earlier Transformer 65.87±2.00 (276,749 params) cited in the ledger only |

## v7.6 claims removed or replaced

| v7.6 claim | Action | Replacement / source |
|---|---|---|
| Title "From Grokking to …" | removed | gradual learning curve, S2 `grokking_curves_3seed` |
| Headline K=8 68.92% "statistically non-inferior" to HGB (TOST) | removed | K=4 tree primary; difference tests only (S1 `stats.json`) |
| Per-turn bootstrap over seeds×turns, Cohen's d/h, power 0.94, LME +3.16 pp | replaced | seed-clustered bootstrap, Holm 12+8, DerSimonian–Laird (S1 `stats.json`) |
| "Pre-registered" | removed | no registry exists; Scope paragraph says choices were informed by exploratory runs |
| 50-trial Hyperband for all models | removed | no tuning code exists; fixed configs in `stage0/run_review_resolution_1110.py` |
| β=0.5, "calibrated" score in [0,1.50], 10-bin reliability audit, decile error rates | replaced | β=1 as in `stage0/run_reviewer_experiments.py`; AUROC + two operating points (S2) |
| Damped update α_d=0.58, γ≤0.35, Bernoulli(p=0.8) masks | replaced | `nca_text/model.py` GraphNCA update (Eq. 4) |
| Causal lower-triangular operator / directed temporal filtering | replaced | `build_adjacency`: undirected, self-loops, row-normalised (Eq. 1–2) |
| α_self × K grid, fig3_pareto, fig4_ablation | removed (hard-coded values in plot scripts) | Table 4 from S2 |
| Contractive variant "dyadic premise on 100% of turns", ρ≤0.47, ratio max 0.437, L_deploy≤0.8 | removed | keys `perturbation_contraction_ratio_*` / `fraction_dyadic_contractive_lt_0_5` in `reviewer_suite_summary.json` are not written by any repo code (code writes `step_ratio_*`); accuracy 61.99 kept (S2); Jacobian audit mean 2.428, max 3.336, 0% < 1, n=42,360 (S1 `jacobian_audit.json`) |
| Bridge constants 0.887/0.890, 62,645 turns, radius 5.14, 1.64 | removed | no result file (0.887 only appears as a learning rate 0.000887 in `grokking_deep_nca_results.json`) |
| "Ten theorems, zero axioms" | replaced | 28 theorems, 13 axiom-free, rest standard axioms, no sorry (S1 `lean_axioms.txt`) |
| Tree concordance 76.63%, κ=0.5628 | removed | value exists in `theorem6_bridge_audit.json` but no repo script produces it; agreement with a separately fitted tree does not test the theorem |
| GCN, LGCA, E(n)-GNCA, Translucent, TextNCA, ARTS, Contriever, Hammer baselines | removed | no results in repo (only text patches `scripts/apply_paper_v4_6.py`, `apply_paper_v4_7.py`) |
| Transformer "1.2M params, 65.87%, 3.8 ms" | removed | S5 `sota_followup_summary.json` B_tx 65.87±2.00, 276,749 params (`B_tx_curated_s0.json`), old protocol; 1.2M/3.8 ms have no file |
| Qwen2.5-1.5B constrained decoding 55.20%; frontier-API latencies | removed | local LLMs kept (S5) |
| Linear classifier 60.71%; kFLOPs; RAM 142.6 MB | replaced | logistic regression row (S1 `multibench.json`); serialised sizes (S1 `cpu_cost.json`) |
| Synthetic 10k-trajectory and probe-leakage numbers | removed | no result file; Proposition 2 (analytic) |
| Xeon 8375C 21.2 ms / 60.6×, multi-hardware audit, RF 18.2M nodes | replaced | Ryzen 9 7950X 1-thread latencies, 307,118 RF nodes (S1 `cpu_cost.json`) |
| HGB 69.47% as main Hermes comparison | kept as sensitivity | S3; main table S1 |
| Parameter breakdown (8,846 "invariant" part, inconsistent totals) | replaced | Eq. 5: 46,752 + 3,072 + 64 + 1,053 + 8,224 + 17 = 59,182, asserted against `multibench.json` |
| β described as variance-equalising; 1/√H scaling ablation table | replaced | χ_H derivation + statement that β has no natural value |
| Hugging Face model link (401); ARTS, Petric 2305.18567, Mrowińska, wrong TextNCA/BraiNCA/fast-models metadata | removed/fixed | GitHub link (HTTP 200); 22 references resolved via arXiv API / Crossref on 2026-11-10 |

## v7.6 numbers that survive (with source)

68.90/68.92 (K4/K8, S2), 67.88 chain K4 and 67.40 K2 (S2), 61.99 contractive (S2), 11.4%/72.45% at θ=0.76 and 72.6%/90.48% at θ=0.50 (S2), AUROC 0.730/0.732/0.441 (S2), Qwen 35.01%/378 ms and Gemma 38.02%/848 ms (S5), Hermes 57,675/3,465/3,530 (S1), 59,182 parameters (S1), HGB 69.47 full-history (S3).

## Hyperparameter search (2026-11-10, phase 2) — S6 `hpo_1110/`

Run: `stage0/run_hpo_1110.py` (commit a495882) via `scripts/hpo_1110_run.sh` on engine (RTX 3060), 08:23–10:57 CET with one reboot interruption (`hpo_1110/run_meta.json`). Numbers: `scripts/build_hpo_tables_1110.py` → `paper/generated/numbers_hpo.tex` (331 macros), `tab_hpo.tex`, `tab_hpo_paired.tex`, `tab_hpo_configs.tex`, `figures/fig4_hpo.pdf`, from `hpo_1110/hpo_summary.json`, `hpo_1110/correct_by_seed_hpo.npz`, `hpo_1110/stats_hpo.json` (+ untuned S1 files). No untuned (S1) number changed.

| Claim / location | Before | After | Source key |
|---|---|---|---|
| Limitation 'No hyperparameter search' (Sec. 7) | fixed configs, no search | 'Small search budget': 20 trials/model/benchmark | run_meta.json, hpo_summary.json trials_per_model |
| Ledger row '50-trial Hyperband' | Removed; no search was run | Replaced: equal-budget random search, 20 trials | hpo_summary.json |
| Search size | — | 800 trials, 10 models × 4 benchmarks; CPU 1.6 h, GPU 1.3 h fit time, wall 1.8 h | stats_hpo.json search_wall_s; run_meta.json wall_h |
| Trial-0 reproduction of defaults | — | ≤1.16 pp (all but Hermes HGB/HGB*: 6.26 pp, sklearn auto early stopping); default selected 0/40 | stats_hpo.json default_reproduction* |
| Abstract/intro/conclusion pooled GNCA−RF | +2.55 [+0.67, +4.43] p=0.008 (default) | tuned +1.99 [-0.30, +4.27] p=0.089 (n.s.); default kept | stats_hpo.json pooled_primary.rf |
| Pooled GNCA−HGB | +0.75 (default) | tuned -0.11 [-2.31, +2.08] p=0.920 | pooled_primary.hgb |
| Pooled GNCA−MLP | -0.26 (default) | tuned +0.09 [-1.55, +1.72] p=0.916 | pooled_primary.mlp |
| Pooled GNCA−RF* | +2.60 (default) | tuned +0.98 [-1.45, +3.41] | pooled_secondary.rf_cell |
| Pooled GNCA−HGB* | +0.12 (default) | tuned -0.78 [-2.45, +0.89] | pooled_secondary.hgb_cell |
| Significant tuned comparisons | — | primary 3/12: API-Bank RF +5.11 (pHolm 0.002), BFCL HGB -2.84 (0.027), BFCL MLP -2.02 (0.027); secondary 2/8: Hermes HGB* -1.33 (0.006), API-Bank RF* +4.55 (0.002) | stats_hpo.json primary/secondary |
| Discussion 'ahead of a RF when pooled' | unqualified | holds with defaults, not significant after tuning | pooled_primary.rf |
| Tuned GNCA tree params | — | 25,678–109,038 | hpo_summary.json final[0].n_params |

### Tuned test top-1 and tuning gain (tuned − default, pp; † Holm p<0.05 over 40)

| Model | Hermes | BFCL | ToolBench | API-Bank | Pooled gain [95% CI] |
|---|---|---|---|---|---|
| logreg | 65.65±0.07 (+0.43) | 76.69±0.33 (+2.17) | 65.56±1.58 (+8.85†) | 85.30±0.55 (+6.07†) | +4.04 [+0.73, +7.34] p=0.017 |
| rf | 69.03±0.22 (+1.10†) | 75.15±0.25 (+0.96) | 73.46±0.63 (-0.87) | 82.89±0.52 (+2.76†) | +0.98 [-0.23, +2.20] p=0.113 |
| hgb | 68.32±0.23 (+2.12†) | 79.39±0.85 (+1.20) | 76.70±0.59 (+0.63) | 85.37±0.78 (+0.62) | +1.18 [+0.47, +1.90] p=0.001 |
| rf_cell | 70.00±0.28 (+0.96) | 75.43±0.29 (+2.79) | 75.91±0.55 (+2.53) | 83.44±0.36 (+1.79) | +1.72 [+0.80, +2.64] p=0.000 |
| hgb_cell | 70.50±0.12 (+3.69†) | 78.13±0.47 (-0.05) | 77.73±0.72 (+0.47) | 85.78±0.66 (+0.14) | +1.09 [-0.90, +3.09] p=0.282 |
| mlp | 69.50±0.28 (-0.42) | 78.57±0.17 (+0.96) | 74.41±0.63 (+0.24) | 85.85±0.67 (-0.62) | -0.08 [-0.71, +0.56] p=0.810 |
| gnca_chain | 69.16±0.43 (+0.05) | 76.73±0.29 (+0.72) | 75.75±0.14 (+0.63) | 86.68±0.63 (+1.59) | +0.47 [-0.18, +1.12] p=0.152 |
| gnca_star | 69.76±0.07 (+0.10) | 76.97±0.22 (+0.67) | 74.56±1.10 (-0.79) | 85.99±0.63 (-1.10) | -0.23 [-0.93, +0.46] p=0.508 |
| gnca_tree | 69.17±0.13 (+0.07) | 76.54±0.93 (+0.14) | 75.36±0.95 (-0.95) | 87.99±0.42 (+2.00) | +0.37 [-0.69, +1.43] p=0.496 |
| gnca_learned | 68.84±0.34 (-0.60) | 76.40±0.33 (-0.43) | 75.04±1.58 (-0.71) | 87.23±0.24 (+0.48) | -0.46 [-0.95, +0.02] p=0.063 |

Keys: `hpo_summary.json` → `benchmarks.<bench>.<model>.test_top1_mean/sd`; `stats_hpo.json` → `gain.<bench>/<model>`, `pooled_gain.<model>`. Selected configurations: `hpo_summary.json` `best_config` (paper Table 7). Validation-selected tuned topology: Her star 69.76, Bfcl star 76.97, Tb chain 75.75, Ab learned 87.23 (`stats_hpo.json val_selected_gnca`).

Layout-only changes in the same pass: float parameters (dbltopnumber 2, fractions), Table 3 source moved after Table 2; new Appendix C (search spaces, Table 7); source S6 added to Table 8.

