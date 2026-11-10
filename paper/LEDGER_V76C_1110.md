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
