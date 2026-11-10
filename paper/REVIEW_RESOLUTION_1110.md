# NCA paper: review resolution, 2026-11-10

Sources: 46 public aiXiv reviews (v1.4–v7.6, IDs 1745–1879, ratings 5–6), split into 349 atomic points and clustered into
17 themes (`/workspace/nca_audit/aixiv_points.json`); own integrity audit of v7.6 (this run); own paper-review runs
(`papers/nca-research/draft/review-1..6.md`, Oct 26; `paper/REVIEW.md` self-score 9.0); klaut runs (Nov 3; this run).
Status refers to branch `research/aixiv-review-resolution-1110`. Evidence = file or section in that branch.

| # | Point (canonical) | Source (first–last version, n points) | Status | Evidence / change |
|---|---|---|---|---|
| A1 | Many v7.6 numbers have no result file (Jacobian 0.887/0.890, 62,645 turns, LME +3.16, TOST, Holm p, Cohen's h, power 0.94, 10k trajectories, grammar-constrained 55.20%, ARTS latencies, RAM 142.6 MB, RF 18.2M, Xeon 8375C 21.2 ms/60.6×) | Own integrity audit; aiXiv T13 (17), T04 (28) | resolved | All removed (App. B ledger). Every number now a macro in `paper/generated/numbers.tex` written by `scripts/build_review_tables_1110.py` from `stage0/stage0_results/review_1110/*.json` |
| A2 | Numbers injected via `scripts/apply_paper_v4_*.py` text patches | Own audit; klaut now (blocker) | resolved | Paper no longer produced by patch scripts; scripts kept for history |
| A3 | Paired bootstrap pooled seeds × examples as independent | Own audit; aiXiv T03 (27) | resolved | `cluster_boot` in build script: per-example correctness averaged over seeds, examples resampled (Sec. 5, Table 2) |
| A4 | Holm/multiple comparisons not real | Own audit; aiXiv T03 | resolved | Holm over 12 primary + 8 secondary comparisons (Table 2) |
| A5 | Hermes "non-inferiority" with CI lower −1.61 vs 1.0 pp margin | Own audit; aiXiv T04 | resolved | No NI/equivalence claims; wording "not significantly different (CI)"; conclusion says equivalence untested |
| A6 | Headline 68.92% came from K=8 while K=4 "validation-selected" | Own audit | resolved | Headline is K=4 tree GNCA, 3 seeds (Table 1); ledger notes it |
| A7 | Paper says causal lower-triangular operator; code is undirected row-normalised | Own audit; aiXiv T06 (5), T12/T13 | resolved | Method §3 describes `build_adjacency` as implemented; Fig. 1 regenerated from code adjacency |
| A8 | Contraction "certificate" for trained router | aiXiv T05 (66, v1.4–v7.6) | resolved | §4: theorems describe idealised automata; measured Jacobian spectral norm mean 2.428, max 3.336 over 42,360 Jacobians (`jacobian_audit.json`); contractive variant 61.99 vs 68.90 |
| A9 | "Zero axioms" | Own audit | resolved | App. A: `#print axioms` for 28 theorems, 13 axiom-free, rest standard axioms only, no sorry |
| A10 | Tree-simulation theorem overstated (76.63% concordance) | aiXiv T07 (6) | resolved | §4 item 2 states it as representability for a symbolic tree; concordance claim removed |
| A11 | Novelty overstated / positioning vs GNCA, neural trees | aiXiv T01 (40) | partial | Claims narrowed to "small router + audited evaluation"; related work differentiated from Grattarola et al. Novelty remains modest (method ≈ MLP) — inherent |
| A12 | Baselines under-tuned; 50-trial Hyperband claimed without code | aiXiv T02 (28); own audit | partial | Tuning claim removed; protocol states no method was tuned (§5, Limitations). Per-benchmark tuning for all models is open (est. 1–2 h GPU) |
| A13 | Tree baselines get fewer input features | Own audit; aiXiv T02 | resolved | New input-matched RF*/HGB* run (`input_matched_trees.json`, 111 s CPU, 3 seeds × 4 benchmarks); secondary family in Table 2: pooled vs RF* +2.60 [+0.12, +5.07], vs HGB* +0.12 [−2.02, +2.25] |
| A14 | Only 3 seeds | aiXiv T03 (v7.6) | partial | 3 seeds kept, stated as limitation; CIs at example level. 5+ seeds open (~30 min GPU) |
| A15 | Single Hermes corpus / external validity | aiXiv T10 (35) | resolved | Four benchmarks: Hermes, BFCL, ToolBench, API-Bank (5,127 test turns) |
| A16 | Pilot/agent-loop results underpowered | aiXiv T11 (14) | resolved | Pilots removed; Limitations says turn-level only. Agent-level evaluation open (multi-hour) |
| A17 | Escalation β hand-tuned; components not ablated | aiXiv T08 (10) | resolved | β fixed a priori (stated); ablation: composite 0.730, entropy 0.732, step-norm 0.441 → recommend entropy alone |
| A18 | Latency on Xeon 8375C without logs; pipeline excluded | aiXiv T09 (2); own audit | resolved | Measured on Ryzen 9 7950X, 1 thread (`cpu_cost.json`); framework overhead stated; limitation "one machine" |
| A19 | RF "18.2M nodes"/compression ratio | aiXiv T07; own audit | resolved | 307,118 nodes from code; parameter-count definition in Table 1 caption |
| A20 | Practical significance of 1–3 pp gains | aiXiv T16 (14) | resolved | "Headline is modest" paragraph; CPU-cost framing; MLP equal |
| A21 | Notation (Laplacian sign, A−I) | aiXiv T12 (49), T13 | resolved | Laplacian/spectral material removed; single update equation (Eq. 1) |
| A22 | Dense exposition | aiXiv T12 | resolved | 7-page rewrite; klaut runaway-sentence nit remains (25/164 > 35 words) |
| A23 | Reproducibility: HF link 401, missing artifacts | aiXiv T14 (7); own audit | resolved | Public GitHub link (HTTP 200); e1v2 checkpoints listed with SHA-256 in `ARTIFACT_MANIFEST.json`, stored on engine + restic (nbg1, fsn1) |
| A24 | Wrong/nonexistent citations (ARTS, Petric 2305.18567, Mrowińska, TextNCA ID, BraiNCA author, fast-models author) | Own audit; klaut now | resolved | 18 references, all resolved via arXiv API/Crossref this run |
| A25 | "Pre-registered" without registry | Own audit; klaut now | resolved | Removed; App. B |
| A26 | Limitations missing / failure modes / dynamic tool sets | aiXiv T16, T99 | resolved | §7 Limitations incl. fixed 13-tool vocabulary |
| A27 | MLP description wrong ("one hidden layer"); BFCL split wording | Own verification this run | resolved | §5 corrected to 2×76 LayerNorm/GELU/dropout; BFCL = label-stratified, identical-goal groups kept together |
| O1 | Single seed; no held-out curated test (review-1/2) | Own paper-review, Oct | resolved | 3 seeds, held-out test splits on 4 benchmarks |
| O2 | LLM baselines weak/unevaluated (review-1..6) | Own paper-review | partial | Local zero-shot Qwen3.5-9B 35.01%, Gemma-4-12B 38.02% on Hermes test (intro); DeepSeek/Llama API runs not cited |
| O3 | Paired CIs between small models (review-3..6) | Own paper-review | resolved | Table 2 |
| O4 | Code private (review-2) | Own paper-review | resolved | Public repo, MIT |
| K1 | Klaut 11-03: split_labels, number_conflict, arithmetic FP | klaut before | resolved / FP fixed | Labels per split in Table 1; Wilson-CI FP fixed in klaut |
| K2 | Klaut now on new paper: β justification, prior-art delta, overclaim anchors, scaling scope, abstract colons | klaut now (deterministic) | resolved | Edits in §2, §3, §1, §7, abstract |

Open experiments (not started; listed with estimates): per-benchmark tuning for all models (~1–2 h GPU on RTX 3060);
5+ seeds (~30 min GPU); agent-level task-success evaluation (multi-hour, needs harness); full pipeline latency incl. TF-IDF/SVD featurisation (~10 min CPU).
