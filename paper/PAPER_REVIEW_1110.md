# paper-review (skill) — NCA, 2026-11-10 (CET)

Target 8.5. Venue type: aiXiv preprint / ML workshop-style empirical paper.

## Before: v7.6 (engine 6653231, aiXiv v7.6)
Verdict 4.75 uncapped → capped at 6.0 by open blockers; effective **4.75/10** (cap not binding). Referee: reject / major revision.
Blockers: (1) ~25 reported numbers with no result file, injected by `scripts/apply_paper_v4_*.py`; (2) causal lower-triangular operator in the text vs undirected row-normalised adjacency in code; (3) paired tests pooled seeds × examples; (4) "non-inferiority" with CI lower bound −1.61 vs margin 1.0; (5) nonexistent/misattributed references (ARTS, Petric 2305.18567, Mrowińska; wrong TextNCA/BraiNCA IDs), HF link 401.
Scores: claims 3 · rigour 4 · novelty 6 · citations 3 · clarity 5 · structure 7 · design 7.

## After: branch research/aixiv-review-resolution-1110 (7 pages, clean tectonic build, no overfull/undefined)
| Dimension | W | Score | Why |
|---|---|---|---|
| Claims supported | 20% | 9.0 | Every number is a macro generated from result files; claims hedged; ledger of removed claims |
| Rigour & reproducibility | 20% | 8.5 | 4 benchmarks × 3 seeds, held-out tests, seed-clustered paired bootstrap, Holm (12 + 8), DL pooling, input-matched tree baselines, Jacobian audit, Lean axiom audit; no tuning, 3 seeds |
| Novelty & significance | 15% | 6.5 | Method ≈ same-size MLP in accuracy; contribution is a careful small-router evaluation |
| Citations | 10% | 9.5 | 19 references, all resolved (arXiv API / Crossref / HTTP 200) this run |
| Clarity & language | 15% | 8.5 | Short, direct; some sentences > 35 words |
| Structure & venue fit | 10% | 9.0 | Contributions, protocol, limitations, data/AI/COI statements |
| Design | 10% | 9.0 | Vector figures from result files, Okabe-Ito, booktabs, fonts ≥ 5.6 pt |
Weighted total **8.5/10**, no open blockers. Referee: minor revision. The remaining ceiling is novelty, not evidence.

Remaining majors/minors (not blockers): per-benchmark tuning for all models (~1–2 h GPU); 5+ seeds (~30 min GPU);
agent-level task-success evaluation (multi-hour); end-to-end latency incl. featurisation (~10 min CPU).
