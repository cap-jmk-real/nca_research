# Paper review (paper-review skill), final pass, 2026-11-10

Re-scored from scratch on `paper/main.tex` + repository after the regression-audit pass (see `REGRESSION_AUDIT_1110.md`). Target 8.5 (Julian's request: 10).

**Verdict: 9.0 / 10 weighted. 0 Blockers, 0 Majors open. Referee call: minor revision. Code public (github.com/cap-jmk-real/nca_research, HTTP 200). klaut 10.0 (0/0/0).**
10/10 is **not** honestly reachable: novelty and rigour are limited by the work itself, not by the text (see bottom).

## Scores
| Dimension | Weight | Score | Justification |
|---|---|---|---|
| Claims supported & accuracy | 20% | 9.5 | Every number is a macro from a result file; headline "competitive and cheap" with default/tuned scope in abstract, results, discussion, conclusion; no equivalence/parity wording. Residual: the latency claim covers default configurations only (stated). |
| Rigour & reproducibility | 20% | 8.5 | Seed-clustered paired bootstrap, Holm per family, DL pooling, MDE stated, equal-budget search with trial logs, environment recorded, public code. Limits: 3 seeds, one-seed HPO selection, turn-level only, current models not tested OOD, escalation on the K=8 factorial model only. |
| Novelty & significance | 15% | 7.0 | Standard GNCA update applied as next-tool router with careful evaluation; stated as such. |
| Citations verified & coverage | 10% | 10 | 26/26 references re-verified this run (arXiv API title/authors/year; Crossref for 3 DOIs; 4 URLs HTTP 200). On-device function calling and tool retrieval now covered. |
| Clarity & language | 15% | 9.5 | Symbol collisions removed (Δ, α, τ, t, C, e). 20 extracted sentences > 40 words are mostly equations/tables/lists extracted by pdftotext; prose run-ons fixed. |
| Structure & venue fit | 10% | 9.5 | Abstract 263 words; contributions, scope, limitations, ledger, sources. Tables 5/6 float into the Discussion page (two-column layout). |
| Design | 10% | 9.5 | Booktabs, vector figures, consistent palette (colour-blind safety not separately checked); Fig. 1(d) label overlap fixed. |

Weighted: 0.2·9.5 + 0.2·8.5 + 0.15·7 + 0.1·10 + 0.15·9.5 + 0.1·9.5 + 0.1·9.5 = **8.98 ≈ 9.0**.

## Findings (open)
- Minor (rigour): 3 seeds; MDE up to 4.1 pp on the small benchmarks. Needs 5+ seeds (new runs).
- Minor (rigour): HPO selected on one seed, 20 trials. Needs multi-seed selection (new runs).
- Minor (rigour): no agent-level task-success evaluation. New experiment.
- Minor (rigour): current models not evaluated OOD; only an earlier pilot on other models (S7) is reported. New experiment on private sessions.
- Minor (rigour): latency of tuned models and end-to-end (with featurisation) not measured.
- Minor (claims): escalation results come from the K=8 factorial model.
- Minor (baselines): no Transformer/GNN/other-NCA/retrieval/fine-tuned function-calling baselines under this protocol; LLMs zero-shot.
- Nit (layout): Table 6 sits on the Discussion page.

## Fixed in this pass
Notation collisions; default-vs-tuned qualifiers on superiority sentences (Results accuracy/paired, CPU cost, abstract latency, limitations); MDE sentence; environment versions + `environment.json`; OOD-pilot limitation + source S7; extended baseline/escalation/window limitations; four verified related-work references; hard-coded 422–692 and 3–6 pp replaced by macros; two PR #8 ledger rows restored; LR `C` vs vocabulary 𝒞; "All runs used one RTX 3060" corrected (trees ran on CPU); Fig. 1(d) labels.

## Citation table
| Ref | Status |
|---|---|
| Mordvintsev 2020 (Distill DOI) | ok |
| Grattarola 2110.14237, Gala 2301.10497, Etcheverry 2609.36126, Pio-Lopez 2604.01932, Mittal 2608.02050, Lee 2603.10055 | ok |
| Kontschieder ICCV 2015 (Crossref), Frosst 1711.09784, Yang 1806.06988 | ok |
| Chen 2305.05176, Ong 2406.18665, Singla 2608.23078, Li 2610.02267 | ok |
| Chen&Li 2404.01744, Erdogan 2409.00608, Lin 2410.04587, Qu 2405.16089 (new) | ok |
| Patil 2305.15334, BFCL HF URL, Qin 2307.16789, Li 2304.08244, Hermes HF URLs (2) | ok |
| Chung 1997 (Crossref, online 1996-12), Katz CAV 2019 (Crossref pp. 443–452), Wang 2103.06624 | ok |

## What 10/10 would take
- Novelty → 10 is not reachable by editing; it would need a new method contribution (e.g. a router that beats tuned baselines significantly, or a new operator with a proof about trained weights).
- Rigour → 10: 5+ seeds for all models and multi-seed HPO selection; agent-level task-success study with enough tasks; OOD test of the current models on held-out sessions; end-to-end latency incl. featurisation and for tuned models; escalation on the primary model; one Transformer/GNN baseline under the same budget.
- With all rigour items done and novelty unchanged: max ≈ 0.2·10+0.2·10+0.15·7+0.1·10+0.15·10+0.1·10+0.1·10 = 9.55.
