# Paper review: rigour/novelty pass (branch research/rigour-novelty-1110), 2026-11-10 (re-scored 16:35 CET after the latency correction)

**Verdict: 9.0 / 10 weighted (target 8.5). 0 Blockers and 0 Majors open. Referee call: minor revision. Code is public (github.com/cap-jmk-real/nca_research, HTTP 200). klaut 10.0 with 0 findings (0 blockers, 0 majors, 0 minors).**

10/10 is still not honestly reachable. Novelty is limited by the work itself: the ablations show that the GNCA's own components do not drive accuracy. Agent-level task success needs an experiment that was not run.

Re-scored from scratch, without reusing the 9.0 from PR #9.

## Scores
| Dimension | Weight | Score | Justification |
|---|---|---|---|
| Claims supported & accuracy | 20% | 9.5 | Every number is a macro produced from result files (S1-S8). The headline ("competitive, not more accurate") matches Table 3, which is the 5-seed, multi-seed-selected main comparison. No parity or equivalence wording (klaut checks this). Latency now follows Julian's definition: headline = inference (model call) only, 0.559 ms for the GNCA; the 7.65 ms featurisation is reported separately as preprocessing in the abstract, Table 4 and §6.2. The v1.5 title ("Sub-Millisecond") is restored and scoped to inference in the ledger. Residual: the session proxy is optimistic, and the text says so. |
| Rigour & reproducibility | 20% | 9.0 | 13 models, the same 20-trial budget, top-5 re-scored with 3 seeds, 5 test seeds; seed-clustered paired bootstrap; Holm in 3 families plus ablation (16) and OOD (12) families; DerSimonian-Laird pooling; MDE 0.6-4.4 pp. OOD on 51 held-out sessions with a session bootstrap. Inference latency of all tuned models, with featurisation timed separately. Escalation and ECE on the primary model. Refit check of 39 model-seed pairs (34 exact; GPU nondeterminism up to 1.05 pp is disclosed). Limits: no agent execution; OOD covers one user; small test sets on 3 benchmarks; 20 trials; one machine. |
| Novelty & significance | 15% | 7.0 | Algorithm box, prior-work delta table and an exact structural proposition (unit-tested) make the contribution precise. It is still a GNCA update used as a router; the ablations show tying, gate, Laplacian block and normalisation have no accuracy effect. |
| Citations verified & coverage | 10% | 10 | 33/33 references re-verified this run: 26 arXiv ids by title and first author via the arXiv API, 5 DOIs via Crossref, 4 URLs HTTP 200. 7 new references (GGNN, MPNN, Universal Transformers, DEQ, Transformer, k-NN, selective classification) cover weight-tied message passing and the new baselines. |
| Clarity & language | 15% | 9.0 | Results lead with the main comparison; earlier protocols are summarised in one paragraph with full detail in App. D. The text is longer than PR #9 because the experiments are new. klaut finds no runaway-sentence finding. |
| Structure & venue fit | 10% | 9.0 | Abstract about 230 words, contributions, scope, limitations rewritten as findings, sources S1-S8, ledger. 21 pages (main text about 13). |
| Design | 10% | 9.5 | Booktabs; full-width tables; Fig. 5 uses the Okabe-Ito palette with redundant marker shapes, a legend outside the data and log ticks at 3/5/10/20/50; no overfull boxes. |

Weighted: 0.2·9.5 + 0.2·9.0 + 0.15·7.0 + 0.1·10 + 0.15·9.0 + 0.1·9.0 + 0.1·9.5 = **8.95 ≈ 9.0**.

Change from PR #9: rigour 8.5 → 9.0; clarity 9.5 → 9.0 and structure 9.5 → 9.0 because the paper is longer; novelty unchanged at 7.0.

## Findings raised in this pass and implemented
- Major (claims), klaut `best_claim_without_significance`: "competitive with methods refined …" was flagged. Rephrased so that it rests on the non-significant pooled differences in Table 3.
- Minor (claims), klaut `parity_without_equivalence_test` ×3: "level with" was replaced by "no significant pooled difference" in the results, discussion and intro.
- Latency correction (Julian, 16:23): earlier drafts of this pass headlined 8.19 ms, which counts featurisation as inference. Now: inference 0.559 ms (model call, as in v1.5 and S1), with featurisation (3.27-9.02 ms, unoptimised Python, needed by every feature-based model) reported separately. v1.5 title restored; ledger row says "Kept, scoped". Fig. 5a now plots inference latency.
- Minor (claims): inference ordering stated plainly. LogReg (0.102), MLP (0.238) and untied GNN (0.531 ms) are faster than the GNCA (0.559 ms); RF 4.44, HGB 9.78 and k-NN 51.3 ms are slower.
- Minor (structure), klaut `abstract_too_long` (253 words after the latency edit): trimmed.
- Minor (structure), klaut `abstract_too_long` (267 words) and colon signposting: abstract rewritten to about 230 words.
- Minor (claims), klaut `overclaim` ×2 (OOD "never used", GPU refit): now anchored to S8 and Table 4; refit numbers are macros from refit_check.jsonl.
- Minor (repro), klaut `inline_filename_artifact`: test-file and proposal paths removed from the prose.
- Minor (design), klaut `shrunk_single_column_table`: Table 5 is now full width without \resizebox.
- Minor (notation): ablation normalisation was written D^{-1/2}SD^{-1/2}; changed to Q^{-1/2}SQ^{-1/2}, which matches the paper's Q.
- Minor (design): Fig. 5 had overlapping labels, colliding log-tick labels and a clipped axis label. Rebuilt.
- Minor (layout): 3 overfull boxes (delta table, Eq. 7, Algorithm 1). Fixed.
- Minor (privacy): per-turn OOD labels of private sessions are no longer committed; only the "label is the majority tool" flag is. All macros are unchanged (verified with a diff).
- Bug: TeX macro names containing digits (`\numREscHerSel80`). Renamed.
- Minor (structure): two layers of results in the main text. Earlier protocols moved to App. D, with a 1-paragraph summary in §6.4.

## Open (minor; need experiments or approval)
- Claims: "Sub-Millisecond" in the title refers to inference only. The per-turn pipeline including our unoptimised featuriser takes 8.19 ms; this is stated in the abstract, §6.2 and the ledger. Other small models (LogReg, MLP, untied GNN) are also sub-millisecond.
- Rigour: no agent-level task success. Only an offline session proxy exists, defined only on Hermes.
- Rigour: OOD covers one user and 51 sessions.
- Baselines: no hosted, few-shot or fine-tuned LLM. The protocol is in paper/LLM_BASELINE_PROPOSAL_1110.md and needs paid-API approval.
- Baselines: no GAT/GCN/LSTM/other-NCA or fine-tuned small function-calling model.
- Novelty: limited by the work itself (see the score).

## Number ledger (headline numbers)
| Value | Where | Result file | Script |
|---|---|---|---|
| Pooled vs HGB +0.10 [-1.05, +1.25] | abstract, §6.1 | stage0/stage0_results/rigour_1110/per_turn_rigour.npz | scripts/build_rigour_tables_1110.py |
| Pooled vs kNN +2.21 [+0.65, +3.78] | abstract, §6.1 | same | same |
| Hermes vs HGB* -1.10, p_Holm 0.036 | abstract, §6.1 | same | same |
| OOD drop 11.76-14.40 pp; GNCA 56.61 [48.57, 63.54] | abstract, §6.2 | same (ood_* arrays, session ids) | same |
| Inference 0.559 ms (p95 0.676); featurisation 7.65 ms; pipeline 8.19 ms | abstract, §6.2, Table 4 | rigour_1110/latency.json | same |
| Escalation AUROC (entropy/composite/step), ECE | §6.3, Table 5 | per_turn_rigour.npz (entropy, pmax, stepnorm) | same |
| Ablations, pooled, 0/16 significant | §6.3, Table 5 | rigour_1110/ablation_correct.npz | same |
| 25/52 selections changed | §6.1 | rigour_1110/select/, hpo_1110 summary | same |
| Refit 34/39 exact, max 1.05 pp | Limitations | rigour_1110/artifacts/refit_check.jsonl | same |
| FLOPs 2.02 M | Prop. 3 | model config | same |

## Text-versus-code ledger (new parts)
| Paper | Code |
|---|---|
| Multi-seed selection: top-5 by seed-0 validation, seeds 1-2, select on 3-seed mean, 5 final seeds | stage0/run_rigour_1110.py phases `select`, `finals` |
| Transformer: pre-LN, learned positions, width 32-128, 1-3 layers, 2/4 heads | `_TinyTransformer`, `space()` in run_rigour_1110.py |
| Untied GNN: same perception/gate/readout, per-step (W1, W2) | `_UntiedGNN` |
| kNN: k 1-100, uniform/distance, cosine/euclidean, tab/cell | `space()` knn |
| Ablations: Q^{-1/2}SQ^{-1/2}, no (A-I)x, no gate, untied | stage0/run_ablation_1110.py |
| Latency: batch 1, 1 thread, 500 turns after 30 warm-up; inference = predict_proba / forward pass, featurisation timed separately | stage0/run_latency_1110.py |
| ECE with 15 bins; AUROC of entropy, step norm and their sum | scripts/build_rigour_tables_1110.py `ece`, escalation block |
| Prop. 3 (i)-(iii) | tests/test_gnca_properties_1110.py (4 tests pass) |
