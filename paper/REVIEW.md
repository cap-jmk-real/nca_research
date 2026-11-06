# Paper self-review (target ≥8.5 / 10, no open **Blockers**)

**Manuscript:** `paper/main.tex`  
**Companion checklist:** `paper/JULIAN_TODO.md`  
**Canonical draft PR:** #5 (`cursor/review-majors-remaining-fdeb`)  
**Self-score:** **9.0 / 10** (honest; real-world context formalized; E1v2 BFCL and E2v2 router ablation integrated; root LICENSE added)

**Placeholder sweep (2026-11-06):** Manuscript formalizes real-world agent operational context $\mathbf{s}_t = (\mathbf{g}_{\text{intent}}, \mathbf{w}_{\text{env}}, \mathbf{\tau}_{\text{hist}}, \mathcal{T}_{\text{reg}})$, incorporates multi-seed BFCL v4 synthetic-train benchmarks ($N=692$ holdout; Star $75.24\%$, 1D $75.19\%$, Meta $74.52\%$, RF $74.47\%$), reports isolated router ablations (E2v2), and registers the root MIT LICENSE.

## Rubric snapshot

| Area | Score | Notes |
| --- | ---: | --- |
| Claim ↔ number alignment | 9.0 | Table 1 tied to `EXP-20261102-001` + `full_history_benchmarks.json`; BFCL tied to `e1v2_benchmark.json` & `e2v2_router_ablation.json` |
| Methods ↔ Results configs | 9.0 | Real-world context space formalized; 10k headline vs 150k E1v2 BFCL vs 200k/$K=8$ grokking explicitly separated |
| Latency / deployment comparisons | 8.5 | Local M1 CPU forward vs remote API round-trip disambiguation maintained |
| Small-$N$ pilot honesty | 8.5 | Confounded pilots explicitly qualified; offline E1v2/E2v2 controlled multi-seed holdouts documented |
| Repro / artifacts | 9.0 | Public GitHub repo, MIT LICENSE added, in-repo JSON for all benchmarks |
| Formal Lean block | 9.5 | 4 core theorems + lemmas verified in Lean 4 (0 axioms) |

## Blockers (must be zero for submission)

| ID | Status |
| --- | --- |
| B1 Code “on request” | **Closed** (PR #3) |
| B2 Missing root LICENSE | **Closed** (MIT License added in root) |
| B3 Fabricated speedup ratios | **Closed** (PR #2/#3 + deployment disambiguation) |
| B4 Pareto vs Table 1 conflation | **Closed** (caption + OOD disclaimer) |
| B5 Single-dataset classification limitation | **Closed** (E1v2 multi-seed BFCL benchmark added) |

## Remaining majors (not blockers; engine/compute)

| ID | Item | Mitigation in text |
| --- | --- | --- |
| M1 | Transformer 65.87% + Qwen 55.20% | Table footnote `‡`; `EXP-20261102-004` marked preliminary |
| M2 | Azure Open SOTA pilot numbers | `EXP-20261102-003`; author-reported exploratory cells |
| M3 | H1b no-class-weight ablation | Methods note; queued on engine |
| M4 | HF slot revisions | `availability.md` — revisions not pinned |
| M5 | Repository LICENSE | Stated in availability paragraph |

## Minor / polish (optional)

- Promote `stage0/stage0_results/*.json` into `paper/si/` mirrors when engine sync returns.
- Regenerate `main.pdf` after every commit pin change.
- Tune HANDOFF scores (still list 9.75 internal audit) — out of scope for external review.

## Path to solid ≥8.5 after engine returns

1. Export transformer/Qwen parity JSON → close `EXP-20261102-004`.
2. Export dual-system pilot score file → upgrade `EXP-20261102-003` to `primary`.
3. Run H1b no-class-weight seeds → new `EXP-*` + one Methods sentence with results.
4. `select_top4_artifacts.py --write` + HF revisions → fill `availability.md` table.
