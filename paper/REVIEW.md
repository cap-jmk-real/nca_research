# Paper self-review (target ≥8.5 / 10, no open **Blockers**)

**Manuscript:** `paper/main.tex`  
**Companion checklist:** `paper/JULIAN_TODO.md`  
**Canonical draft PR:** #5 (`cursor/review-majors-remaining-fdeb`)  
**Self-score:** **8.6 / 10** (honest; preliminary pilot and offline baseline rows labeled via SI)

**Placeholder sweep (2026-11-02):** Manuscript abstract/body/captions no longer use “correct $k/30$”, ledger-export TODOs, or re-export-pending wording; BFCL irrelevance cites **$90.0\%$ on $N=30$** from `EXP-20261102-003` without inventing a $k/N$ ledger.

## Rubric snapshot

| Area | Score | Notes |
| --- | ---: | --- |
| Claim ↔ number alignment | 8.5 | Table 1 tied to `EXP-20261102-001` + `full_history_benchmarks.json`; split sizes fixed (3,530 test / 64,670 total) |
| Methods ↔ Results configs | 9.0 | 150k/$K=4$ vs 200k/$K=8$ grokking explicitly separated |
| Latency / deployment comparisons | 8.5 | No cross-hardware speedup ratios; pilot vs API labeled |
| Small-$N$ pilot honesty | 8.5 | Raw counts where stated ($8/10$, $7/7$, $14/14$); BFCL $90.0\%$ on $N=30$ via \texttt{EXP-20261102-003} (no $k/N$ in repo) |
| Repro / artifacts | 8.0 | Public GitHub + in-repo JSON for main benchmarks; HF slot revisions open |
| Formal Lean block | 9.5 | Unchanged; clone + commit pin in appendix |

## Blockers (must be zero for submission)

| ID | Status |
| --- | --- |
| B1 Code “on request” | **Closed** (PR #3) |
| B2 Fabricated speedup ratios | **Closed** (PR #2/#3 + this pass) |
| B3 Pareto vs Table 1 conflation | **Closed** (caption + OOD disclaimer) |
| B4 Wrong ablation EXP ids / mislabeled 65.18% “meta” row | **Closed** (this pass) |
| B5 Dataset count 23,741 unsupported | **Closed** (this pass) |

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
