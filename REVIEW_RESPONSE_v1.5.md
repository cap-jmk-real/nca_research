# Response to aiXiv Official Agent review v1.5 (id 1755)

Manuscript: `paper/main.tex` · Branch: `cursor/review-majors-remaining-fdeb` · Related v1.4 items folded where unchanged.

| Review point (v1.5) | Status | Change or plan |
| --- | --- | --- |
| **Eq. (2): sign / normalization of Ã and L̃** | **Done (text)** | Rewrote perception as row-stochastic $A$ with low-pass $\bar{x}_i=\sum_j A_{ij}x_j$ and high-pass $\delta_i=\bar{x}_i-x_i$ (matches `nca_text/model.py`). Symmetric $\tilde{A},\tilde{L}$ moved to optional analysis only, not the forward pass. |
| **Eq. (10): justify $\beta$; calibrate $\tau_{\mathrm{escalate}}$?** | **Done (text)** | Both summands defined as dimensionless; $\beta=0.5$ documented as equal-weight heuristic. Explicitly **not** isotonic-calibrated on held-out escalation labels; SWE $0.76$ is a hand-tuned pilot threshold. |
| **Theorem 2: star assumption; empirical meta diameter** | **Done (text + audit JSON)** | Theorem restated for idealized `starSimpleGraph` vs 1D chain. Fixed-topology diameters on $V{=}5$ recorded in `stage0/stage0_results/topology_diameter_audit.json` (star $2$, 1D $4$, tree $3$; meta init softmax $\tau{=}0.1$ diameter $3$). Trained $\alpha$ checkpoints not in repo → **E3** in `EXPERIMENT_PLAN.md`. |
| **Meta topology 65.33% vs star 66.19%** | **Done (text)** | Results, abstract, table caption, and Discussion state meta **does not beat** star; bilevel search framed as recovering competitive graphs, not dominating hand-crafted star. |
| **Pilots $N{=}10$ / $N{=}30$ confounded** | **Done (text)** | Table~\ref{tab:live-benchmarks} relabeled preliminary; removed “Relative Gain” claims; added confounds paragraph; Methods pilot paragraph tightened. |
| **Repro: SVD + splits; Lean lemmas; release** | **Done (text)** | New “Feature construction” + expanded `make_splits.py` description; Lean lemma bullet list in §4; availability paragraph (GitHub public, HF scrubbed dataset/models, full corpus + LICENSE pending per `JULIAN_TODO.md`). |
| **Limitations: single dataset** | **Done (text)** | New Discussion § “Single-dataset classification evidence.” |
| **Second public classification benchmark** | **Planned** | `EXPERIMENT_PLAN.md` **E1** (BFCL-derived split, 3 seeds, held-out test, bootstrap). |
| **Router ablation (random / 1D / no-router)** | **Planned** | `EXPERIMENT_PLAN.md` **E2** (fixed FrogNano-4B + harness, 3 seeds). |
| **v1.4: Table 3 vs Table 1 meta 65.18% vs 65.33%** | **Done (prior + clarified)** | Table~\ref{tab:ablations} caption labels val-only 26.7k ablations (`EXP-20261006-*`); headline 65.33% test from `EXP-20261102-001` / Table~\ref{tab:feature-parity}. |
| **v1.4: Grattarola 2023 topology training** | **Deferred** | Not required for v1.5 “no new compute” scope; can add related-work sentence in a follow-up edit. |
| **Parameter budget $D$ mapping** | **Done (text)** | Explicit $D = d_{\mathrm{svd}} + 4 + \|\mathcal{V}\| + 1$ with 82 / 146 and 59,207 params cite. |

## Runtime estimates (from repo artifacts)

| Plan | Hardware | Estimate |
| --- | --- | --- |
| **E1** BFCL classification parity (4 topologies × 3 seeds @ 150k steps) | RTX 3060-class GPU | ~**2.4–2.6 h** GPU (scaled from ~46–51 s per 10k steps in `full_history_benchmarks.json`) + ~**15 min** CPU tabular/SVD |
| **E2** Pilot router ablation (4 routers × 3 seeds) | CPU or local GPU actuator | ~**4–6 h** total if actuator on GPU; ~**30–90 min** per (router, seed) dominated by LLM |
| **E3** Meta adjacency diameter export | CPU | **< 5 min** |

## Files touched

- `paper/main.tex` (PDF: run `cd paper && pdflatex main.tex && pdflatex main.tex` locally; cloud VM lacks `pdflatex`)
- `stage0/stage0_results/topology_diameter_audit.json`
- `EXPERIMENT_PLAN.md`
- `REVIEW_RESPONSE_v1.5.md` (this file)
