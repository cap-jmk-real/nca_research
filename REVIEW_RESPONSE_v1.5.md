# Response to aiXiv Official Agent review v1.5 (id 1755)

Manuscript: `paper/main.tex` · Branch: `cursor/review-majors-remaining-fdeb` · Related v1.4 items folded where unchanged.

| Review point (v1.5) | Status | Change or plan |
| --- | --- | --- |
| **Eq. (2): sign / normalization of Ã and L̃** | **Done (text)** | Rewrote perception as row-stochastic $A$ with low-pass $\bar{x}_i=\sum_j A_{ij}x_j$ and high-pass $\delta_i=\bar{x}_i-x_i$ (matches `nca_text/model.py`). Symmetric $\tilde{A},\tilde{L}$ moved to optional analysis only, not the forward pass. |
| **Eq. (10): justify $\beta$; calibrate $\tau_{\mathrm{escalate}}$?** | **Done (text)** | Both summands defined as dimensionless; $\beta=0.5$ documented as equal-weight heuristic. Explicitly **not** isotonic-calibrated on held-out escalation labels; SWE $0.76$ is a hand-tuned pilot threshold. |
| **Theorem 2: star assumption; empirical meta diameter** | **Done (text + E3 pipeline)** | Theorem restated for idealized star. Init-only diameters in `topology_diameter_audit.json`. **Trained α:** run `./scripts/run_e1.sh` on engine → `e1_checkpoints/seed{N}_meta.pt`, `seed{N}_meta_adjacency.json`, auto `e3_meta_adjacency_audit.json`. |
| **Meta topology 65.33% vs star 66.19%** | **Done (text)** | Results, abstract, table caption, and Discussion state meta **does not beat** star; bilevel search framed as recovering competitive graphs, not dominating hand-crafted star. |
| **Pilots $N{=}10$ / $N{=}30$ confounded** | **Done (text)** | Table~\ref{tab:live-benchmarks} relabeled preliminary; removed “Relative Gain” claims; added confounds paragraph; Methods pilot paragraph tightened. |
| **Repro: SVD + splits; Lean lemmas; release** | **Done (text)** | New “Feature construction” + expanded `make_splits.py` description; Lean lemma bullet list in §4; availability paragraph (GitHub public, HF scrubbed dataset/models, full corpus + LICENSE pending per `JULIAN_TODO.md`). |
| **Limitations: single dataset** | **Done (text)** | New Discussion § “Single-dataset classification evidence.” |
| **Second public classification benchmark** | **Runnable (E1)** | `./scripts/run_e1.sh` — BFCL HF download, 3 seeds, holdout, resume, E3 auto. |
| **Router ablation (random / 1D / no-router)** | **Runnable (E2)** | `./scripts/run_e2.sh` — uses E1 checkpoints; `--dry-run` smoke. |
| **v1.4: Table 3 vs Table 1 meta 65.18% vs 65.33%** | **Done (prior + clarified)** | Table~\ref{tab:ablations} caption labels val-only 26.7k ablations (`EXP-20261006-*`); headline 65.33% test from `EXP-20261102-001` / Table~\ref{tab:feature-parity}. |
| **v1.4: Grattarola 2023 topology training** | **Deferred** | Not required for v1.5 “no new compute” scope; can add related-work sentence in a follow-up edit. |
| **Parameter budget $D$ mapping** | **Done (text)** | Explicit $D = d_{\mathrm{svd}} + 4 + \|\mathcal{V}\| + 1$ with 82 / 146 and 59,207 params cite. |

## Runtime estimates (from repo artifacts)

| Plan | Hardware | Estimate |
| --- | --- | --- |
| **E1** BFCL classification parity (4 topologies × 3 seeds @ 150k steps) | RTX 3060-class GPU | ~**2.4–2.6 h** GPU (scaled from ~46–51 s per 10k steps in `full_history_benchmarks.json`) + ~**15 min** CPU tabular/SVD |
| **E2** Pilot router ablation (4 routers × 3 seeds) | CPU or local GPU actuator | ~**4–6 h** total if actuator on GPU; ~**30–90 min** per (router, seed) dominated by LLM |
| **E3** Meta adjacency diameter export | CPU | **< 5 min** (auto after E1) |

## Provenance: paper **65.33%** Graph NCA (Meta-Learned) test top-1

| Question | Answer |
| --- | --- |
| **Source file** | `stage0/stage0_results/full_history_benchmarks.json` → model key `GraphNCA_meta`, seeds 0–2, field `test.top1` on **3,530** curated Hermes holdout turns (same split counts as `EXP-20261102-001`). |
| **SI registry** | `paper/si/experiments/EXP-20261102-001.json` exposes `metrics.headline_test_top1_mean_percent.GraphNCA_meta: 65.33` (mean of per-seed `test.top1`, not `test.boot_mean`). |
| **Per-seed `test.top1`** | seed 0: **65.61%**; seed 1: **65.10%**; seed 2: **65.27%** → arithmetic mean **65.3267%** → **65.33%** rounded to two decimals. |
| **Per-seed `test.boot_mean`** | 65.608, 65.080, 65.274 → mean **65.654%** (this is **not** the 65.33% table entry). |
| **Script / steps** | `stage0/run_full_history_benchmarks.py` at **`--steps 10000`** (see `EXP-20261102-001` `training.total_steps` and `repro.command`), **not** the paper’s 150k-step wording for Table 1 primary rows. |
| **Hardware** | `EXP-20261102-001` records **`device: cuda`**, note *engine-class GPU host* (~46.5 s train per topology per seed in JSON). **Machine hostname / GPU model string is not stored** in the artifact. |
| **Git pin in SI** | `repro.git_commit`: `1ceac61faaac21a32be2463d6b09eb987eef7807` |
| **Learned α / `.pt` checkpoints** | **Not in repo and not on engine** (per Julian addendum). JSON metrics only — **cannot trace saved `alpha` or `meta_seed*.pt` for this 65.33% run.** E1 on BFCL will produce new meta checkpoints + `e3_meta_adjacency_audit.json`; those are **BFCL-trained**, not the Hermes Table 1 run. |

**Flag:** Table 1 / Table 2 **65.33%** is traceable to **`full_history_benchmarks.json`** with the caveats above (10k-step run, no checkpoint archive, SI field mislabeled as boot_mean aggregate).

## Files touched

- `paper/main.tex` (PDF: run `cd paper && pdflatex main.tex && pdflatex main.tex` locally; cloud VM lacks `pdflatex`)
- `scripts/run_e1.sh`, `scripts/run_e2.sh`
- `stage0/e1_prepare_bfcl_splits.py`, `stage0/run_e1_benchmark.py`, `stage0/run_e2_router_ablation.py`
- `stage0/export_meta_adjacency.py`, `stage0/graph_topology_export.py`, `configs/dual_system_e2.yaml`
- `EXPERIMENT_PLAN.md`
- `REVIEW_RESPONSE_v1.5.md` (this file)
