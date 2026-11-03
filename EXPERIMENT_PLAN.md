# Experiment plan (aiXiv review v1.5)

Engine commands assume repo root, `NCA_DATA_ROOT` (default `./.nca_data`), and CUDA PyTorch on the RTX 3060 host. **No private Hermes corpus** in E1/E2. Results land under `stage0/stage0_results/`.

## E1 — BFCL public classification (+ meta checkpoints + automatic E3)

**Data:** `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (auto-download via `huggingface_hub`). Split name: `bfcl_e1`.

**Full engine run (3 seeds, 150k steps, all topologies, holdout test, resume, E3):**

```bash
cd /path/to/nca_research
export NCA_DATA_ROOT="${NCA_DATA_ROOT:-$PWD/.nca_data}"
./scripts/run_e1.sh
```

Equivalent Python:

```bash
python stage0/e1_prepare_bfcl_splits.py   # if splits missing
python stage0/run_e1_benchmark.py \
  --seeds 0,1,2 \
  --steps 150000 \
  --topologies tree,meta,star,1d \
  --out stage0/stage0_results/e1_benchmark.json
# On success, automatically runs stage0/export_meta_adjacency.py → e3_meta_adjacency_audit.json
```

**Smoke (<2 min, CPU/GPU):** trains `1d`, `meta`, and `star` (150 steps, seed 0) so E2 dry-run can load the star actuator.

```bash
./scripts/run_e1.sh --dry-run
./scripts/run_e2.sh --dry-run
# or: bash scripts/test_stage0_e1_e2_dry_run.sh
```

**Outputs:**

| Artifact | Path |
| --- | --- |
| Aggregate metrics | `stage0/stage0_results/e1_benchmark.json` |
| Per-seed | `stage0/stage0_results/e1_seed{0,1,2}.json` |
| Checkpoints (resume) | `stage0/stage0_results/e1_checkpoints/seed{N}_{tree,meta,star,1d}.pt` |
| Learned α / A (meta) | `stage0/stage0_results/e1_checkpoints/seed{N}_meta_adjacency.json` |
| E3 audit (auto) | `stage0/stage0_results/e3_meta_adjacency_audit.json` |

**Runtime (RTX 3060):** ~46–51 s per 10k steps per topology in `full_history_benchmarks.json` → ~**2.4–2.6 h** GPU for 4×3 full runs + ~15 min tabular/SVD CPU.

---

## E2 — Router ablation (fixed star actuator)

**Routers:** `random`, `nca1d`, `none`, `learned` (meta). **Actuator:** fixed `GraphNCA` star weights from E1 (`seed{N}_star.pt`). **Holdout:** `bfcl_e1/test`.

```bash
./scripts/run_e2.sh
# or smoke:
./scripts/run_e2.sh --dry-run
```

**Outputs:** `stage0/stage0_results/e2_router_ablation.json`, `e2_seed{N}.json`

**Runtime:** Dominated by loading checkpoints + forward on test split; typically **minutes** after E1 completes (no LLM required in default eval).

Optional LLM actuator: set `E2_ACTUATOR_OLLAMA_MODEL` on the host only (never commit); see `configs/dual_system_e2.yaml`.

---

## E3 — Meta adjacency / diameter export

Normally **automatic after E1**. Manual re-export:

```bash
python stage0/export_meta_adjacency.py \
  --checkpoint-dir stage0/stage0_results/e1_checkpoints \
  --seeds 0,1,2 \
  --out stage0/stage0_results/e3_meta_adjacency_audit.json
```

**Runtime:** **< 5 min** CPU.

---

## E4 — Escalation hyperparameter sensitivity (pilot-only; no headline GPU)

**Scope:** $\beta$ in the $\tau_{\text{escalate}}$ score and cutoff thresholds for Dual-System pilots only (does not affect curated test top-1 in Table 1).

**Planned grid (after E2 harness freeze):**

| Knob | Values | Metric |
| --- | --- | --- |
| $\beta$ | 0.25, 0.5, 1.0 | BFCL irrelevance triage + SWE repair trigger rate on validation probes |
| $\tau_{\text{escalate}}$ cutoff | sweep on curated val abstain precision target | precision@escalation (no public label set yet) |

**Status:** Not run in repo; document results in `EXP-20261102-003` follow-up or new EXP id when pilot harness is frozen.

---

## Paper Hermes 65.33% meta (provenance only — not E1)

See `REVIEW_RESPONSE_v1.5.md` § Provenance. That number is **not** reproduced by E1; E1/E3 meta checkpoints are BFCL-trained.
