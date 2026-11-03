# Experiment plan (aiXiv review v1.5 — requires new compute)

Commands assume repository root `/agent/repos/nca_research`, Python venv with PyTorch, and curated splits under `$NCA_DATA` (see `stage0/common.py`). Holdout evaluation uses the frozen **curated test split** (3,530 turns; no `session_id` overlap with train) and **seeds 0, 1, 2** unless noted.

## E1 — Second public next-tool classification benchmark (BFCL-derived)

**Goal:** Report top-1 / top-3 with the same bootstrap protocol as Table 2 (`stage0/run_full_history_benchmarks.py`) on a dataset distinct from Hermes.

**Protocol sketch:**

1. Build a BFCL v3/v4 irrelevance + simple AST classification jsonl with the 13-label Hermes vocabulary (or expanded label set documented in SI), session-grouped where possible.
2. Mint `stage0/make_splits.py curated`-style holdout for the new corpus (new split name, new `SPLIT_SEED` documented in SI).
3. Train Graph NCA (star + meta + 1d), HistGradientBoosting, and Random Forest with **identical** TF-IDF→SVD128 + struct features (`obs_pack.py` / `run_full_history_benchmarks.py` parity).

**Commands (after E1.1 staging script exists — placeholder paths):**

```bash
export NCA_DATA=$HOME/nca_data   # or repo-local data root per stage0/common.py
python scripts/stage_bfcl_next_tool_rows.py --out $NCA_DATA/datasets/bfcl_next_tool13
python stage0/make_splits.py bfcl_next_tool13   # to be implemented: mirror curated split logic

python stage0/run_full_history_benchmarks.py \
  --steps 150000 \
  --batch 512 \
  --seeds 0,1,2 \
  --topologies tree,meta,star,1d \
  --split bfcl_next_tool13 \
  --out stage0_results/bfcl_next_tool13_benchmarks.json

python scripts/record_experiment.py --id EXP-TBD-BFCL-CLS --json stage0/stage0_results/bfcl_next_tool13_benchmarks.json
```

**Holdout:** curated-style test partition only; bootstrap 10,000 resamples on test labels (same as `bootstrap_top1` in `run_full_history_benchmarks.py`).

**Runtime estimate (RTX 3060-class GPU):** `full_history_benchmarks.json` records ~46–51 s wall time for **10,000** steps per Graph NCA topology per seed on CUDA. Scaling linearly, **150,000** steps ≈ **12–13 min** × 4 topologies × 3 seeds ≈ **2.4–2.6 h** GPU. Tabular HGB+RF ≈ **15 min** CPU total. TF-IDF/SVD cache fit ≈ **15 min** CPU once per seed (see `stage0/input_parity_audit_20261026.md`).

---

## E2 — Pilot router ablation (τ²-bench mock + BFCL irrelevance)

**Goal:** Hold **FrogNano-4B actuator** and Dual-System harness fixed; swap only System-1 router: (a) random tool, (b) 1D lattice GNCA, (c) no-router (full schema to LLM), vs (d) production GNCA router.

**Prerequisite:** Check in reproducible pilot harness (`paper/JULIAN_TODO.md` item 7) with frozen configs and pinned actuator weights.

**Commands (harness TBD — interface contract):**

```bash
export ACTUATOR=FrogNano-4B
export ACTUATOR_REV=<pinned_hf_revision>
export HARNESS_CONFIG=configs/dual_system_pilot_v1.yaml

for ROUTER in random nca1d none gnca_meta; do
  for SEED in 0 1 2; do
    python scripts/run_dual_system_pilot.py \
      --config $HARNESS_CONFIG \
      --router $ROUTER \
      --seed $SEED \
      --tau2-mock-tasks 10 \
      --bfcl-irrelevance-probes 30 \
      --out results/pilot/router_ablation_${ROUTER}_seed${SEED}.json
  done
done

python scripts/aggregate_pilot_ablation.py \
  --glob 'results/pilot/router_ablation_*.json' \
  --out stage0/stage0_results/pilot_router_ablation_summary.json
```

**Holdout:** τ² mock task ids and BFCL probe ids held out from any router training (routers are frozen checkpoints; report per-task Pass¹ and irrelevance triage with Wilson CIs).

**Runtime estimate:** Prior exploratory session (`EXP-20261102-003`) covered 89 routed τ² turns + 30 BFCL probes on Apple M1 CPU. Expect **~30–90 min** per (router × seed) on CPU if actuator inference dominates; **~4–6 h** total for 4 routers × 3 seeds on a single RTX 3060-class GPU host if actuator runs locally on GPU. No-router baseline removes GNCA forward but retains actuator time (~25–40% faster per turn than dual route).

---

## E3 — Meta-learned adjacency audit (optional, small)

**Goal:** Log empirical directed diameter and edge sparsity of **trained** `GraphNCA_meta` checkpoints (seeds 0–2) after 150k steps; compare to fixed star/tree.

```bash
python stage0/export_meta_adjacency.py \
  --checkpoints $NCA_DATA/runs/EXP-20261102-001/meta_seed{0,1,2}.pt \
  --out stage0/stage0_results/meta_adjacency_audit.json
```

**Runtime estimate:** **< 5 min** CPU (no retraining).
