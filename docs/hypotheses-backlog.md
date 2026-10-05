# Hypotheses backlog (auto-research)

**Date:** 2026-10-05  
**Priority:** 1 = run next when the auto-research loop fires. Tie-break by estimated wall time (shorter first).

Each item links to a **gate** in `config/benchmark_gates.json` or a **research-design** gap. Klaut queries for failed gates are in `scripts/auto_research.py` (`GAP_TO_QUERY`).

---

## Closed-set Hermes (current 13-label head)

### H2b — Session carry at **eval** time, not just training

- **Claim:** Carrying hidden channels across consecutive rows in the same `session_id` at inference beats resetting hidden on local Hermes, without hurting curated val.
- **Gate:** `local_hermes_top1_nca`
- **Priority:** 1
- **Klaut query:** `recurrent state tool agent session memory`
- **Cost:** Small code change in eval + short train continuation from mix13 best.

### H16 — Train sampler matches **curated val prior** (52% terminal), not coding-mix 37% respond

- **Claim:** Oversampling `respond` from OpenHands/SWE-smith hurts local terminal-heavy sessions; Hermes-prior batching fixes local without dropping coding rows entirely.
- **Gate:** `local_hermes_top1_nca`
- **Priority:** 2
- **Klaut query:** `class imbalance imitation learning tool selection`
- **Cost:** Already partially wired (`CURATED_TRAIN_COUNTS`); ablation vs uniform coding mix.

### H17 — Two-timescale hidden (fast per-row + slow session buffer)

- **Claim:** A fixed-size session buffer updated with a gated write each decision beats single carry (H2) and avoids oversmoothing the 5-cell grid.
- **Gate:** `local_hermes_top1_nca`
- **Priority:** 3
- **Klaut query:** `dual memory fast slow neural network agent`
- **Cost:** +1 small GRU or linear state per session; M1-feasible.

### H5 — Goal-only readout cell vs flatten-all-cells

- **Claim:** Reading out from cell 0 (goal) after updates beats flattening all hidden cells (reduces oversquashing from far steps).
- **Gate:** `curated_val_top1_nca`
- **Priority:** 4
- **Klaut query:** `graph neural network readout node classification`
- **Cost:** One-line head change + retrain 8k–50k steps.

### H6 — Long-range goal edge **in production training** (not 8k ablation only)

- **Claim:** The +0.2pt curated val from [hypothesis-tests.md](hypothesis-tests.md) `long_range` holds at 150k with mix13 + FineEnvs prior.
- **Gate:** `curated_val_top1_nca`
- **Priority:** 5
- **Klaut query:** `long range skip connections sequence modeling`
- **Cost:** Full train resume from best checkpoint.

### H20 — Ensemble NCA logits + flat logits (same obs)

- **Claim:** Convex combination or stacked logits beats both alone on local and val (cheap hybrid without attention).
- **Gate:** `local_hermes_top1_nca`, `curated_val_top1_nca`
- **Priority:** 6
- **Klaut query:** `ensemble linear classifier neural cellular automata`
- **Cost:** Inference-only calibration on val, no retrain.

### H_long_range_prod — Long-range + Hermes prior + no FineEnvs teacher traces

- **Claim:** Rewiring helps even when FineEnvs jsonl is absent (avoids respond dilution from 27B teacher SFT).
- **Gate:** `curated_val_top1_nca`
- **Priority:** 7
- **Klaut query:** `multi harness reinforcement learning tool overfitting`
- **Cost:** Train with `fineenvs_common_train.jsonl` withheld.

### H18 — Damage during training (regeneration objective)

- **Claim:** Random zeroing of 1–2 trace cells during train improves damaged-grid accuracy (H3) at full depth.
- **Gate:** (research) H3 metric in `nca_text/hypotheses.py`
- **Priority:** 8
- **Klaut query:** `growing neural cellular automata regeneration damage`
- **Cost:** Data aug in train loop.

### H9 — Asynchronous message passing (MAPN-style)

- **Claim:** Updating cells in random order (or goal-first schedule) beats synchronous 4-step on local sessions.
- **Gate:** `local_hermes_top1_nca`
- **Priority:** 9
- **Klaut query:** `asynchronous graph neural network message passing`
- **Cost:** New update schedule in `TextNCA.evolve`.

### H10 — Selective scan in perception only (MbaGCN analogue)

- **Claim:** Replace fixed Laplacian stack with a 1D selective state on the 5 cells while keeping growing-NCA residual update.
- **Gate:** `curated_val_top1_nca`
- **Priority:** 10
- **Klaut query:** `Mamba graph neural network message passing`
- **Cost:** Moderate; keep param count < 100k.

---

## Distillation (teacher ≤ 9B — **labels and text only**, not NCA weights)

### H7 — 9B pseudo-labels only on low-confidence NCA rows

- **Claim:** `qwen3.5:9b` (cap) relabels train rows where NCA entropy > τ; improves local rare tools without 27B-style respond flood.
- **Gate:** `local_hermes_top1_nca`
- **Priority:** 2
- **Klaut query:** `confidence based pseudo labeling small model distillation`
- **Cost:** Engine batch inference; write new jsonl slice on SSD.

### H14 — Verifier-correct FineEnvs rollouts only (already staged), **no** 27B completion dump

- **Claim:** Multi-harness **verified** tool rows at Hermes prior beat adding raw teacher SFT (FineEnvs guide: 27B SFT barely moved avg).
- **Gate:** `curated_val_top1_nca`, `fineenvs_val_top1_*`
- **Priority:** 3
- **Klaut query:** `multi harness RL tool calling distillation`
- **Cost:** Continue `train_fineenvs` run; compare checkpoints.

### H28 — 9B shortens tool **observation text** only (not target label)

- **Claim:** Hermes-first-run step 4: engine compresses tool results into schema; NCA trains on cleaned obs, same labels.
- **Gate:** `local_hermes_top1_nca`
- **Priority:** 11
- **Klaut query:** `context compression tool result summarization LLM`
- **Cost:** Offline preprocessing pass on jsonl.

---

## Benchmark unlock (needs new heads or harness wiring)

### H11 — Explicit **abstain** logit for BFCL irrelevance

- **Claim:** Separate `respond`/abstain threshold tuned on val beats mapping `respond` alone to irrelevance.
- **Gate:** `bfcl_irrelevance_accuracy`
- **Priority:** 4
- **Klaut query:** `function calling abstention irrelevance benchmark`
- **Cost:** Train with irrelevance rows from BFCL jsonl as extra class or threshold.

### H12 — Discrete **argument bucket** head (BFCL AST partial)

- **Claim:** Emit (function, arg_bucket)* where buckets come from gold vocabulary per function; beats name-only for zero-arg BFCL.
- **Gate:** `bfcl_ast_accuracy`
- **Priority:** 5
- **Klaut query:** `tool call argument prediction compact classification`
- **Cost:** New label space + staging script from BFCL possible answers.

### H21 — BFCL function-name alias table (Hermes → BFCL)

- **Claim:** Many BFCL failures are vocabulary mismatch, not routing; learnable alias layer beats raw 13 names.
- **Gate:** `bfcl_ast_accuracy`
- **Priority:** 6
- **Klaut query:** `tool name normalization API schema mapping`
- **Cost:** Script + eval only first.

### H22 — Irrelevance from **entropy gate** on NCA logits

- **Claim:** High entropy → abstain beats fixed `respond` class for BFCL irrelevance slice.
- **Gate:** `bfcl_irrelevance_accuracy`
- **Priority:** 7
- **Klaut query:** `selective prediction tool use abstention`
- **Cost:** No retrain; calibrate on val.

### H23 — Length-2 call queue head for NESTFUL

- **Claim:** Emit two tool names autoregressively from same grid state; improves full-sequence metric vs single shot.
- **Gate:** `nestful_full_sequence_accuracy`
- **Priority:** 8
- **Klaut query:** `nested API call sequence generation`
- **Cost:** New head; train on NESTFUL-aligned rows if staged.

### H24 — NESTFUL gold as **secondary train slice** (not eval leak)

- **Claim:** Train on NESTFUL train split mapped to Hermes-like labels; eval on held-out NESTFUL only.
- **Gate:** `nestful_full_sequence_accuracy`
- **Priority:** 12
- **Klaut query:** `NESTFUL nested function calling dataset`
- **Cost:** Staging + train.

### H25 — τ2: NCA as **tool router**, 9B as language user-facing turn

- **Claim:** Hybrid stays under distillation cap: cellular policy picks tool id; local LLM fills natural language when τ2 requires reply.
- **Gate:** `tau2_airline_pass1`, `tau2_retail_pass1`
- **Priority:** 13
- **Klaut query:** `tau bench modular agent tool router`
- **Cost:** Install τ2, wire LiteLLM to Ollama on engine.

### H26 — AppWorld: map APIs to 13 Hermes tools + `execute_code` body from template

- **Claim:** Until full program head exists, template programs from tool choice beat random for dev tgc>0.
- **Gate:** `appworld_dev_tgc`
- **Priority:** 14
- **Klaut query:** `AppWorld benchmark program synthesis API`
- **Cost:** Install AppWorld bundles on SSD.

### H27 — **Program latent** channel (8-d) written by NCA, filled by 9B once per step

- **Claim:** Under 9B cap, teacher fills a short code latent; NCA decides when to invoke — unlocks AppWorld without full LM in the rule.
- **Gate:** `appworld_dev_tgc`
- **Priority:** 15
- **Klaut query:** `latent planning code generation tool agent`
- **Cost:** Large; multimodal ladder step 2.5.

### H13 — Harness embedding (OpenCode / Codex / Mini-SWE) in obs channels

- **Claim:** KwaiKAT-style format overfitting drops when harness id is written into cell 0 side channel.
- **Gate:** `tau2_*`, FineEnvs holdout
- **Priority:** 10
- **Klaut query:** `multi harness tool calling overfitting`
- **Cost:** Extend jsonl schema + `build_trace_obs`.

---

## Multimodal (research-design H1 — not in benchmark gates yet)

### H1 — Vision + instruction write slices (research-design step 1)

- **Claim:** Grid with both slices beats either ablation on a digit+instruction task before Hermes screenshots.
- **Gate:** (future) `minigrid_dual_modal`
- **Priority:** 20
- **Klaut query:** `multimodal neural cellular automata vision language`
- **Cost:** New env; not Hermes benchmark suite.

---

## Specialist coder stack (Python → TS → Shell → Orchestrator)

See [specialist-coder-stack.md](specialist-coder-stack.md) and `config/specialists.json`. Distill ≤9B; evaluate with open harnesses; register `EXP-*`.

### H31 — Compile Python specialist corpus from staged coding dumps

- **Claim:** Language-tagged rows (≥0.45 score) from SWE-smith / OpenHands / R2E are enough to start Python SFT.
- **Gate:** `specialist_python_dataset_rows` (informal ≥10k)
- **Priority:** 1
- **Command:** `python scripts/compile_specialist_datasets.py --langs python --max-rows-per-source 50000`

### H32 — Compile TypeScript specialist corpus

- **Claim:** Same pipeline yields a usable TS/JS set for `tsc_repo_smoke`.
- **Priority:** 2
- **Command:** `python scripts/compile_specialist_datasets.py --langs typescript`

### H33 — Distill Python student from `qwen3.5:9b` on compiled rows

- **Claim:** 9B teacher plans + tool tags improve Python specialist over base small model on LiveCodeBench slice.
- **Priority:** 3
- **Command:** `python scripts/distill_specialist.py --lang python --limit 256` (engine up)

### H34 — Distill TypeScript student (same teacher)

- **Priority:** 4

### H35 — Shell specialist from terminal-heavy traces + Terminal-Bench subset

- **Priority:** 5

### H36 — Orchestrator routes to {python, typescript, shell} then plugs specialists

- **Claim:** Routed system beats single 9B on multi-lang smoke and does not regress Python-only LiveCodeBench.
- **Priority:** 6
- **Eval:** `multi_specialist_smoke`, then SWE-Lite mixed slice

### H37 — TextNCA next-tool prior inside orchestrator (optional)

- **Claim:** Hermes-13 prior reduces tool thrash without specialist weight distillation into the NCA.
- **Priority:** 8

## Auto-research meta

### H29 — Klaut report before every **new architecture family**

- **Claim:** One `POST /v1/research/reports` per family (SSM-in-NCA, hybrid readout, etc.) reduces wasted 150k runs.
- **Priority:** 1 (process)
- **Klaut query:** (topic = failing gate string)
- **Cost:** ~5–15 min engine time; cache on SSD.

### H30 — Scoreboard gate drives experiment queue order

- **Claim:** Running `scripts/auto_research.py` daily and only training top `suggested_hypotheses` beats ad-hoc ablations.
- **Priority:** 1 (process)
- **Cost:** Zero; use Cursor skill `nca-auto-research`.
