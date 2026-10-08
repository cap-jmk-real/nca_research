# NCA auto-research — final 3-day report

**Window:** 2026-10-05T13:11Z → 2026-10-08T13:11Z (deadline met)  
**Branch:** `cursor/auto-research-klaut-1754`  
**Sources:** SSD `results/auto_research/`, `hypothesis_tests/`, mix13/fineenvs metrics, `paper/si/experiments/`, daemon state.  
**Daemon stop:** STOP touched at report time; mid-flight job was `carry` @ step **5k**/150k (not near done).

## Daemon / queue

| Field | Value |
| --- | --- |
| Cycles / completed / failed | 29 / 28 / 1 |
| Rollovers | 3 |
| Failed job | `H31_compile_python_ts` exit **-9** (2026-10-06; later runs succeeded) |
| Aborted | `H6_20k` smoke (floor raised to ≥150k) |
| Last console | `carry` step 5000, val_top1 0.6277 |
| Heartbeat | 2026-10-08T13:05:19Z |

**Completed job ids (unique):** scoreboard×4, H6_long×4, H2b_carry_long×3, H_halt_long×3, scoreboard_2×3, H16_mix_prior×3, H31_compile_python_ts×2, H33_distill_python×3, eval_specialist_stubs×3.

## Best scores (SSD / SI)

| Metric | NCA | Flat | Always-terminal | Qwen2.5-1.5B |
| --- | --- | --- | --- | --- |
| Curated val top-1 (mix13 best @26k, EXP-20261003-001) | **0.6518** | 0.6071 | 0.5430 | — |
| Curated val (fineenvs best @60k, EXP-20261005-001) | 0.5841 | 0.4954 | 0.5430 | — |
| Local Hermes top-1 (fineenvs best) | 0.4033 | 0.4484 | **0.4947** | **0.0629** |
| Local Hermes (mix13 best) | 0.4009 | 0.4472 | 0.4947 | 0.0629 |

Gate scoreboard (`latest.json` 2026-10-08T15:05): curated **0.5841** (min 0.652), local Hermes **0.4033** (min 0.495). Mix13 best curated **0.6518** is ~0.0002 under the curated gate.

## Message-passing ablations (150k curated, `hypothesis_tests/summary.json`)

| Variant | Best val | @step | Local top-1 | Params |
| --- | --- | --- | --- | --- |
| long_range | **0.6355** | 10k | 0.2301 | 34238 |
| carry | 0.6277 | 5k | **0.3903** | 26749 |
| halt | 0.6267 | 10k | 0.2977 | 26766 |
| baseline | 0.6260 | 10k | 0.2586 | 26749 |
| flat | 0.6249 | 15k | 0.3737 | 4043 |

Cellular baseline beats flat on curated val; long_range best val; carry best local among cellular variants. Always-terminal still wins local Hermes.

## BFCL / NESTFUL (scored; head mismatch)

From `benchmarks_run.json` (mix13_best step 26k):

| Gate | Value | Min | Note |
| --- | --- | --- | --- |
| bfcl_ast_accuracy | 0.0 | 0.15 | 13-way Hermes name ≠ BFCL AST |
| bfcl_irrelevance_accuracy | 0.0542 | 0.50 | respond≈no-call rule |
| nestful_full_sequence_accuracy | 0.0 | 0.05 | single tool name ≠ nested sequence |

## Gates remaining

**Failing (5):** `curated_val_top1_nca`, `local_hermes_top1_nca`, `bfcl_ast_accuracy`, `bfcl_irrelevance_accuracy`, `nestful_full_sequence_accuracy`.  
**Blocked (3):** `tau2_airline_pass1`, `tau2_retail_pass1`, `appworld_dev_tgc` (install / episode runner).  
**Passing:** none.

## Engine / Klaut

29 probes: Klaut up **9**, Ollama up **12**, both down **17**. Live `--klaut-search` when up (latest query: next-tool imitation; 12 papers cached). Distill teacher path uses engine Ollama `qwen3.5:9b` when reachable.

## Distillation (≤9B) / specialists

| Action | Result |
| --- | --- |
| Compile (H31) | python **43684**, typescript **12**, orchestrator routes **43696** (`compile_stats.json`) |
| Distill python (H33) | **64** rows from `qwen3.5:9b`, `cap_b=9` |
| LiveCodeBench eval | **blocked** (harness not installed) |
| SI | EXP-20261005-003 compile; distill not a scored gate win |

## New checkpoints (mission window)

- Hypotheses: `baseline/carry/halt/long_range/flat_best.pt` under `hypothesis_tests/checkpoints/` (mtime through 2026-10-08).
- FineEnvs continuation: `text_nca_1d_fineenvs_{best,last}.pt` + step snaps (best tracked @60k).
- Primary still: `text_nca_1d_mix13_best.pt` (curated 0.6518).

## SI experiments touched

Primary: EXP-20261003-001 (mix13), EXP-20261004-001 (Qwen), EXP-20261005-001 (fineenvs), EXP-20261006-001…007 (150k ablations; some duplicate register). Smoke: EXP-20261005-002 (8k). Preliminary: EXP-20261005-003 (compile).

## Experiments completed vs failed

**Completed (representative):** multi-cycle 150k H6 long_range+flat, H2b carry, H_halt; H16 mix-prior / fineenvs train; H31 compile (after kill); H33 distill×3; specialist stub eval; scoreboards + SI register/build-tables; Klaut search when engine up.  
**Failed / aborted:** H31 once (SIGKILL); H6_20k aborted for 150k floor.

## Verdict

Mission **did not clear gates**. NCA **beats Qwen2.5-1.5B** on local Hermes (~0.40 vs 0.063) and **beats flat** on curated val (mix13 0.652≈gate; ablations ~0.63), but **loses to always-terminal** on local Hermes and remains near-zero on BFCL/NESTFUL with the current head. Architecture deltas (long_range / carry) are small vs the prior-mismatch / head-format gap.

## Recommended next week

1. **Prior match (H16):** train sampler → curated ~52% terminal; re-score local Hermes vs always-terminal before more 150k architecture loops.
2. **Promote mix13 best into scoreboard** (or re-eval gates on that ckpt) — curated gate is essentially hit at 0.6518.
3. **BFCL/NESTFUL:** separate arg/AST or sequence head; stop scoring Hermes-name logits as full AST.
4. **Specialists:** install LiveCodeBench; scale distill beyond 64 rows on engine; TS needs non-SWE sources.
5. **Unblock τ2/AppWorld** only after local Hermes beats always-terminal.
6. Keep distill ≤9B, `num_predict≤24`; no new long trains until (1)–(2) land.
