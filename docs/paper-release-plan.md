# Paper release plan (execution)

Ordered plan so HF distribution, quants, and the manuscript stay aligned.

## Phase 0 — Ops (you at the computer)

1. Move worker/repo out of `~/Documents` (TCC) → e.g. `~/code/nca_research`
2. Full Disk Access for Cursor + `cursor-agent`; prevent sleep during long runs
3. Install Cursor GitHub App on private `nca_research`
4. Create HF token with write to `cap-jmk-real` (or org `nca-research`)
5. Merge this patch set onto `cursor/auto-research-klaut-1754` if applied from `~/code`

## Phase 1 — Freeze claims

1. Lock gates: `config/benchmark_gates.json` + `config/hf_publish.json`
2. Finish H6 150k (`long_range` / `baseline` / `flat`) → SI EXP
3. No architectural claim without ≥150k + flat + always-terminal

## Phase 2 — Datasets

1. Finish specialist compile (python/ts); scrub via `prepare_hf_dataset.py`
2. Mint dataset EXP; promote slot **D** only if scrub_ok and row minimums met
3. Dry-run `publish_hf.py dataset`

## Phase 3 — Models

1. Export best TextNCA + flat checkpoints (safetensors + config)
2. Distill/eval python specialist when engine Ollama is up (≤24 tokens)
3. `select_top4_artifacts.py --write` → slots A/B/C

## Phase 4 — Quantization tests

1. Run ladder on A and C: fp16, fp8_e4m3, int8, nf4, iq2_xxs (~2.5), iq2_s
2. Same eval protocol as fp16; record `quant_*_latest.json`
3. Keep ≤1 quant companion per slot (98% rule / Pareto)

## Phase 5 — Publish four only

1. `publish_hf.py all --from-top4 --dry-run` → review cards
2. `--push` only for the four winners
3. Update `paper/availability.md` with URLs + EXP ids

## Phase 6 — Manuscript

1. Methods: architecture + token budgets + quant ladder pointer
2. Results: SI tables only
3. Efficiency section: tokens + quant Pareto vs small transformers
4. Availability section: four HF links
5. Limitations: harness blockers, engine downtime, no teacher re-host

## Done when

- [ ] Four HF repos on `main` match `results/hf/top4.json`
- [ ] Quant table in SI for A and C
- [ ] Every paper number has an EXP id
- [ ] Efficiency claims do not rely on unoptimized multi-step chatter
