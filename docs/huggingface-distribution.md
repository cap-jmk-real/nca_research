# Hugging Face distribution + quantization + paper release

**Date:** 2026-10-05  
**Status:** plan + scaffolds (publish gated on gates + top-4 selection)  
**Org / namespace (proposed):** `nca-research` (or `cap-jmk-real` until org exists)

This document is the release contract for datasets, models, floating-point
quants (including ~2.5-bit), and the paper/SI. **Only the four best artifacts
stay public on Hugging Face at any time.** Older HF revisions are archived
locally on the SSD and referenced in SI, not left as live “latest”.

---

## 1. What we publish (four slots)

| Slot | Artifact | HF repo id (pattern) | Wins by |
| --- | --- | --- | --- |
| **A** | Best TextNCA next-tool checkpoint | `…/textnca-next-tool` | curated val top-1 − always-terminal; tie-break vs Qwen2.5-1.5B gate |
| **B** | Best flat / transformer baseline we beat or match | `…/baseline-next-tool` | same gate, for honest comparison |
| **C** | Best specialist student (usually python) | `…/specialist-python` | LiveCodeBench / smoke harness + distill fidelity |
| **D** | Best shared dataset release | `…/hermes-next-tool-mix` (dataset) | coverage + license-clean + reproducible split |

If a new run does not beat the occupant of a slot on the **frozen gate set**
(`config/benchmark_gates.json` + `config/hf_publish.json`), it is **not**
uploaded as `main` / `latest`. It may be stored as an SSD EXP record only.

Rule: **publish only the four best always** — one live revision per slot.
Use HF revision tags `exp-<EXP_ID>` for provenance; move `main` only when
`scripts/select_top4_artifacts.py` promotes a candidate.

---

## 2. Datasets on Hugging Face

### 2.1 Release units

| Dataset | Contents | Privacy |
| --- | --- | --- |
| `hermes-next-tool-mix` | Public-safe next-tool rows (goal ≤160 chars, tool names, labels); no raw secrets | scrubbed |
| `specialist-python-v1` | Route/tool tags + path hints (compiled fields only) | scrubbed |
| `specialist-typescript-v1` | Same for TS (may be small) | scrubbed |
| `specialist-shell-v1` / `orchestrator-routes-v1` | When row counts clear minimums | scrubbed |

Do **not** upload full OpenHands / SWE-smith parquet dumps. Only the compiled
jsonl slices under `datasets/specialists/` after scrub.

### 2.2 Card requirements

Every dataset card must include: license, source attribution, `TEXT_CAP=160`,
train/val split rule (`public_split`), row counts, and the EXP id that minted
the release.

### 2.3 Commands

```bash
python scripts/prepare_hf_dataset.py --lang python --dry-run
python scripts/publish_hf.py dataset --slot D --exp EXP-… --dry-run
# real push when HF_TOKEN set and top-4 promotes slot D
python scripts/publish_hf.py dataset --slot D --exp EXP-… --push
```

---

## 3. Models on Hugging Face

### 3.1 TextNCA / flat

Export format:

- `model.safetensors` (preferred) or `model.pt`
- `config.json` (obs_c, hidden_c, steps K, seq_len, vocab, halt/long_range flags)
- `README.md` model card with gate table + EXP id
- optional `val_metrics.json`

### 3.2 Specialist students (≤9B teacher → small student)

When LoRA/SFT students exist: publish adapter or merged weights for the **slot C**
winner only. Teachers (`qwen3.5:9b`) are **not** re-hosted unless license allows
and we need a frozen eval mirror — prefer pointing at the upstream id.

---

## 4. Quantization plan (incl. floating-point ~2.5)

Goal: measure **accuracy vs size vs tokens/latency** so the paper can claim
efficiency honestly (see also `docs/efficiency.md`). If a quant loses the gate
**and** is not smaller/faster enough, do not publish it in the top-4.

### 4.1 Ladder (test all; publish at most one quant companion per slot)

| Level | Approx bits | Method (priority) | Applies to |
| --- | --- | --- | --- |
| fp32 / fp16 / bf16 | 32 / 16 | native train export | TextNCA, specialists |
| fp8 | 8 | E4M3 weight-only (torchao / transformer-engine / MLX where available) | specialists, large baselines |
| int8 | 8 | dynamic / static PTQ | TextNCA + specialists |
| int4 / nf4 | 4 | bitsandbytes NF4 / AWQ / GPTQ | specialists |
| **~2.5 bpw** | ~2–3 | llama.cpp **IQ2_XXS / IQ2_S** (or MLX equivalent); optional QuIP# if deps OK | specialists only |
| extreme | ≤2 | research-only; SI appendix, not a live HF slot unless it wins slot C | specialists |

TextNCA is already tiny (classifier, K≤4). For the paper, still report **int8**
TextNCA vs fp32 to show cellular readout is not the memory bottleneck —
transformers/specialists are.

### 4.2 Protocol (`scripts/quantize_checkpoint.py`)

For each candidate checkpoint:

1. Load fp16/bf16 reference.
2. Build each quant in the ladder (skip if backend missing; record `blocked`).
3. Run the **same** closed-set next-tool or specialist smoke eval (obs cap 160,
   `num_predict` ≤ 24 for generative students).
4. Write `quant_report.json` next to the EXP on the SSD.
5. Promotion: a quant may replace the slot’s published companion file set only
   if it keeps ≥ **98%** of fp16 top-1 **or** is the Pareto winner on
   (size, latency, top-1) among quants for that slot.

### 4.3 What “2.5” means here

We treat **~2.5-bit floating / residual quants** as the IQ2 / mxfp-style band
(not naive int2). Paper wording: “sub-3-bit weight-only quantization (≈2.5 bpw
IQ2-class)”. Exact codec recorded in SI per EXP.

---

## 5. Top-4 selection algorithm

Implemented in `scripts/select_top4_artifacts.py` using `config/hf_publish.json`:

1. Load all SI EXP records with `status` in `{accepted, preliminary}` and
   required metrics present.
2. Score TextNCA / flat candidates:  
   `score = val_top1 - always_terminal_top1` (primary), then `val_top1 - qwen15_top1`.
3. Score specialists: harness score if present, else distill tag-match rate.
4. Score datasets: `n_rows` × license_ok × scrub_ok (boolean gates).
5. Emit `results/hf/top4.json` and exit nonzero if a `--push` was requested
   without promotion.

Daemon / human release flow:

```bash
python scripts/select_top4_artifacts.py --write
python scripts/quantize_checkpoint.py --from-top4 --ladder default
python scripts/publish_hf.py all --from-top4 --dry-run
# after review:
python scripts/publish_hf.py all --from-top4 --push
```

---

## 6. Paper plan (structure + what HF backs)

### 6.1 Narrative arc

1. **Problem:** next-tool / coding-control policies burn tokens; small transformers
   are strong baselines.
2. **Method:** 1D TextNCA + optional long-range / halt / carry; specialist plug
   stack; distillation ≤9B with tight token budgets.
3. **Evidence:** EXP table (SI); ≥150k long-train floor for architectural claims.
4. **Efficiency:** token budgets + **quantization Pareto** (fp16 → fp8 → int4 → ≈2.5).
5. **Release:** four HF artifacts + SI reproducibility bundle.

### 6.2 Paper sections ↔ artifacts

| Section | Needs |
| --- | --- |
| Methods | architecture, TEXT_CAP, VAL_EVERY, distill caps |
| Experiments | EXP-* from `paper/si/` |
| Efficiency | `docs/efficiency.md` + quant reports |
| Availability | HF top-4 links + commit hash + EXP ids |
| Limitations | engine downtime, harness blockers (τ2/AppWorld), TCC/worker ops out of scope |

### 6.3 SI checklist before submission

- [ ] All claimed numbers have an EXP json
- [ ] Flat + always-terminal + Qwen1.5B present for next-tool claims
- [ ] Long-train (≥150k) for architectural claims
- [ ] Quant table for slot A and slot C winners
- [ ] HF `main` matches top-4 file (`results/hf/top4.json`)
- [ ] Scrub log for datasets (no secrets)

### 6.4 Writing order (execution)

1. Freeze gates in `config/benchmark_gates.json` + `config/hf_publish.json`
2. Finish H6 150k + SI register
3. Compile/scrub specialist datasets → prepare HF dataset packs
4. Distill/eval when engine up → specialist candidate
5. Quant ladder on top candidates
6. `select_top4` → dry-run publish → real publish
7. Fill `paper/availability.md` + methods efficiency subsection
8. Draft results from SI tables only (no orphan numbers)

---

## 7. Secrets & ops

- `HF_TOKEN` in environment / cloud secrets (write access to namespace)
- Never commit tokens
- SSD mirror: `/Volumes/Extreme SSD/nca_research/results/hf/`
- Local Mac worker: keep repo **outside** `~/Documents` (TCC) so release
  scripts can read checkpoints

---

## 8. Non-goals

- Hosting full 9B teachers on HF
- Publishing every ablation checkpoint
- Claiming quant parity without the same eval protocol
- Multi-specialist chatter stacks that blow token budgets
