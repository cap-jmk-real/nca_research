# Julian TODO — items requiring new runs, exports, or release work

Paper edits in this branch only change text/figures where existing repo artifacts already support the claim. Complete the items below before resubmission.

## Blockers (review panel)

1. **Public GitHub URL** — Manuscript keeps `https://github.com/cap-jmk-real/nca_research`; re-check HTTP 200 immediately before each submission (returned 200 in the cloud-agent environment on 2026-11-02).
2. **LICENSE** — Add a root `LICENSE` (e.g. MIT or Apache-2.0) and matching Hugging Face dataset/model license tags; update the availability paragraph once chosen.
3. **Same-hardware latency control** — Run the LLM routing baseline on the same Apple M1 host as GNCA (local quantized model or on-device API) and report paired per-turn latency; until then keep Table live-benchmarks free of speedup ratios.

## Pilot diagnostics (raw counts & statistics)

4. **BFCL v4 irrelevance** — Export the $N=30$ probe ledger to `paper/si/` with per-probe labels; report $k/30$ next to $90.0\%$ and Wilson 95\% CI in `main.tex`.
5. **BFCL tool activation** — Record denominator and $k/N$ for both v4 table cells and the `BFCL_v3_simple` functional subset; add Wilson CI if $N \le 100$.
6. **False triggers / context overflow** — Instrument baseline vs Dual GNCA for wrong-tool selections and context-limit/truncation events, or remove distraction/exhaustion language entirely.
7. **Pilot reproduction package** — Check in scripts + frozen configs for $\tau^2$-bench mock ($N=10$), BFCL probes, and SWE episode; pin actuator checkpoint (FrogNano-4B) revision.

## Primary benchmark & methods

8. **Session manifest** — Publish `summary.json` from `stage0/make_splits.py` (per-split session counts) alongside row counts already cited from `full_history_benchmarks.json`.
9. **Compute footprint** — Log GPU/CPU wall-clock and estimated kWh for 150k GNCA runs (M1 MPS) and engine RTX 3060 tabular fits; add a short table in Methods or SI.
10. **Per-seed primary table** — Optional: append mean $\pm$ SD across seeds 0–2 for each Table 1 row (bootstrap CIs already in Table feature-parity).

## Formal proofs & artifacts

11. **Lean / mathlib pin** — If mathlib is added later, record `lake-manifest.json` revision and CI badge in SI.
12. **Zenodo / DOI** — Archive code+weights+dataset snapshot with revision hash for long-term citation.

## Already addressed in manuscript (no new numbers)

- Curated split row counts ($57{,}675 / 3{,}465 / 3{,}530$) from `stage0/stage0_results/full_history_benchmarks.json`.
- Token reduction arithmetic ($209{,}734 \to 117{,}462$; $44.0\%$).
- Wilson CIs for $8/10$, $7/7$, and $14/14$ pilot fractions (computed from stated numerators).
- Pareto figure overlays primary 150k curated-test points vs local Hermes OOD ablations.
- Compute-parity disclaimer for zero-shot Qwen vs trained GNCA/Transformer.
