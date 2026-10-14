# Hypothesis tests: message passing ablations

**Date:** 2026-10-14

Equal-budget runs on curated Hermes train / curated val. Each variant trained for 150000 AdamW steps (batch 512, seed 0, lr 0.001, MPS when available).

## Variants

| Variant | What changes | Best val top-1 | Step | Local top-1 | Params |
| --- | --- | --- | --- | --- | --- |
| `baseline` | Current 1D rule; hidden reset every row | 0.6260 | 10000 | 0.2586 | 26749 |
| `carry` | Hidden carried across consecutive session decisions in training | 0.6277 | 5000 | 0.3903 | 26749 |

Always-terminal on curated val: **0.5430**.

## Verdicts

- **H2 (carry):** carry training beats baseline (0.6277 vs 0.6260). Val still resets hidden per row.

## H3 regeneration (baseline checkpoint)

Zeroed `cells_2_3` on curated val. Clean vs damaged top-1 by number of update steps:

| Steps | Clean | Damaged | Damaged − damaged@0 |
| --- | --- | --- | --- |
| 0 | 0.0017 | 0.0017 | +0.0000 |
| 1 | 0.4638 | 0.4054 | +0.4037 |
| 2 | 0.5498 | 0.5311 | +0.5294 |
| 3 | 0.6027 | 0.5665 | +0.5648 |
| 4 | 0.6260 | 0.5448 | +0.5431 |

Readout on zero hidden (k=0) is near chance, so Δ from k=0 is not the regeneration claim. From one update onward, damaged peaks at **k=3 (0.5665)**, which is +0.1611 vs damaged@1 (0.4054). At full depth, damaged is 0.5448 against clean 0.6260 (gap +0.0812).
Extra updates after the first help the damaged grid (partial support for H3); they do not fully close the gap to the clean grid.

## Scope

- Hypothesis 1 (image + instruction channel ablation) needs vision writes; not run.
- Carry is trained with truncated BPTT across a window of session decisions; the reported val metric still resets hidden so it stays comparable.
- Artifacts: `/Volumes/Extreme SSD/nca_research/results/hypothesis_tests`.

## Sources

- [research-design.md](research-design.md) hypotheses 2–4.
- [multi-harness-and-message-passing.md](multi-harness-and-message-passing.md) carry / halt / long-range recommendations.
