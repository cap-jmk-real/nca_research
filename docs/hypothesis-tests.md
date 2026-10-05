# Hypothesis tests: message passing ablations

**Date:** 2026-10-05

Equal-budget runs on curated Hermes train / curated val. Each variant trained for 8000 AdamW steps (batch 512, seed 0, lr 0.001, MPS when available).

## Variants

| Variant | What changes | Best val top-1 | Step | Local top-1 | Params |
| --- | --- | --- | --- | --- | --- |
| `baseline` | Current 1D rule; hidden reset every row | 0.6377 | 5500 | 0.2883 | 26749 |
| `halt` | Learned soft halt on update depth (AMP analogue) | 0.6329 | 6500 | 0.3369 | 26766 |
| `long_range` | Goal-cell long-range edge into every position | 0.6400 | 6500 | 0.1815 | 34238 |
| `carry` | Hidden carried across consecutive session decisions in training | 0.6203 | 5000 | 0.3760 | 26749 |
| `flat` | Hypothesis 4: linear readout, no cellular update | 0.6296 | 4000 | 0.4294 | 4043 |

Always-terminal on curated val: **0.5430**.

## Verdicts

- **H4 (flat baseline):** cellular baseline beats flat (0.6377 vs 0.6296).
- **H_halt:** learned halt loses to baseline (0.6329 vs 0.6377).
- **H_long_range:** goal edge beats baseline (0.6400 vs 0.6377).
- **H2 (carry):** carry training loses to baseline (0.6203 vs 0.6377). Val still resets hidden per row.

## H3 regeneration (baseline checkpoint)

Zeroed `cells_2_3` on curated val. Clean vs damaged top-1 by number of update steps:

| Steps | Clean | Damaged | Damaged − damaged@0 |
| --- | --- | --- | --- |
| 0 | 0.0017 | 0.0017 | +0.0000 |
| 1 | 0.5257 | 0.4865 | +0.4848 |
| 2 | 0.5784 | 0.5783 | +0.5766 |
| 3 | 0.6150 | 0.5976 | +0.5959 |
| 4 | 0.6377 | 0.5783 | +0.5766 |

Readout on zero hidden (k=0) is near chance, so Δ from k=0 is not the regeneration claim. From one update onward, damaged peaks at **k=3 (0.5976)**, which is +0.1111 vs damaged@1 (0.4865). At full depth, damaged is 0.5783 against clean 0.6377 (gap +0.0595).
Extra updates after the first help the damaged grid (partial support for H3); they do not fully close the gap to the clean grid.

## Scope

- Hypothesis 1 (image + instruction channel ablation) needs vision writes; not run.
- Carry is trained with truncated BPTT across a window of session decisions; the reported val metric still resets hidden so it stays comparable.
- Artifacts: `/Volumes/Extreme SSD/nca_research/results/hypothesis_tests`.

## Sources

- [research-design.md](research-design.md) hypotheses 2–4.
- [multi-harness-and-message-passing.md](multi-harness-and-message-passing.md) carry / halt / long-range recommendations.
