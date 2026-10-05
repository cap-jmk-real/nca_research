# Token / compute efficiency

**Date:** 2026-10-05

NCA-style local rules and multi-step specialists can **spend more compute (and, when LMs are involved, more tokens) than a small transformer** for equal or worse accuracy if misused. Rules below keep the stack lean.

## Agent monitoring (Cursor)

- Supervise every **12h**, not 6h; prompts ≤ ~10 lines.
- Supervision replies: status table only (no re-summaries of known docs).
- Daemon logs milestones to SSD; does not spam stdout.

## Algorithms

| Component | Cheap use | Expensive anti-pattern |
| --- | --- | --- |
| TextNCA | 13-way next-tool logits, K≤4, L=5 | Generating text; deep K; frequent full-val |
| Orchestrator LM | `ROUTE:<lang>` only (≤48 tokens) | Long plans every step |
| Specialists | Patch/tool tags; tight `num_predict` | Unbounded chat completions |
| Distill teacher ≤9B | Short prompts, temp 0, 48-token cap | 512-token “explain” dumps |
| Dataset compile | Field slices, skip `._*`, no full-row dumps | `json.dumps(entire parquet row)` |

## Defaults

- Hypotheses val every **5000** steps on ≥150k runs.
- Distill: `num_predict=48`, goal truncated to 240 chars.
- Plug order still: orchestrator → one specialist → tools → verifier (no multi-specialist chatter).

If a cellular / specialist system loses to a flat or small transformer baseline on the same gate **and** uses more tokens, mark the EXP `claims_not_supported` and prefer the cheaper model until an optimization lands.
