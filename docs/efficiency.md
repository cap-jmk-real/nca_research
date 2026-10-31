# Token / compute efficiency

**Date:** 2026-10-05

NCA-style local rules and multi-step specialists can **spend more compute (and, when LMs are involved, more tokens) than a small transformer** for equal or worse accuracy if misused. Rules below keep the stack lean.

## Agent monitoring (Cursor)

- Supervise every **12h**, not 6h.
- First command: `python scripts/status_lean.py` (or read `daemon/status_lean.txt`).
- Reply with that status block only — no re-summaries of known docs, no full log dumps.
- Do not `cat` `console.log` / `daemon.log` unless status_lean shows a stall or error.

## Algorithms

| Component | Cheap use | Expensive anti-pattern |
| --- | --- | --- |
| TextNCA | 13-way next-tool logits, K≤4, L=5 | Generating text; deep K; frequent full-val |
| Orchestrator LM | `ROUTE:<lang>` only (≤24 tokens out) | Long plans every step |
| Specialists | `TOOL:<name>` / patch tags; tight `num_predict` | Unbounded chat completions |
| Distill teacher ≤9B | Short prompts, temp 0, **24**-token cap, goal ≤160 chars | 512-token “explain” dumps |
| LLM baseline | Same 160-char obs cap as NCA `TEXT_CAP` | Full tool dumps into the prompt |
| Dataset compile | Field slices, skip `._*`, goal ≤240, paths ≤8 | `json.dumps(entire parquet row)` |

## Hard budgets

| Knob | Default |
| --- | --- |
| Distill `num_predict` | **24** (`NCA_DISTILL_NUM_PREDICT`) |
| Distill goal chars | **160** (matches `nca_text.data.TEXT_CAP`) |
| Distill path hints | **4** |
| Distill row limit (daemon) | **64** |
| LLM baseline `max_new_tokens` | **8** |
| LLM baseline obs truncate | **160** chars / step |
| Hypotheses `VAL_EVERY` | **5000** on ≥150k runs |
| Supervise interval | **12h** |

## Defaults

- Plug order still: orchestrator → one specialist → tools → verifier (no multi-specialist chatter).
- TextNCA stays a classifier — never a generator competing with transformers on open text.

## Prefer the cheaper model

If a cellular / specialist system loses to a flat or small transformer baseline on the same gate **and** uses more tokens or wall-clock, mark the EXP `claims_not_supported` and prefer the cheaper model until an optimization lands.

## Release

HF top-4 + quant ladder: [huggingface-distribution.md](huggingface-distribution.md), [paper-release-plan.md](paper-release-plan.md).

Worker ops (TCC/SSD/DNS): [mac-worker-prep.md](mac-worker-prep.md).
