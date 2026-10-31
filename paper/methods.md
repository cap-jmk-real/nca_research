# Methods (draft)

Numbers that are already measured cite experiment ids. Unmeasured procedures are written as protocol.

## Task

At each assistant decision in a Hermes-style session, predict the next tool name from a fixed vocabulary of 13 labels (including `respond`). Session-level splits: no `session_id` in both train and curated val. Local `~/.hermes` sessions are a held-out test and are never written to disk as transcripts ([EXP registry](si/tables/S1_experiment_registry.md)).

## Observation

Sequence length 5: cell 0 = latest user goal; cells 1–4 = up to four recent (tool, truncated result) steps, right-aligned. Each cell: 48-d hashed bag-of-tokens, one-hot of prior tool if in vocab, occupied bit. Observation channels are not updated by the rule ([docs/text-nca-architecture.md](../docs/text-nca-architecture.md)).

## Model

**TextNCA.** Hidden channels \(H=16\). Perception: fixed depthwise 1D identity / gradient / Laplacian. Shared 1×1 update produces residual + sigmoid gate (Growing-NCA style). \(K=4\) iterations with per-step embedding. Readout: linear map on flattened hidden state.

**Variants (ablations).**

| Name | Change |
| --- | --- |
| `baseline` | Reset hidden every decision row |
| `halt` | Soft learned halt on remaining depth |
| `long_range` | Goal-cell broadcast into perception (zero-init scale) |
| `carry` | Hidden carried across consecutive session decisions in training (truncated BPTT) |
| `flat` | Linear map on flattened observation (hypothesis 4) |

## Training

- Optimizer: AdamW, lr \(10^{-3}\) cosine to 1%, weight decay \(10^{-2}\), grad clip 1.0.
- Loss: class-weighted CE with \(\sqrt{\text{inverse frequency}}\) weights.
- **Architectural comparisons:** ≥ **150 000** steps (long-training floor; see [docs/auto-research-loop.md](../docs/auto-research-loop.md)).
- Mix13 primary: batch 1024, MPS, seed 0 — see primary EXP for mix13.
- Curated ablations: batch 512 (carry: 128 sessions × window 4).

## Evaluation

- Curated val top-1 (checkpoint selection).
- Local Hermes top-1 (same 13 labels).
- Constant **always-terminal**.
- **Flat** twin trained in the same loop when applicable.
- Small LLM baseline: Qwen2.5-1.5B-Instruct, greedy ≤8 new tokens, parse closed label set ([scripts/score_local_next_tool.py](../scripts/score_local_next_tool.py)).

## Distillation policy

Teachers at most **9B** parameters (`qwen3.5:9b`). Allowed: verifier-filtered labels, observation text cleanup, confidence-gated pseudo-labels. Forbidden: matching teacher hidden states or logits into TextNCA weights.

## Reporting

Every completed run is registered under `paper/si/experiments/EXP-*.json` via [scripts/record_experiment.py](../scripts/record_experiment.py). Main-text claims use `status: primary` only.
