# Manuscript outline

**Title:** *Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection*

**Status:** Living outline. Every Results claim must cite an `EXP-*` id from [si/experiments/](si/experiments/).

## 1. Introduction

- Agent tool loops leave structured traces; next-tool prediction is a closed evaluation.
- Local update rules (NCA) as a substrate for binding goal + recent steps + memory.
- Gap: NCAs studied mostly for images; TextNCA (Mittal et al., 2026) probes LM, not tool routing.
- Contributions (tentative — only after primary EXP records exist):
  1. 1D growing-NCA-style rule on Hermes next-tool labels.
  2. Equal-budget message-passing ablations (carry, halt, long-range) at ≥150k steps.
  3. Comparison to flat readout and small local LLM baselines under a 9B distillation cap.

## 2. Related work

- Growing NCA (Mordvintsev et al., 2020); TextNCA; BraiNCA.
- GNN failure modes: oversmoothing, oversquashing, underreaching; Adaptive Message Passing.
- Subquadratic / hybrid sequence models (Mamba, DeltaNet, xLSTM) as state design, not cost savings at L=5.
- Multi-harness tool RL (FineEnvs / Harbor / OpenEnv).
- Long training for structured reasoning (Sudoku trial-and-error transformers; cite in Methods training budget).

Pointers: [docs/literature.md](../docs/literature.md), [docs/multi-harness-and-message-passing.md](../docs/multi-harness-and-message-passing.md).

## 3. Method

See [methods.md](methods.md). SI: architecture equations, hyperparameter table, data licenses.

## 4. Experimental setup

- Data: curated Hermes + extras; local `~/.hermes` held out.
- Metrics: top-1; always-terminal; flat twin; Qwen2.5-1.5B closed-set baseline.
- Long-training floor: 150k AdamW steps for architecture claims.
- Distillation: teachers ≤ 9B; labels / observation text only.

Gates: [config/benchmark_gates.json](../config/benchmark_gates.json).

## 5. Results

Populate only from [si/tables/](si/tables/) and primary EXP records.

1. Primary mix13 vs flat vs always-terminal (curated + local).
2. Message-passing ablations at 150k (not 8k smoke).
3. FineEnvs-prior continuation.
4. LLM baselines (1.5B; later 9B).
5. Benchmark suite (BFCL, NESTFUL, τ2, AppWorld) when heads allow.
6. **Specialist coder stack** ([docs/specialist-coder-stack.md](../docs/specialist-coder-stack.md)): Python → TypeScript → Shell students (distilled ≤9B), orchestrator routing, open harnesses (LiveCodeBench, SWE-bench Lite, Terminal-Bench, tsc smoke).

## 6. Discussion

- Local vs curated prior mismatch.
- What long-range / carry / halt do after adequate training.
- Limits of 13-name head for AST / nested APIs.

## 7. Conclusion

## Supporting information

Index: [si/README.md](si/README.md). Tables S1–S3 generated; add S4+ for learning curves and regeneration plots.
