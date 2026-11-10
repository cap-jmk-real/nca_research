# Proposal: LLM next-tool baselines on the rigour_1110 splits (NOT RUN)

Status: proposal only. It was not run because it needs a paid API, and paid LLM calls were not approved.

## Question
How accurate is a hosted, instruction-tuned LLM at the same next-tool decision, and what does that cost per turn,
compared with the tuned small routers (Table "rig-main" of the paper)?

## Design
- **Splits:** the same Hermes, BFCL, ToolBench and API-Bank test splits, plus hermes_ood. Use the full split; never select on test.
- **Input:** the same information the router sees: the goal plus the last four (tool, result) steps, truncated as in the cell featuriser, and the benchmark's tool subset written as a list of names and one-line descriptions.
- **Output:** constrained to one tool name (JSON enum / tool-choice API). Unparseable answers count as wrong.
- **Variants:**
  - (a) zero-shot;
  - (b) 8-shot, with examples drawn from the training split by a fixed seed;
  - (c) 8-shot retrieval-augmented, using the k-NN neighbours from the training split.
- **Selection:** prompt variant and temperature (0 only) chosen on the validation split, 3 repeats where the API is nondeterministic.
- **Metrics:** top-1 accuracy; paired difference to the tree GNCA (seed-clustered bootstrap, Holm within an LLM family); latency p50/p95 from the client; USD per 1,000 turns.
- **Models:** one small and one mid-size hosted model from an EU region.

## Cost estimate (to be confirmed before any run)
About 9,000 test turns, plus about 4,000 validation turns × 3 variants × 2 models, at about 600 input tokens per turn, comes to about 47 M input tokens. That is roughly EUR 5-40, depending on the model; this must be checked against the provider's price list and the EUR 20/month cap.

## Approval needed
Julian must approve the provider, the model list and the budget before anything runs. No API key is stored in the repo.
