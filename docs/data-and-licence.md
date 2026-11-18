# Data and licence

## Data

- The Hermes next-tool benchmark is published as the Hugging Face dataset `cap-jmk-real/hermes-next-tool-mix`.
- The BFCL split for E1 and E2 is built by `stage0/e1_prepare_bfcl_splits.py` from the public Berkeley Function Calling Leaderboard files (`gorilla-llm/Berkeley-Function-Calling-Leaderboard`). Check that dataset's own licence before redistributing it.
- `data/curated/sample.jsonl` and `data/extra/sample.jsonl` are small samples. Full trace files, checkpoints and logs stay outside the repository (`.gitignore` excludes them).
- Dataset sources and their licences are listed in the working notes `docs/curated-traces.md`, `docs/extra-traces.md` and `docs/agent-trace-datasets.md`.

## Licence

- Code: MIT, see `LICENSE`.
- Manuscript text and figures: CC BY 4.0.
- Third-party datasets keep their own licences.
