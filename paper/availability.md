# Availability (paper)

**Canonical draft:** PR #5 (`cursor/review-majors-remaining-fdeb`). Self-review: `paper/REVIEW.md`.

## Code (public)

| Item | Location |
| --- | --- |
| Repository | https://github.com/cap-jmk-real/nca_research |
| Manuscript pin | `9cbe9be` — https://github.com/cap-jmk-real/nca_research/tree/9cbe9be |
| Primary benchmark JSON | `stage0/stage0_results/full_history_benchmarks.json` (`EXP-20261102-001`) |
| Grokking JSON | `stage0/stage0_results/ultra_grokking_results.json` (`EXP-20261102-002`) |
| SI registry | `paper/si/experiments/` + `paper/si/experiments_index.json` |
| Open runs / exports | `paper/JULIAN_TODO.md` |

**License:** none in repo yet — see `paper/JULIAN_TODO.md`.

## Hugging Face (top-4 slots — revisions not pinned)

| Slot | Role | Repository | Revision |
| --- | --- | --- | --- |
| A | Best TextNCA next-tool | `cap-jmk-real/textnca-next-tool` | _unpinned_ |
| B | Best baseline comparator | `cap-jmk-real/baseline-next-tool` | _unpinned_ |
| C | Best specialist student | `cap-jmk-real/specialist-python` | _unpinned_ |
| D | Best scrubbed dataset | `cap-jmk-real/hermes-next-tool-mix` | _unpinned_ |

Policy: [docs/huggingface-distribution.md](../docs/huggingface-distribution.md).  
Selector: `scripts/select_top4_artifacts.py`.

## Offline / exploratory SI records (author-reported aggregates)

| Claim | SI record | Notes |
| --- | --- | --- |
| Transformer 6L-4H + constrained Qwen Table 1 rows | `EXP-20261102-004` | Author-reported offline; no score JSON in repo |
| Dual-system pilot (GNCA column, token audit) | `EXP-20261102-003` | Preliminary; per-probe BFCL ledger not in repo |
| H1b no-class-weight seeds | _(none)_ | Not registered in `paper/si/` |

## Reproduce

```bash
cd formal_proofs && lake build && lake exe ncametalearning
python stage0/run_full_history_benchmarks.py --steps 10000 --seeds 0,1,2
python stage0/run_ultra_grokking_nca.py
cd paper && pdflatex main.tex && pdflatex main.tex
```

Long-train architectural claims require ≥150k steps ([docs/auto-research-loop.md](../docs/auto-research-loop.md)).
