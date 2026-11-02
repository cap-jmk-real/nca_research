# Availability (paper)

Last updated: 2026-11-02 (review pass toward ≥8.5/10).

## Code (public)

| Item | Location |
| --- | --- |
| Repository | https://github.com/cap-jmk-real/nca_research |
| Manuscript pin | `cddb8e5` — https://github.com/cap-jmk-real/nca_research/tree/cddb8e5 |
| Primary benchmark JSON | `stage0/stage0_results/full_history_benchmarks.json` (`EXP-20261102-001`) |
| Grokking JSON | `stage0/stage0_results/ultra_grokking_results.json` (`EXP-20261102-002`) |
| SI experiment registry | `paper/si/experiments/` + `paper/si/experiments_index.json` |

Lean 4 proofs: `formal_proofs/NCAMetaLearning` (`lake build`).

## Hugging Face (top-4 slots — revisions still pending promotion)

| Slot | Role | Repository | Revision |
| --- | --- | --- | --- |
| A | Best TextNCA next-tool | `cap-jmk-real/textnca-next-tool` | _pending_ |
| B | Best baseline comparator | `cap-jmk-real/baseline-next-tool` | _pending_ |
| C | Best specialist student | `cap-jmk-real/specialist-python` | _pending_ |
| D | Best scrubbed dataset | `cap-jmk-real/hermes-next-tool-mix` | _pending_ |

Policy: [docs/huggingface-distribution.md](../docs/huggingface-distribution.md).  
Selector: `scripts/select_top4_artifacts.py`.

## Pending engine exports (not blockers for text fixes)

| Claim | SI record | Status |
| --- | --- | --- |
| Transformer 6L-4H + constrained Qwen Table 1 rows | `EXP-20261102-004` | Score JSON missing (engine offline) |
| Azure / Open SOTA pilot latencies & BFCL compares | `EXP-20261102-003` | Preliminary log only; JSON re-export pending |
| H1b no-class-weight seeds | _(none)_ | Queued on engine; not in SI |

## Reproduce (local)

```bash
cd formal_proofs && lake build && lake exe ncametalearning
python stage0/run_full_history_benchmarks.py --steps 10000 --seeds 0,1,2
python stage0/run_ultra_grokking_nca.py
cd paper && pdflatex main.tex && pdflatex main.tex
```

Long-train architectural claims require ≥150k steps ([docs/auto-research-loop.md](../docs/auto-research-loop.md)).
