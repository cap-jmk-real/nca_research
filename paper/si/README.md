# Supporting information

## Contents

| Item | File |
| --- | --- |
| Experiment JSON Schema | [experiment_schema.json](experiment_schema.json) |
| Experiment records | [experiments/](experiments/) |
| Registry index | [experiments_index.json](experiments_index.json) |
| Results master | [results_master.md](results_master.md) |
| Table S1 | [tables/S1_experiment_registry.md](tables/S1_experiment_registry.md) |
| Table S2 | [tables/S2_next_tool_top1.md](tables/S2_next_tool_top1.md) |
| Table S3 | [tables/S3_claims_ledger.md](tables/S3_claims_ledger.md) |
| Figure captions | [figures/README.md](figures/README.md) |

## How to add a result

1. Finish training/eval; keep checkpoints and metrics on the Extreme SSD.
2. Register:
   - Hypotheses suite: `python scripts/record_experiment.py register --from-hypotheses-summary`
   - Or hand-authored JSON: `python scripts/record_experiment.py register --file record.json`
3. Rebuild tables: `python scripts/record_experiment.py build-tables`
4. Update claim strings in the record (`claims_supported` / `claims_not_supported`).
5. Cite the `EXP-*` id in [../outline.md](../outline.md) Results and any prose docs.

## Status meanings

| Status | Use in paper |
| --- | --- |
| `primary` | Main text + SI |
| `preliminary` | SI with caveat; short discussion only |
| `smoke` | SI appendix only; **no architecture claims** |
| `running` | Placeholder until metrics land |
| `superseded` | Kept for audit trail |
| `failed` | Negative result / aborted |

## Large artifacts

Checkpoints, jsonl logs, and TensorBoard event files are **not** stored in git. Each record’s `artifacts` object points at `/Volumes/Extreme SSD/nca_research/results/...`.
