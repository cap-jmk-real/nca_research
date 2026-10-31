# Paper workspace

Working material for a methods + results manuscript on **1D trace neural cellular automata for next-tool prediction**, with supporting information (SI) that can be compiled into a paper.

| Path | Role |
| --- | --- |
| [outline.md](outline.md) | Section outline and claim inventory |
| [methods.md](methods.md) | Methods draft (architecture, data, training, eval) |
| [si/README.md](si/README.md) | Supporting information index |
| [si/experiment_schema.json](si/experiment_schema.json) | JSON Schema for every registered run |
| [si/experiments/](si/experiments/) | One record per experiment (`EXP-YYYYMMDD-###.json`) |
| [si/tables/](si/tables/) | Generated markdown tables for the manuscript |
| [si/figures/](si/figures/) | Captions + paths to SSD TensorBoard / plots |

**Rule:** if a number appears in the paper draft, it must cite an `EXP-*` id (or a table built only from those ids). Smoke runs (`status: smoke`) never enter main-text claims.

Large artifacts (checkpoints, jsonl logs, TensorBoard) stay on `/Volumes/Extreme SSD/nca_research/results/` and are referenced by absolute path inside each experiment record.

Register or refresh SI from the repo root:

```bash
python scripts/record_experiment.py backfill
python scripts/record_experiment.py build-tables
```
