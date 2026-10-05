# Availability (paper)

**Draft — fill links after top-4 promotion.**

## Hugging Face (only four live artifacts)

| Slot | Role | Repository | Revision |
| --- | --- | --- | --- |
| A | Best TextNCA next-tool | `cap-jmk-real/textnca-next-tool` | _pending_ |
| B | Best baseline comparator | `cap-jmk-real/baseline-next-tool` | _pending_ |
| C | Best specialist student | `cap-jmk-real/specialist-python` | _pending_ |
| D | Best scrubbed dataset | `cap-jmk-real/hermes-next-tool-mix` | _pending_ |

Policy: [docs/huggingface-distribution.md](../docs/huggingface-distribution.md).
Selector: `scripts/select_top4_artifacts.py`. Quants: `scripts/quantize_checkpoint.py`
(fp16 → fp8 → int8 → nf4 → ≈2.5 bpw IQ2-class). At most one quant companion
per slot; promote only if ≥98% of fp16 top-1 or Pareto-better on size/latency.

## Code

- Git commit: _pending_
- Branch used for release cut: _pending_
- SI experiment ids for claimed numbers: see `paper/si/experiments/`

## Reproduce

```bash
python scripts/select_top4_artifacts.py --write
python scripts/quantize_checkpoint.py --from-top4 --slot A
python scripts/prepare_hf_dataset.py --lang python
python scripts/publish_hf.py all --from-top4 --dry-run
```

Long-train architectural claims require ≥150k steps
([docs/auto-research-loop.md](../docs/auto-research-loop.md)).
