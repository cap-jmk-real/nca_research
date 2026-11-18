"""Recompute the Hermes full-history table from the committed results file.

Reads ``stage0/stage0_results/full_history_benchmarks.json`` (written by
``stage0/run_full_history_benchmarks.py``) and prints, per model, the mean and
sample standard deviation of the per-seed test top-1 (percent) on the curated
Hermes test split. This is a re-aggregation of stored results. It trains
nothing and needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "stage0" / "stage0_results" / "full_history_benchmarks.json"


def summarize(path: Path) -> list[tuple[str, float, float, int | None]]:
    """Return ``(model, mean_top1, sd_top1, params)`` rows for every model in ``path``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, runs in data["models"].items():
        top1 = [run["test"]["top1"] for run in runs]
        sd = statistics.stdev(top1) if len(top1) > 1 else 0.0
        params = runs[0].get("params")
        rows.append((name, statistics.mean(top1), sd, params))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    print(f"test turns: {data['n_test']}, seeds: {data['seeds']}")
    for name, mean, sd, params in summarize(args.results):
        extra = f", {params} params" if params else ""
        print(f"{name:<22} {mean:6.2f} +- {sd:4.2f}{extra}")


if __name__ == "__main__":
    main()
