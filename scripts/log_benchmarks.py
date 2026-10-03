#!/usr/bin/env python3
"""Write benchmark scores to TensorBoard on the external SSD.

Accepts a JSON object of scores. No scores are invented.

    python3 scripts/log_benchmarks.py scores.json
    python3 scripts/log_benchmarks.py --json '{"BFCL": 0.42, "tau2": 0.15}'

A wrapped object is also fine: ``{"step": 1, "scores": {"BFCL": 0.42}}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nca_text.tb_log import BENCH_DIR, log_benchmarks  # noqa: E402


def _load(args: argparse.Namespace) -> dict:
    if args.json:
        payload = json.loads(args.json)
    elif args.scores == "-":
        payload = json.loads(sys.stdin.read())
    elif args.scores:
        payload = json.loads(Path(args.scores).read_text())
    elif not sys.stdin.isatty():
        payload = json.loads(sys.stdin.read())
    else:
        raise SystemExit("pass a JSON file, --json, or stdin")
    if not isinstance(payload, dict):
        raise SystemExit("benchmark JSON must be an object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Log benchmark scores to TensorBoard")
    parser.add_argument("scores", nargs="?", help="JSON file of scores, or - for stdin")
    parser.add_argument("--json", help="inline JSON object")
    args = parser.parse_args()
    tags = log_benchmarks(_load(args))
    if not tags:
        raise SystemExit(f"no finite scores written under {BENCH_DIR}")
    print(json.dumps({"logdir": str(BENCH_DIR), "tags": tags}), flush=True)


if __name__ == "__main__":
    main()
