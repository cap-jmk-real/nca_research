#!/usr/bin/env python3
"""Evaluate specialist / orchestrator checkpoints on open harnesses.

Harness adapters are opt-in: if the package or Docker image is missing, the
script writes a blocked EXP-ready JSON stub instead of failing silently.

Supported --harness ids (config/specialists.json):
  LiveCodeBench, SWE-bench_Lite, Terminal-Bench, tsc_repo_smoke, multi_specialist_smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SSD = Path("/Volumes/Extreme SSD/nca_research/results/specialists")
REPO = Path(__file__).resolve().parents[1]


def _stub(harness: str, lang: str, reason: str) -> dict:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harness": harness,
        "lang": lang,
        "status": "blocked",
        "blocker": reason,
        "score": None,
        "n": 0,
        "si_note": "Register as EXP with status preliminary/blocked once unblocked.",
    }


def eval_livecodebench(lang: str, limit: int) -> dict:
    if lang != "python":
        return _stub("LiveCodeBench", lang, "LiveCodeBench is python-only here")
    try:
        import livecodebench  # noqa: F401
    except ImportError:
        return _stub(
            "LiveCodeBench",
            lang,
            "livecodebench not installed; see https://github.com/LiveCodeBench/LiveCodeBench",
        )
    return _stub("LiveCodeBench", lang, "imported but runner not wired (next step)")


def eval_swe_lite(lang: str, limit: int) -> dict:
    try:
        import swebench  # noqa: F401
    except ImportError:
        return _stub("SWE-bench_Lite", lang, "swebench not installed; Docker required for full score")
    return _stub("SWE-bench_Lite", lang, f"imported; wire instance slice for lang={lang}, limit={limit}")


def eval_terminal_bench(lang: str, limit: int) -> dict:
    return _stub(
        "Terminal-Bench",
        lang,
        "Harbor/terminal-bench sandbox not configured on this machine",
    )


def eval_tsc_smoke(lang: str, limit: int) -> dict:
    if lang != "typescript":
        return _stub("tsc_repo_smoke", lang, "typescript only")
    data = Path("/Volumes/Extreme SSD/nca_research/datasets/specialists/typescript")
    if not data.is_dir():
        return _stub("tsc_repo_smoke", lang, "compile typescript specialist dataset first")
    n = sum(1 for p in data.glob("*.jsonl") for _ in p.open())
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harness": "tsc_repo_smoke",
        "lang": lang,
        "status": "dataset_ready",
        "blocker": "apply-patch + tsc runner not implemented",
        "dataset_rows": n,
        "score": None,
    }


def eval_multi_smoke(lang: str, limit: int) -> dict:
    root = Path("/Volumes/Extreme SSD/nca_research/datasets/specialists")
    present = {
        name: (root / name).is_dir()
        for name in ("python", "typescript", "shell", "orchestrator")
    }
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "harness": "multi_specialist_smoke",
        "lang": lang,
        "status": "scaffold",
        "specialists_present": present,
        "score": None,
        "note": "End-to-end plug test once students exist",
    }


HARNESS = {
    "LiveCodeBench": eval_livecodebench,
    "SWE-bench_Lite": eval_swe_lite,
    "Terminal-Bench": eval_terminal_bench,
    "tsc_repo_smoke": eval_tsc_smoke,
    "multi_specialist_smoke": eval_multi_smoke,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="python")
    parser.add_argument("--harness", required=True, choices=sorted(HARNESS))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    SSD.mkdir(parents=True, exist_ok=True)
    result = HARNESS[args.harness](args.lang, args.limit)
    out = SSD / f"eval_{args.harness}_{args.lang}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
