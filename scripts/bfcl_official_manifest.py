#!/usr/bin/env python3
"""Emit a reproducible manifest for the official BFCL v4 leaderboard eval set.

Counts entries bundled with the pinned ``bfcl-eval`` wheel (not the paper's
692-turn bfcl_e1v2 routing split). Writes JSON under stage0/stage0_results/.
"""

from __future__ import annotations

import importlib.metadata as im
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "bfcl_official_protocol.json").read_text())
OUT = ROOT / "stage0" / "stage0_results" / "bfcl_official_manifest.json"


def main() -> int:
    try:
        import bfcl_eval
        import bfcl_eval.constants.category_mapping as cm
    except ImportError:
        print(
            "bfcl-eval not installed. Run: pip install 'bfcl-eval=="
            + CFG["bfcl_eval_version_pin"]
            + "'",
            file=sys.stderr,
        )
        return 2

    data_dir = Path(bfcl_eval.__path__[0]) / "data"
    prefix = cm.VERSION_PREFIX
    categories: dict[str, int | None] = {}
    counted_files: dict[str, int] = {}

    def count_file(rel: str) -> int:
        if rel in counted_files:
            return counted_files[rel]
        p = data_dir / rel
        if not p.exists():
            counted_files[rel] = 0
            return 0
        n = sum(1 for ln in p.read_text().splitlines() if ln.strip())
        counted_files[rel] = n
        return n

    memory_n = count_file(f"{prefix}_memory.json")
    web_n = count_file(f"{prefix}_web_search.json")
    for cat in cm.ALL_SCORING_CATEGORIES:
        if cat.startswith("memory_"):
            categories[cat] = memory_n or None
            continue
        if cat.startswith("web_search_"):
            categories[cat] = web_n or None
            continue
        p = data_dir / f"{prefix}_{cat}.json"
        categories[cat] = sum(1 for ln in p.read_text().splitlines() if ln.strip()) if p.exists() else None

    scoring_total = sum(counted_files.values())
    for cat in cm.ALL_SCORING_CATEGORIES:
        if cat.startswith(("memory_", "web_search_")):
            continue
        fname = f"{prefix}_{cat}.json"
        if fname not in counted_files and (data_dir / fname).exists():
            count_file(fname)
    scoring_total = sum(counted_files.values())

    fmt = data_dir / f"{prefix}_format_sensitivity.json"
    format_n = sum(1 for ln in fmt.read_text().splitlines() if ln.strip()) if fmt.exists() else 0

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bfcl_eval_version": im.version("bfcl-eval"),
        "gorilla_commit_pin": CFG["gorilla_commit_pin"],
        "version_prefix": prefix,
        "test_collection": CFG["test_collection"],
        "n_scoring_entries": scoring_total,
        "n_format_sensitivity_entries": format_n,
        "n_total_packaged_v4_files": scoring_total + format_n,
        "categories": categories,
        "routing_split_note": (
            "Paper Table multi-benchmark BFCL column uses bfcl_e1v2 test n=692 "
            "(Hermes-tool routing labels). It is not this manifest count."
        ),
        "leaderboard_note": (
            "Public Overall Acc aggregates category accuracies with bfcl evaluate; "
            "entry counts vary slightly by bfcl-eval release (multi-turn episodes "
            "vs. turns). Re-run this script after pinning bfcl-eval on engine."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
