#!/usr/bin/env python3
"""Prepare scrubbed specialist / next-tool jsonl packs for Hugging Face.

Writes under /Volumes/Extreme SSD/nca_research/results/hf/datasets/<name>/.
Refuses the repo results/ directory. Never uploads parquet dumps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nca_text.ssd import assert_external  # noqa: E402

CFG = json.loads((ROOT / "config" / "hf_publish.json").read_text())
SCRUB = CFG["scrub"]
SRC = Path("/Volumes/Extreme SSD/nca_research/datasets/specialists")
OUT = Path("/Volumes/Extreme SSD/nca_research/results/hf/datasets")


def _outdir(name: str) -> Path:
    dest = OUT / name
    assert_external(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _scrub_row(row: dict) -> dict:
    goal = str(row.get("goal") or "")
    goal = " ".join(goal.split())[: int(SCRUB["goal_chars"])]
    paths = list(row.get("path_hints") or [])[: int(SCRUB["path_hints_max"])]
    out = {
        "specialist": row.get("specialist") or row.get("route_label"),
        "route_label": row.get("route_label"),
        "session_id": row.get("session_id"),
        "goal": goal,
        "path_hints": paths,
        "source": row.get("source"),
        "resolved": row.get("resolved"),
    }
    return out


def prepare(lang: str, limit: int | None) -> dict:
    src = SRC / lang
    dest = _outdir(f"specialist-{lang}-v1")
    n_in = n_out = 0
    out_path = dest / "train.jsonl"
    if not src.is_dir():
        return {"lang": lang, "status": "blocked", "reason": f"missing {src}"}
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted(src.glob("*.jsonl")):
            if path.name.startswith("._"):
                continue
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    n_in += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    scrubbed = _scrub_row(row)
                    out.write(json.dumps(scrubbed, ensure_ascii=False) + "\n")
                    n_out += 1
                    if limit is not None and n_out >= limit:
                        break
            if limit is not None and n_out >= limit:
                break
    card = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lang": lang,
        "n_in": n_in,
        "n_out": n_out,
        "text_cap": SCRUB["goal_chars"],
        "path": str(out_path),
        "rule": "compiled fields only; no parquet dumps",
    }
    (dest / "prepare_meta.json").write_text(json.dumps(card, indent=2) + "\n")
    (dest / "README.md").write_text(
        f"# specialist-{lang}-v1\n\nScrubbed rows for NCA research release.\n"
        f"goal≤{SCRUB['goal_chars']} chars. n={n_out}.\n"
    )
    return card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["python", "typescript", "shell", "orchestrator"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"lang": args.lang, "src": str(SRC / args.lang), "would_write": str(OUT)}, indent=2))
        return
    print(json.dumps(prepare(args.lang, args.limit), indent=2), flush=True)


if __name__ == "__main__":
    main()
