#!/usr/bin/env python3
"""Download public BFCL eval JSON and mint session-level train/val/test jsonl (no Hermes corpus)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from bfcl_label_map import gold_label_from_answer
from common import DATA, SPLITS, SPLIT_SEED, dist

REPO = Path(__file__).resolve().parents[1]
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
SPLIT_NAME = "bfcl_e1"

AST_FILES = (
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
)
IRREL_FILE = "BFCL_v3_irrelevance.json"


def _hf_download(filename: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=BFCL_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(cache_dir / "gorilla_bfcl"),
        )
    )


def _load_json_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _question_text(row: dict) -> str:
    parts: list[str] = []
    for turn in row.get("question") or []:
        messages = turn if isinstance(turn, list) else [turn]
        for message in messages:
            if isinstance(message, dict):
                parts.append(str(message.get("content") or ""))
    for function in row.get("function") or []:
        if isinstance(function, dict) and function.get("name"):
            parts.append(str(function["name"]))
    return " ".join(parts)[:500]


def build_examples(
    cache_dir: Path,
    *,
    max_rows: int | None = None,
    ast_files: tuple[str, ...] = AST_FILES,
) -> list[dict]:
    examples: list[dict] = []
    for ast_name in ast_files:
        qpath = _hf_download(ast_name, cache_dir)
        apath = _hf_download(f"possible_answer/{ast_name}", cache_dir)
        answers = {r["id"]: r for r in _load_json_rows(apath)}
        for row in _load_json_rows(qpath):
            rid = str(row["id"])
            label = gold_label_from_answer(answers.get(rid), irrelevance=False)
            examples.append(
                {
                    "session_id": rid,
                    "step_index": 0,
                    "goal": _question_text(row),
                    "recent_steps": [],
                    "target_tool": label,
                    "source": ast_name,
                    "bfcl_id": rid,
                }
            )
            if max_rows and len(examples) >= max_rows:
                return examples

    irrel_path = _hf_download(IRREL_FILE, cache_dir)
    for row in _load_json_rows(irrel_path):
        rid = str(row["id"])
        examples.append(
            {
                "session_id": rid,
                "step_index": 0,
                "goal": _question_text(row),
                "recent_steps": [],
                "target_tool": "respond",
                "source": IRREL_FILE,
                "bfcl_id": rid,
            }
        )
        if max_rows and len(examples) >= max_rows:
            break
    return examples


def write_splits(examples: list[dict], split_name: str = SPLIT_NAME) -> dict:
    by_sess: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_sess[ex["session_id"]].append(ex)
    ids = sorted(by_sess)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train_ids = set(ids[n_test + n_val :])

    parts = {
        "train": [ex for sid in train_ids for ex in by_sess[sid]],
        "val": [ex for sid in val_ids for ex in by_sess[sid]],
        "test": [ex for sid in test_ids for ex in by_sess[sid]],
    }

    out_dir = SPLITS / split_name
    if out_dir.exists():
        for p in out_dir.glob("*.jsonl"):
            pass  # allow overwrite of bfcl_e1 only
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"seed": SPLIT_SEED, "split_name": split_name, "dataset": BFCL_REPO}
    for split, rows in parts.items():
        with open(out_dir / f"{split}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[split] = {
            "rows": len(rows),
            "sessions": len({r["session_id"] for r in rows}),
            "labels": dist([r["target_tool"] for r in rows]),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DATA / "hf_cache")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Only BFCL_v3_simple + 64 irrelevance rows (fast download)",
    )
    args = ap.parse_args()
    max_rows = args.max_rows if args.max_rows > 0 else None
    ast_files: tuple[str, ...] = ("BFCL_v3_simple.json",) if args.smoke else AST_FILES
    if args.smoke and max_rows is None:
        max_rows = 256
    examples = build_examples(args.cache_dir, max_rows=max_rows, ast_files=ast_files)
    summary = write_splits(examples)
    print(json.dumps({"n_examples": len(examples), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
