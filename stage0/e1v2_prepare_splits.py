#!/usr/bin/env python3
"""E1/E2 v2 split: all public BFCL v3 AST + live + irrelevance files, stratified by label.

Replaces the v1 ``bfcl_e1`` split, which used only the 256-row ``--smoke`` subset
(test n=25, 80% one class). Design, fixed before any training:

* Source: every BFCL v3 non-executable single-turn file that has possible_answer
  (8 AST files) plus both irrelevance files (gold label ``respond``).
* Labels: unchanged heuristic map (bfcl_label_map.py) onto labels.json.
* Leakage guard: rows with identical normalised goal text are grouped and kept
  in one split.
* Stratification: groups are assigned per label with a fixed seed, 60/20/20
  train/val/test (largest-remainder rounding; every class with >=3 groups gets
  >=1 group in each split).
* Model selection uses val only. Test is written once and never read by this
  script for any decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from bfcl_label_map import gold_label_from_answer
from common import DATA, SPLITS, dist
from e1_prepare_bfcl_splits import BFCL_REPO, _hf_download, _load_json_rows, _question_text

SPLIT_NAME = "bfcl_e1v2"
SPLIT_SEED_V2 = 20261103
FRACS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
AST_FILES = (
    "BFCL_v3_simple.json",
    "BFCL_v3_multiple.json",
    "BFCL_v3_parallel.json",
    "BFCL_v3_parallel_multiple.json",
    "BFCL_v3_live_simple.json",
    "BFCL_v3_live_multiple.json",
    "BFCL_v3_live_parallel.json",
    "BFCL_v3_live_parallel_multiple.json",
)
IRREL_FILES = ("BFCL_v3_irrelevance.json", "BFCL_v3_live_irrelevance.json")
MANIFEST = Path(__file__).resolve().parent / "data_manifests" / f"{SPLIT_NAME}.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_examples(cache_dir: Path) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    sources: dict[str, str] = {}
    for name in AST_FILES:
        q = _hf_download(name, cache_dir)
        a = _hf_download(f"possible_answer/{name}", cache_dir)
        sources[name] = _sha(q)
        sources[f"possible_answer/{name}"] = _sha(a)
        answers = {str(r["id"]): r for r in _load_json_rows(a)}
        for row in _load_json_rows(q):
            rid = str(row["id"])
            examples.append(_ex(row, gold_label_from_answer(answers.get(rid), irrelevance=False), name))
    for name in IRREL_FILES:
        q = _hf_download(name, cache_dir)
        sources[name] = _sha(q)
        for row in _load_json_rows(q):
            examples.append(_ex(row, "respond", name))
    ids = [e["session_id"] for e in examples]
    assert len(ids) == len(set(ids)), "duplicate BFCL ids"
    return examples, sources


def _ex(row: dict, label: str, source: str) -> dict:
    rid = str(row["id"])
    return {"session_id": rid, "step_index": 0, "goal": _question_text(row), "recent_steps": [],
            "target_tool": label, "source": source, "bfcl_id": rid}


def _alloc(n: int) -> dict[str, int]:
    raw = {k: n * f for k, f in FRACS}
    out = {k: int(v) for k, v in raw.items()}
    if n >= 3:
        for k in out:
            if out[k] == 0:
                out[k] = 1
    while sum(out.values()) > n:
        k = max(out, key=lambda s: out[s]); out[k] -= 1
    rem = sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True)
    i = 0
    while sum(out.values()) < n:
        out[rem[i % 3]] += 1; i += 1
    return out


def assign(examples: list[dict], seed: int = SPLIT_SEED_V2) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        groups[" ".join(e["goal"].lower().split())].append(e)
    by_label: dict[str, list[str]] = defaultdict(list)
    for key, rows in groups.items():
        lab = Counter(r["target_tool"] for r in rows).most_common(1)[0][0]
        by_label[lab].append(key)
    rng = random.Random(seed)
    parts: dict[str, list[dict]] = {k: [] for k, _ in FRACS}
    for lab in sorted(by_label):
        keys = sorted(by_label[lab]); rng.shuffle(keys)
        alloc = _alloc(len(keys)); i = 0
        for split, _ in FRACS:
            for key in keys[i : i + alloc[split]]:
                parts[split].extend(groups[key])
            i += alloc[split]
    for split in parts:
        parts[split].sort(key=lambda e: e["session_id"])
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=DATA / "hf_cache")
    args = ap.parse_args()
    examples, sources = build_examples(args.cache_dir)
    parts = assign(examples)
    out_dir = SPLITS / SPLIT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"split_name": SPLIT_NAME, "seed": SPLIT_SEED_V2, "dataset": BFCL_REPO,
                "fractions": dict(FRACS), "grouping": "normalised goal text", "n_total": len(examples),
                "source_sha256": sources, "splits": {}}
    for split, rows in parts.items():
        p = out_dir / f"{split}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        manifest["splits"][split] = {"rows": len(rows), "sha256": _sha(p),
                                     "ids_sha256": hashlib.sha256("\n".join(r["session_id"] for r in rows).encode()).hexdigest(),
                                     "labels": dist([r["target_tool"] for r in rows]),
                                     "sources": dict(Counter(r["source"] for r in rows))}
    sets = {s: {r["session_id"] for r in rows} for s, rows in parts.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({s: {"rows": v["rows"], "labels": {k: x["n"] for k, x in v["labels"].items()}} for s, v in manifest["splits"].items()}, indent=1))


if __name__ == "__main__":
    main()
