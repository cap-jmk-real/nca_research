#!/usr/bin/env python3
"""Compile language-tagged specialist corpora from staged coding trajectories.

Writes jsonl under /Volumes/Extreme SSD/nca_research/datasets/specialists/.
Starts with python and typescript; shell is a third pass over terminal-heavy rows.

Does not download new Hub dumps (uses staged coding/ + optional Hermes extra).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SSD = Path("/Volumes/Extreme SSD/nca_research")
CODING = SSD / "datasets" / "coding"
OUT = SSD / "datasets" / "specialists"
REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "specialists.json"
# Prefer a working pyarrow install; do not prefer py-tools if broken.
_PYDEPS = SSD / "results" / "pydeps"
if _PYDEPS.is_dir():
    sys.path = [str(_PYDEPS)] + [p for p in sys.path if "py-tools" not in p]

PY_EXT = re.compile(r"\.(py|pyi)\b", re.I)
TS_EXT = re.compile(r"\.(tsx?|jsx?|mts|cts)\b", re.I)
SH_EXT = re.compile(r"\.(sh|bash)\b|Dockerfile\b", re.I)
PATHISH = re.compile(
    r"(?:[A-Za-z0-9_\-./]+?\.(?:py|pyi|ts|tsx|js|jsx|mts|cts|sh|bash))\b"
)


def _load_config() -> dict:
    if REPO_CONFIG.is_file():
        return json.loads(REPO_CONFIG.read_text())
    return {}


def _iter_parquet_rows(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for batch in table.to_batches(max_chunksize=64):
        for row in batch.to_pylist():
            yield row


def _text_blob(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except TypeError:
        return str(obj)


def _paths_in(text: str) -> list[str]:
    return list(dict.fromkeys(PATHISH.findall(text or "")))


def _lang_scores(text: str, paths: list[str]) -> dict[str, float]:
    blob = (text or "") + " " + " ".join(paths)
    py = len(PY_EXT.findall(blob)) + sum(1 for p in paths if p.endswith((".py", ".pyi")))
    ts = len(TS_EXT.findall(blob)) + sum(
        1 for p in paths if re.search(r"\.(tsx?|jsx?|mts|cts)$", p, re.I)
    )
    sh = len(SH_EXT.findall(blob)) + sum(
        1 for p in paths if re.search(r"\.(sh|bash)$", p, re.I)
    )
    # Keyword hints
    if re.search(r"\b(pytest|unittest|pip |conda |django|flask)\b", blob, re.I):
        py += 2
    if re.search(r"\b(tsc|npm |pnpm |yarn |vitest|jest|ts-node)\b", blob, re.I):
        ts += 2
    if re.search(r"\b(bash |sudo |chmod |systemctl |apt-get )\b", blob, re.I):
        sh += 2
    total = float(py + ts + sh) or 1.0
    return {"python": py / total, "typescript": ts / total, "shell": sh / total, "raw": {"py": py, "ts": ts, "sh": sh}}


def _route(scores: dict, min_score: float = 0.45) -> str | None:
    ranking = sorted(
        ((k, scores[k]) for k in ("python", "typescript", "shell")),
        key=lambda kv: kv[1],
        reverse=True,
    )
    lang, score = ranking[0]
    if score < min_score:
        return None
    return lang


def _from_openhands_messages(row: dict, source: str) -> dict | None:
    messages = row.get("messages") or row.get("trajectory") or []
    text = _text_blob(messages)
    paths = _paths_in(text)
    scores = _lang_scores(text, paths)
    lang = _route(scores)
    if not lang:
        return None
    goal = ""
    for msg in messages if isinstance(messages, list) else []:
        if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
            goal = str(msg.get("content") or "")[:500]
            break
    return {
        "specialist": lang,
        "route_label": lang,
        "session_id": str(row.get("instance_id") or row.get("id") or ""),
        "lang_score": scores[lang],
        "lang_raw": scores["raw"],
        "goal": goal,
        "resolved": row.get("resolved"),
        "source": source,
        "path_hints": paths[:32],
        "n_messages": len(messages) if isinstance(messages, list) else None,
    }


def _from_swe_smith_tool(row: dict, source: str) -> dict | None:
    text = _text_blob(row)
    paths = _paths_in(text)
    scores = _lang_scores(text, paths)
    lang = _route(scores)
    if not lang:
        return None
    return {
        "specialist": lang,
        "route_label": lang,
        "session_id": str(row.get("instance_id") or row.get("id") or ""),
        "lang_score": scores[lang],
        "lang_raw": scores["raw"],
        "goal": str(row.get("problem_statement") or row.get("prompt") or "")[:500],
        "resolved": row.get("resolved"),
        "source": source,
        "path_hints": paths[:32],
    }


def _scan_dir(name: str, directory: Path, limit: int, langs: set[str]):
    if not directory.is_dir():
        print(f"skip missing {directory}", flush=True)
        return
    count = 0
    writers = {}
    stats = Counter()
    for path in sorted(directory.rglob("*.parquet")):
        if path.name.startswith("._"):
            continue
        print(f"scan {path.relative_to(CODING)}", flush=True)
        for row in _iter_parquet_rows(path):
            if "messages" in row or "trajectory" in row:
                rec = _from_openhands_messages(row, name)
            else:
                rec = _from_swe_smith_tool(row, name)
            if not rec or rec["specialist"] not in langs:
                continue
            lang = rec["specialist"]
            if lang not in writers:
                out_dir = OUT / lang
                out_dir.mkdir(parents=True, exist_ok=True)
                writers[lang] = (out_dir / f"{name}.jsonl").open("w")
            writers[lang].write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[lang] += 1
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break
    for handle in writers.values():
        handle.close()
    print(f"done {name}: {dict(stats)} (rows_kept={count})", flush=True)


def _orchestrator_index(langs: set[str]) -> None:
    """Build a tiny route dataset from specialist jsonl heads."""
    out_dir = OUT / "orchestrator"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "routes_from_specialists.jsonl"
    n = 0
    with out_path.open("w") as out:
        for lang in langs:
            for path in sorted((OUT / lang).glob("*.jsonl")):
                with path.open() as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        out.write(
                            json.dumps(
                                {
                                    "specialist": "orchestrator",
                                    "route_label": row.get("route_label") or lang,
                                    "session_id": row.get("session_id"),
                                    "goal": row.get("goal"),
                                    "path_hints": row.get("path_hints"),
                                    "source": row.get("source"),
                                    "lang_score": row.get("lang_score"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n += 1
                        if n >= 200_000:
                            break
                if n >= 200_000:
                    break
    print(f"orchestrator routes written={n} -> {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["python", "typescript"],
        choices=["python", "typescript", "shell"],
    )
    parser.add_argument("--max-rows-per-source", type=int, default=50_000)
    parser.add_argument("--with-shell", action="store_true")
    args = parser.parse_args()
    langs = set(args.langs)
    if args.with_shell:
        langs.add("shell")

    cfg = _load_config()
    print(f"out={OUT} langs={sorted(langs)} config_version={cfg.get('version')}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sources = [
        ("openhands-sft", CODING / "openhands-sft"),
        ("swe-smith-trajectories-tool", CODING / "swe-smith-trajectories-tool"),
        ("r2e-gym-sft", CODING / "r2e-gym-sft"),
        # Nebius is huge; cap still applies per source
        ("swe-rebench-openhands", CODING / "swe-rebench-openhands"),
    ]
    for name, directory in sources:
        _scan_dir(name, directory, args.max_rows_per_source, langs)

    _orchestrator_index(langs)

    # Stats file for SI
    stats = {"langs": {}, "out": str(OUT)}
    for lang in sorted(langs | {"orchestrator"}):
        paths = list((OUT / lang).glob("*.jsonl")) if (OUT / lang).is_dir() else []
        n = 0
        for path in paths:
            with path.open() as handle:
                n += sum(1 for line in handle if line.strip())
        stats["langs"][lang] = {"files": [p.name for p in paths], "rows": n}
    (OUT / "compile_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
