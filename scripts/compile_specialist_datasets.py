#!/usr/bin/env python3
"""Compile language-tagged specialist corpora from staged coding trajectories.

Writes jsonl under /Volumes/Extreme SSD/nca_research/datasets/specialists/.
Starts with python and typescript; shell is a third pass over terminal-heavy rows.

Does not download new Hub dumps (uses staged coding/ + optional Hermes extra).
"""

from __future__ import annotations

import argparse
import gc
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
# Patch columns only. Measured on this 16 GB Mac: 32768 model_patch rows ≈ 1.9 GB RSS.
# Trajectory bodies are ~775 MB uncompressed per 4096-row group and are not loaded.
BATCH_ROWS = 32768
_PATCH_COLUMNS = (
    "model_patch",
    "patch",
    "problem_statement",
    "prompt",
    "instance_id",
    "id",
    "traj_id",
    "trajectory_id",
    "repo",
    "base_commit",
    "resolved",
)
_MESSAGE_COLUMNS = _PATCH_COLUMNS + ("messages", "conversations", "trajectory")


def _rss_mb() -> float:
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _load_config() -> dict:
    if REPO_CONFIG.is_file():
        return json.loads(REPO_CONFIG.read_text())
    return {}


def _columns_for(names: set[str]) -> list[str] | None:
    """Project routing columns. Never pull trajectory bodies when a patch exists."""
    if ("model_patch" in names or "patch" in names) and "trajectory" in names:
        wanted = _PATCH_COLUMNS
    else:
        wanted = _MESSAGE_COLUMNS
    columns = [name for name in wanted if name in names]
    return columns or None


def _iter_parquet_rows(path: Path):
    """Stream one parquet file in batches. Drop each batch before reading the next."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path, memory_map=True)
    try:
        columns = _columns_for(set(pf.schema_arrow.names))
        heavy = bool(columns) and any(
            name in {"messages", "conversations", "trajectory"} for name in columns
        )
        # Message shards: 1024 rows measured ~1.5 GB RSS. Patches use BATCH_ROWS.
        batch_size = 1024 if heavy else BATCH_ROWS
        print(
            f"read {path.name} columns={columns} batch={batch_size}",
            flush=True,
        )
        for batch in pf.iter_batches(
            batch_size=batch_size, columns=columns, use_threads=False
        ):
            rows = batch.to_pylist()
            del batch
            pa.default_memory_pool().release_unused()
            print(
                f"batch {path.name} rows={len(rows)} rss_mb={_rss_mb():.0f}",
                flush=True,
            )
            try:
                for row in rows:
                    yield row
            finally:
                del rows
    finally:
        pf.close()
        gc.collect()
        try:
            import pyarrow as pa

            pa.default_memory_pool().release_unused()
        except Exception:
            pass


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


def _bounded_text(obj, limit: int = 64_000) -> str:
    """Sample string fields without dumping a whole trajectory into one allocation."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj[:limit]
    chunks: list[str] = []
    used = 0

    def walk(value, depth: int = 0) -> None:
        nonlocal used
        if used >= limit or depth > 12:
            return
        if isinstance(value, str):
            take = value[: limit - used]
            chunks.append(take)
            used += len(take)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)
                if used >= limit:
                    return
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)
                if used >= limit:
                    return

    walk(obj)
    return "\n".join(chunks)


def _from_openhands_messages(row: dict, source: str) -> dict | None:
    messages = row.get("messages") or row.get("trajectory") or []
    text = _bounded_text(messages)
    paths = _paths_in(text)
    scores = _lang_scores(text, paths)
    lang = _route(scores)
    if not lang:
        return None
    goal = ""
    for msg in messages if isinstance(messages, list) else []:
        if isinstance(msg, dict) and msg.get("role") in {"user", "human"}:
            # 240 chars is enough for distill prompts (teacher sees ≤160).
            goal = str(msg.get("content") or "")[:240]
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
        "path_hints": paths[:8],
        "n_messages": len(messages) if isinstance(messages, list) else None,
    }


def _from_swe_smith_tool(row: dict, source: str) -> dict | None:
    # Avoid json.dumps(entire row) — huge RAM/CPU vs transformers' token cost.
    goal = str(row.get("problem_statement") or row.get("prompt") or "")[:240]
    bits = [goal]
    for key in ("instance_id", "repo", "base_commit"):
        if row.get(key):
            bits.append(str(row[key])[:500])
    patch = row.get("model_patch") or row.get("patch")
    if isinstance(patch, str) and patch:
        if not goal:
            goal = patch[:240]
        bits.append(patch[:200_000])
    else:
        for key in ("messages", "trajectory", "conversations"):
            if key in row and row[key] is not None:
                bits.append(_text_blob(row[key])[:48_000])
                break
    text = "\n".join(bits)
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
        "goal": goal,
        "resolved": row.get("resolved"),
        "source": source,
        "path_hints": paths[:8],
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
        rows = _iter_parquet_rows(path)
        try:
            for row in rows:
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
        finally:
            rows.close()
        if count >= limit:
            break
    for handle in writers.values():
        handle.close()
    print(f"done {name}: {dict(stats)} (rows_kept={count}) rss_mb={_rss_mb():.0f}", flush=True)


def _jsonl_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("*.jsonl")
        if not path.name.startswith("._") and path.is_file()
    )


def _orchestrator_index(langs: set[str]) -> None:
    """Build a tiny route dataset from specialist jsonl heads."""
    out_dir = OUT / "orchestrator"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "routes_from_specialists.jsonl"
    n = 0
    with out_path.open("w") as out:
        for lang in langs:
            for path in _jsonl_files(OUT / lang):
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


def _source_rows(name: str, langs: set[str]) -> int:
    total = 0
    for lang in langs:
        path = OUT / lang / f"{name}.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open() as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _finished_in_log(log_path: Path) -> set[str]:
    """Sources whose latest compile.log event is a completed scan."""
    last: dict[str, str] = {}
    if not log_path.is_file():
        return set()
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("done "):
            name = line.split(":", 1)[0].removeprefix("done ").strip()
            last[name] = "done"
        elif line.startswith("scan "):
            rel = line.removeprefix("scan ").strip()
            last[rel.split("/", 1)[0]] = "scan"
    return {name for name, status in last.items() if status == "done"}


def _remove_partial(name: str, langs: set[str]) -> int:
    removed = 0
    for lang in langs:
        path = OUT / lang / f"{name}.jsonl"
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


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
    print(
        f"out={OUT} langs={sorted(langs)} batch_rows={BATCH_ROWS} config_version={cfg.get('version')}",
        flush=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)

    sources = [
        ("openhands-sft", CODING / "openhands-sft"),
        ("swe-smith-trajectories-tool", CODING / "swe-smith-trajectories-tool"),
        ("r2e-gym-sft", CODING / "r2e-gym-sft"),
        # Nebius is huge; cap still applies per source
        ("swe-rebench-openhands", CODING / "swe-rebench-openhands"),
    ]
    finished = _finished_in_log(OUT / "compile.log")
    for name, directory in sources:
        existing = _source_rows(name, langs)
        at_cap = existing >= args.max_rows_per_source
        if at_cap or (name in finished and existing > 0):
            print(
                f"skip {name}: existing_rows={existing} at_cap={at_cap} log_done={name in finished}",
                flush=True,
            )
            continue
        removed = _remove_partial(name, langs)
        if removed:
            print(f"removed partial {name}: {removed} file(s)", flush=True)
        _scan_dir(name, directory, args.max_rows_per_source, langs)

    _orchestrator_index(langs)
    print(f"peak_rss_mb={_rss_mb():.0f}", flush=True)

    # Stats file for SI
    stats = {"langs": {}, "out": str(OUT)}
    for lang in sorted(langs | {"orchestrator"}):
        paths = _jsonl_files(OUT / lang)
        n = 0
        for path in paths:
            with path.open() as handle:
                n += sum(1 for line in handle if line.strip())
        stats["langs"][lang] = {"files": [p.name for p in paths], "rows": n}
    (OUT / "compile_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
