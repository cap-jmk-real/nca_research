#!/usr/bin/env python3
"""Curate next-tool rows from public Hermes agent traces.

Reads the DJLougen filtered slice of lambda/hermes-agent-reasoning-traces
and writes one row per decision point. Full jsonl goes to the external data
root (see docs/data-root.md). This script never reads ~/.hermes.

Rerun from the repo root:

    python3 scripts/curate_traces.py
    NCA_DATA_ROOT=/Volumes/Other/nca_research python3 scripts/curate_traces.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

DEFAULT_DATA_ROOT = Path("/Volumes/Extreme SSD/nca_research")
FILTERED_REPO = "DJLougen/hermes-agent-traces-filtered"
FILTERED_REVISION = "f91a8ef669194408a706741cd6f32a1686a8fd78"
FILTERED_FILENAME = "data/train.jsonl"
UPSTREAM_REPO = "lambda/hermes-agent-reasoning-traces"

GOAL_CHARS = 500
RESULT_CHARS = 500
RECENT_STEPS = 4
PENDING = "[pending]"
SAMPLE_ROWS = 100

# Shared with the local Hermes vocabulary. `respond` is the no-tool label.
KEEP_TOOLS = frozenset(
    {
        "terminal",
        "read_file",
        "write_file",
        "patch",
        "execute_code",
        "search_files",
        "web_search",
        "todo",
        "memory",
        "skill_view",
        "session_search",
        "process",
        "delegate_task",
    }
)

# Filled only after the dataset tool schema shows a renamed kept tool.
# Keys are names as they appear in tool_call JSON. Values are KEEP_TOOLS names.
# Inspect with: python3 scripts/curate_traces.py --inspect-only
ALIASES: dict[str, str] = {}

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESP_OPEN = "<tool_response>"
TOOL_RESP_CLOSE = "</tool_response>"

REPO_ROOT = Path(__file__).resolve().parents[1]
REJECTED_VOLUME_NAMES = {"Macintosh HD", "SanDisk Unlocker"}


def truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap]


def stringify(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def resolve_data_root(override: str | None) -> Path:
    raw = override or os.environ.get("NCA_DATA_ROOT") or str(DEFAULT_DATA_ROOT)
    root = Path(raw)
    if not root.is_absolute():
        raise SystemExit(f"NCA_DATA_ROOT must be absolute, got {raw}")
    parts = root.parts
    if len(parts) < 3 or parts[1] != "Volumes" or parts[2] in REJECTED_VOLUME_NAMES:
        raise SystemExit(
            "Refusing to write datasets on the internal disk. "
            f"NCA_DATA_ROOT must be /Volumes/<external>/..., got {root}"
        )
    if not Path("/Volumes", parts[2]).exists():
        raise SystemExit(f"External volume is not mounted: /Volumes/{parts[2]}")
    resolved = root.resolve()
    home = Path.home().resolve()
    if resolved == home or home in resolved.parents:
        raise SystemExit(f"Refusing to write datasets under the home directory: {resolved}")
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit(f"Refusing to write datasets inside the repo: {resolved}")
    return root


def configure_hf(data_root: Path) -> Path:
    hf_home = data_root / "datasets" / "hf_home"
    datasets_cache = data_root / "datasets" / "hf_datasets_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
    return hf_home


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_filtered(hf_home: Path) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=FILTERED_REPO,
            filename=FILTERED_FILENAME,
            repo_type="dataset",
            revision=FILTERED_REVISION,
            cache_dir=str(hf_home / "hub"),
        )
    except Exception as exc:
        raise SystemExit(
            "Download of DJLougen/hermes-agent-traces-filtered failed. "
            "No lambda fallback was written, because that corpus is parquet "
            f"and this environment has no pyarrow/pandas. Error: {exc}"
        ) from exc
    downloaded = Path(path)
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise SystemExit(f"Filtered dataset file is missing or empty: {downloaded}")
    return downloaded


def parse_tool_defs(tools_field: object) -> list[tuple[str, str]]:
    if isinstance(tools_field, str):
        try:
            tools_field = json.loads(tools_field)
        except json.JSONDecodeError:
            return []
    if not isinstance(tools_field, list):
        return []
    found: list[tuple[str, str]] = []
    for item in tools_field:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        body = function if isinstance(function, dict) else item
        name = body.get("name")
        description = body.get("description") or ""
        if isinstance(name, str) and name:
            found.append((name, description if isinstance(description, str) else ""))
    return found


def extract_tool_calls(text: str) -> list[str | None]:
    """One entry per tool_call block. None means the block did not parse.

    Unparsed blocks are kept so later tool_response tags stay aligned.
    """
    names: list[str | None] = []
    start = 0
    while True:
        open_at = text.find(TOOL_CALL_OPEN, start)
        if open_at < 0:
            break
        close_at = text.find(TOOL_CALL_CLOSE, open_at + len(TOOL_CALL_OPEN))
        if close_at < 0:
            names.append(None)
            break
        blob = text[open_at + len(TOOL_CALL_OPEN) : close_at].strip()
        start = close_at + len(TOOL_CALL_CLOSE)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            names.append(None)
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                names.append(None)
                continue
            name = item.get("name")
            if not isinstance(name, str):
                function = item.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
            if not isinstance(name, str) or not name:
                names.append(None)
                continue
            names.append(name)
    return names


def extract_tool_results(text: str) -> tuple[list[str], int]:
    results: list[str] = []
    errors = 0
    start = 0
    while True:
        open_at = text.find(TOOL_RESP_OPEN, start)
        if open_at < 0:
            break
        close_at = text.find(TOOL_RESP_CLOSE, open_at + len(TOOL_RESP_OPEN))
        if close_at < 0:
            errors += 1
            break
        blob = text[open_at + len(TOOL_RESP_OPEN) : close_at].strip()
        start = close_at + len(TOOL_RESP_CLOSE)
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            results.append(truncate(blob, RESULT_CHARS))
            continue
        if isinstance(payload, dict) and "content" in payload:
            results.append(truncate(stringify(payload["content"]), RESULT_CHARS))
        else:
            results.append(truncate(stringify(payload), RESULT_CHARS))
    return results, errors


def classify(raw_name: str) -> tuple[str, str | None]:
    """Return ('browser'|'unmapped'|'keep', canonical name or None)."""
    if raw_name.startswith("browser_"):
        return "browser", None
    canonical = ALIASES.get(raw_name, raw_name)
    if canonical != raw_name:
        if canonical not in KEEP_TOOLS:
            raise RuntimeError(f"Alias {raw_name!r} -> {canonical!r} is not a kept tool")
        return "keep", canonical
    if raw_name in KEEP_TOOLS:
        return "keep", raw_name
    return "unmapped", None


def split_is_val(session_id: str) -> bool:
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def iter_sessions(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"_parse_error": str(exc)}
                continue
            if not isinstance(row, dict):
                yield line_no, {"_parse_error": "row is not an object"}
                continue
            yield line_no, row


def role_of(message: dict) -> str:
    raw = message.get("from") or message.get("role") or ""
    if not isinstance(raw, str):
        return ""
    return raw.strip().lower()


def message_text(message: dict) -> str:
    value = message.get("value")
    if value is None:
        value = message.get("content")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return stringify(value)


def snapshot(history: list[dict[str, str]]) -> list[dict[str, str]]:
    tail = history[-RECENT_STEPS:]
    return [{"tool": step["tool"], "result": step["result"]} for step in tail]


def inspect_tools(source: Path, report_path: Path) -> dict:
    schema_counts: Counter[str] = Counter()
    schema_desc: dict[str, str] = {}
    call_counts: Counter[str] = Counter()
    sessions = 0
    parse_errors = 0
    for line_no, row in iter_sessions(source):
        if "_parse_error" in row:
            parse_errors += 1
            print(f"bad json line {line_no}: {row['_parse_error']}", file=sys.stderr)
            continue
        sessions += 1
        for name, description in parse_tool_defs(row.get("tools")):
            schema_counts[name] += 1
            schema_desc.setdefault(name, truncate(description.replace("\n", " "), 240))
        for message in row.get("conversations") or []:
            if not isinstance(message, dict):
                continue
            if role_of(message) not in {"gpt", "assistant"}:
                continue
            for name in extract_tool_calls(message_text(message)):
                if name:
                    call_counts[name] += 1
        if sessions % 500 == 0:
            print(f"inspected {sessions} sessions", file=sys.stderr)
    report = {
        "sessions": sessions,
        "parse_errors": parse_errors,
        "schema_tools": [
            {
                "name": name,
                "sessions_listing_tool": schema_counts[name],
                "description": schema_desc.get(name, ""),
                "call_count": call_counts.get(name, 0),
            }
            for name in sorted(schema_counts)
        ],
        "call_names_absent_from_schema": [
            {"name": name, "call_count": count}
            for name, count in sorted(call_counts.items())
            if name not in schema_counts
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def curate(source: Path, train_path: Path, val_path: Path) -> dict:
    train_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict = {
        "source_sessions": 0,
        "source_parse_errors": 0,
        "sessions_with_rows": 0,
        "sessions_dropped_no_kept_decision": 0,
        "rows_train": 0,
        "rows_val": 0,
        "sessions_train": 0,
        "sessions_val": 0,
        "dropped_browser_calls": 0,
        "dropped_unmapped_calls": 0,
        "unparsed_tool_calls": 0,
        "unparsed_tool_responses": 0,
        "orphan_tool_responses": 0,
        "assistant_turns_missing_tool_message": 0,
        "alias_applications": Counter(),
        "excluded_calls": Counter(),
        "tools_train": Counter(),
        "tools_val": Counter(),
        "unexpected_roles": Counter(),
        "local_hermes_sessions_included": 0,
    }
    train_sessions: set[str] = set()
    val_sessions: set[str] = set()
    step_index: dict[str, int] = {}
    sample: list[str] = []

    with train_path.open("w", encoding="utf-8") as train_out, val_path.open("w", encoding="utf-8") as val_out:
        for line_no, row in iter_sessions(source):
            if "_parse_error" in row:
                stats["source_parse_errors"] += 1
                print(f"bad json line {line_no}: {row['_parse_error']}", file=sys.stderr)
                continue
            stats["source_sessions"] += 1
            session_id = row.get("id")
            if not isinstance(session_id, str) or not session_id:
                stats["source_parse_errors"] += 1
                continue
            is_val = split_is_val(session_id)
            out = val_out if is_val else train_out
            split_name = "val" if is_val else "train"
            goal = ""
            history: list[dict[str, str]] = []
            pending: list[int | None] = []
            emitted = 0

            def emit(target: str) -> None:
                nonlocal emitted
                index = step_index.get(session_id, 0)
                step_index[session_id] = index + 1
                record = {
                    "session_id": session_id,
                    "step_index": index,
                    "goal": goal,
                    "recent_steps": snapshot(history),
                    "target_tool": target,
                }
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                out.write(line + "\n")
                emitted += 1
                if split_name == "train":
                    stats["rows_train"] += 1
                    stats["tools_train"][target] += 1
                    if len(sample) < SAMPLE_ROWS:
                        sample.append(line)
                else:
                    stats["rows_val"] += 1
                    stats["tools_val"][target] += 1

            conversations = row.get("conversations") or []
            if not isinstance(conversations, list):
                conversations = []
            for message in conversations:
                if not isinstance(message, dict):
                    continue
                role = role_of(message)
                text = message_text(message)
                if role in {"human", "user"}:
                    goal = truncate(text.strip(), GOAL_CHARS)
                    continue
                if role in {"system"}:
                    continue
                if role in {"tool"}:
                    results, resp_errors = extract_tool_results(text)
                    stats["unparsed_tool_responses"] += resp_errors
                    for result in results:
                        if not pending:
                            stats["orphan_tool_responses"] += 1
                            continue
                        slot = pending.pop(0)
                        if slot is None:
                            continue
                        history[slot]["result"] = result
                    if pending:
                        stats["assistant_turns_missing_tool_message"] += 1
                        pending.clear()
                    continue
                if role not in {"gpt", "assistant"}:
                    if role:
                        stats["unexpected_roles"][role] += 1
                    continue
                if pending:
                    stats["assistant_turns_missing_tool_message"] += 1
                    pending.clear()
                names = extract_tool_calls(text)
                if not names:
                    if TOOL_CALL_OPEN in text:
                        stats["unparsed_tool_calls"] += 1
                        continue
                    emit("respond")
                    continue
                for raw_name in names:
                    if raw_name is None:
                        stats["unparsed_tool_calls"] += 1
                        pending.append(None)
                        continue
                    kind, canonical = classify(raw_name)
                    if kind == "browser":
                        stats["dropped_browser_calls"] += 1
                        stats["excluded_calls"][raw_name] += 1
                        pending.append(None)
                        continue
                    if kind == "unmapped":
                        stats["dropped_unmapped_calls"] += 1
                        stats["excluded_calls"][raw_name] += 1
                        pending.append(None)
                        continue
                    assert canonical is not None
                    if raw_name in ALIASES:
                        stats["alias_applications"][f"{raw_name}->{canonical}"] += 1
                    emit(canonical)
                    history.append({"tool": canonical, "result": PENDING})
                    pending.append(len(history) - 1)

            if pending:
                stats["assistant_turns_missing_tool_message"] += 1
                pending.clear()
            if emitted:
                stats["sessions_with_rows"] += 1
                bucket = val_sessions if is_val else train_sessions
                bucket.add(session_id)
            else:
                stats["sessions_dropped_no_kept_decision"] += 1
            if stats["source_sessions"] % 500 == 0:
                print(
                    f"curated {stats['source_sessions']} sessions "
                    f"({stats['rows_train'] + stats['rows_val']} rows)",
                    file=sys.stderr,
                )

    overlap = train_sessions & val_sessions
    if overlap:
        raise SystemExit(f"Session overlap between train and val: {len(overlap)}")
    stats["sessions_train"] = len(train_sessions)
    stats["sessions_val"] = len(val_sessions)
    stats["sample_lines"] = sample
    return stats


def counter_dict(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def tool_histogram(counter: Counter) -> dict[str, int]:
    filled = {name: 0 for name in sorted(KEEP_TOOLS | {"respond"})}
    filled.update(counter_dict(counter))
    return filled


def build_stats_payload(
    source_path: Path,
    source_sha256: str,
    train_path: Path,
    val_path: Path,
    raw_stats: dict,
) -> dict:
    sample_lines = raw_stats.pop("sample_lines")
    return {
        "source": {
            "dataset": FILTERED_REPO,
            "revision": FILTERED_REVISION,
            "filename": FILTERED_FILENAME,
            "file_sha256": source_sha256,
            "file_bytes": source_path.stat().st_size,
            "upstream": UPSTREAM_REPO,
            "upstream_note": (
                "Filtered Kimi slice of lambda/hermes-agent-reasoning-traces. "
                "GLM-5.1 trajectories are not in this slice."
            ),
            "license": "apache-2.0",
            "local_path": str(source_path),
        },
        "filters": {
            "kept_tools": sorted(KEEP_TOOLS),
            "respond_label": "respond",
            "dropped": "tool names starting with browser_, plus any name outside the kept set",
            "aliases": dict(ALIASES),
            "goal_chars": GOAL_CHARS,
            "result_chars": RESULT_CHARS,
            "recent_steps": RECENT_STEPS,
            "pending_result": PENDING,
            "split": "sha256(session_id) first 8 bytes % 10 == 0 is val, otherwise train",
            "local_hermes_in_train": False,
        },
        "counts": {
            "source_sessions": raw_stats["source_sessions"],
            "source_parse_errors": raw_stats["source_parse_errors"],
            "sessions_with_rows": raw_stats["sessions_with_rows"],
            "sessions_dropped_no_kept_decision": raw_stats["sessions_dropped_no_kept_decision"],
            "sessions_train": raw_stats["sessions_train"],
            "sessions_val": raw_stats["sessions_val"],
            "rows_train": raw_stats["rows_train"],
            "rows_val": raw_stats["rows_val"],
            "dropped_browser_calls": raw_stats["dropped_browser_calls"],
            "dropped_unmapped_calls": raw_stats["dropped_unmapped_calls"],
            "unparsed_tool_calls": raw_stats["unparsed_tool_calls"],
            "unparsed_tool_responses": raw_stats["unparsed_tool_responses"],
            "orphan_tool_responses": raw_stats["orphan_tool_responses"],
            "assistant_turns_missing_tool_message": raw_stats["assistant_turns_missing_tool_message"],
            "local_hermes_sessions_included": raw_stats["local_hermes_sessions_included"],
        },
        "alias_applications": counter_dict(raw_stats["alias_applications"]),
        "excluded_calls": counter_dict(raw_stats["excluded_calls"]),
        "tools_train": tool_histogram(raw_stats["tools_train"]),
        "tools_val": tool_histogram(raw_stats["tools_val"]),
        "unexpected_roles": counter_dict(raw_stats["unexpected_roles"]),
        "outputs": {
            "train": str(train_path),
            "val": str(val_path),
            "train_bytes": train_path.stat().st_size,
            "val_bytes": val_path.stat().st_size,
        },
        "_sample_lines": sample_lines,
    }


def write_repo_artifacts(payload: dict, results_path: Path) -> None:
    sample_lines: list[str] = payload.pop("_sample_lines")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(text, encoding="utf-8")
    repo_dir = REPO_ROOT / "data" / "curated"
    repo_dir.mkdir(parents=True, exist_ok=True)
    stats_path = repo_dir / "stats.json"
    sample_path = repo_dir / "sample.jsonl"
    stats_path.write_text(text, encoding="utf-8")
    sample_path.write_text("\n".join(sample_lines) + ("\n" if sample_lines else ""), encoding="utf-8")
    sample_bytes = sample_path.stat().st_size
    if sample_bytes >= 1_000_000:
        sample_path.unlink()
        raise SystemExit(f"Sample is {sample_bytes} bytes, not under 1MB. Not written.")
    print(f"stats {stats_path} ({stats_path.stat().st_size} bytes)", file=sys.stderr)
    print(f"sample {sample_path} ({sample_bytes} bytes, {len(sample_lines)} rows)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Overrides NCA_DATA_ROOT")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Download and write a tool-name report. Do not write train/val jsonl.",
    )
    args = parser.parse_args()
    data_root = resolve_data_root(args.data_root)
    (data_root / "datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "results").mkdir(parents=True, exist_ok=True)
    hf_home = configure_hf(data_root)
    print(f"data root {data_root}", file=sys.stderr)
    print(f"HF_HOME {os.environ['HF_HOME']}", file=sys.stderr)
    source = download_filtered(hf_home)
    print(f"source {source} ({source.stat().st_size} bytes)", file=sys.stderr)
    if args.inspect_only:
        report_path = data_root / "results" / "tool-schema-scan.json"
        report = inspect_tools(source, report_path)
        print(
            f"inspected {report['sessions']} sessions, "
            f"{len(report['schema_tools'])} schema tools -> {report_path}",
            file=sys.stderr,
        )
        return
    train_path = data_root / "datasets" / "curated" / "train.jsonl"
    val_path = data_root / "datasets" / "curated" / "val.jsonl"
    raw_stats = curate(source, train_path, val_path)
    source_hash = sha256_file(source)
    payload = build_stats_payload(source, source_hash, train_path, val_path, raw_stats)
    write_repo_artifacts(payload, data_root / "results" / "curated-stats.json")
    print(
        f"rows train={payload['counts']['rows_train']} val={payload['counts']['rows_val']} "
        f"sessions train={payload['counts']['sessions_train']} val={payload['counts']['sessions_val']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
