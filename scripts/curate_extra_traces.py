#!/usr/bin/env python3
"""Curate extra next-tool rows that are not already in the Kimi filtered slice.

Reads lambda/hermes-agent-reasoning-traces (Apache-2.0) at a pinned revision:
the Kimi config with DJLougen filtered ids removed, and the full GLM-5.1 config.
Writes the same row schema as scripts/curate_traces.py. Does not open ~/.hermes
and does not write datasets/curated/train.jsonl or val.jsonl.

Rerun from the repo root, with the external data volume mounted:

    python3 scripts/curate_extra_traces.py
    NCA_DATA_ROOT=/Volumes/Other/nca_research python3 scripts/curate_extra_traces.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import curate_traces as base

LAMBDA_REPO = "lambda/hermes-agent-reasoning-traces"
LAMBDA_REVISION = "b92885e4f0161d4b2536512710e004d4892cac6e"
LAMBDA_LICENSE = "apache-2.0"
KIMI_FILENAME = "data/kimi/train.parquet"
GLM_FILENAME = "data/glm-5.1/train.parquet"

# Already curated by scripts/curate_traces.py. Ids from this file are skipped.
FILTERED_REPO = base.FILTERED_REPO
FILTERED_REVISION = base.FILTERED_REVISION
FILTERED_FILENAME = base.FILTERED_FILENAME

SAMPLE_ROWS = base.SAMPLE_ROWS


def extra_root(data_root: Path) -> Path:
    return data_root / "datasets" / "extra"


def assert_not_curated(path: Path, data_root: Path) -> None:
    curated = (data_root / "datasets" / "curated").resolve()
    resolved = path.resolve()
    if resolved == curated or curated in resolved.parents:
        raise SystemExit(f"Refusing to write into the existing curated split: {resolved}")


def ensure_pyarrow(data_root: Path) -> None:
    pydeps = extra_root(data_root) / "pydeps"
    pydeps.mkdir(parents=True, exist_ok=True)
    deps = str(pydeps)
    if deps not in sys.path:
        sys.path.insert(0, deps)
    try:
        import pyarrow  # noqa: F401

        return
    except ImportError:
        if os.environ.get("NCA_PYARROW_REEXEC") == "1":
            raise SystemExit(f"pyarrow is installed under {pydeps} but still will not import")
    cache = extra_root(data_root) / "pip-cache"
    tmp = extra_root(data_root) / "tmp"
    cache.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(cache)
    env["TMPDIR"] = str(tmp)
    print(f"installing pyarrow into {pydeps}", file=sys.stderr)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "--target", deps, "pyarrow"],
        env=env,
    )
    # A failed import in this process caches the miss. Restart once.
    os.environ["NCA_PYARROW_REEXEC"] = "1"
    prior = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = deps if not prior else deps + os.pathsep + prior
    os.execv(sys.executable, [sys.executable, *sys.argv])


def download(repo: str, filename: str, revision: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache),
        )
    except Exception as exc:
        raise SystemExit(
            f"Download failed for {repo}@{revision} file {filename}: {exc}"
        ) from exc
    downloaded = Path(path)
    if not downloaded.is_file() or downloaded.stat().st_size == 0:
        raise SystemExit(f"Downloaded file is missing or empty: {downloaded}")
    return downloaded


def load_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for line_no, row in base.iter_sessions(path):
        if "_parse_error" in row:
            raise SystemExit(f"Filtered id file has bad JSON on line {line_no}: {row['_parse_error']}")
        session_id = row.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise SystemExit(f"Filtered id file line {line_no} has no string id")
        ids.add(session_id)
    return ids


def jsonable(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def normalize_session(row: dict) -> dict:
    session_id = row.get("id")
    if isinstance(session_id, bytes):
        session_id = session_id.decode("utf-8")
    conversations = jsonable(row.get("conversations"))
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    if not isinstance(conversations, list):
        conversations = []
    return {"id": session_id, "conversations": conversations}


def iter_parquet_sessions(path: Path) -> Iterator[dict]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=16, columns=["id", "conversations"]):
        for row in batch.to_pylist():
            yield normalize_session(row)


def read_parquet_ids(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(path).read(columns=["id"])
    ids: list[str] = []
    for value in table.column("id").to_pylist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        ids.append(value if isinstance(value, str) else "")
    return ids


def write_kept_jsonl(source: Path, dest: Path, skip_ids: set[str]) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    missing_id = 0
    seen: set[str] = set()
    duplicate_ids = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in iter_parquet_sessions(source):
            session_id = row["id"]
            if not isinstance(session_id, str) or not session_id:
                missing_id += 1
                continue
            if session_id in skip_ids:
                skipped += 1
                continue
            if session_id in seen:
                duplicate_ids += 1
                continue
            seen.add(session_id)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
            if kept % 500 == 0:
                print(f"wrote {kept} kept sessions from {source.name}", file=sys.stderr)
    return {
        "kept_sessions": kept,
        "skipped_already_curated": skipped,
        "missing_id": missing_id,
        "duplicate_ids_skipped": duplicate_ids,
        "bytes": dest.stat().st_size,
        "path": str(dest),
    }


def concat_files(paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as out:
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")


def session_ids_in_rows(path: Path) -> set[str]:
    found: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            found.add(json.loads(line)["session_id"])
    return found


def counter_dict(counter: Counter) -> dict[str, int]:
    return base.counter_dict(counter)


def summarize_source(name: str, raw: dict, train_path: Path, val_path: Path) -> dict:
    return {
        "name": name,
        "counts": {
            "source_sessions": raw["source_sessions"],
            "source_parse_errors": raw["source_parse_errors"],
            "sessions_with_rows": raw["sessions_with_rows"],
            "sessions_dropped_no_kept_decision": raw["sessions_dropped_no_kept_decision"],
            "sessions_train": raw["sessions_train"],
            "sessions_val": raw["sessions_val"],
            "rows_train": raw["rows_train"],
            "rows_val": raw["rows_val"],
            "dropped_browser_calls": raw["dropped_browser_calls"],
            "dropped_unmapped_calls": raw["dropped_unmapped_calls"],
            "unparsed_tool_calls": raw["unparsed_tool_calls"],
            "unparsed_tool_responses": raw["unparsed_tool_responses"],
            "orphan_tool_responses": raw["orphan_tool_responses"],
            "assistant_turns_missing_tool_message": raw["assistant_turns_missing_tool_message"],
            "local_hermes_sessions_included": raw["local_hermes_sessions_included"],
        },
        "excluded_calls": counter_dict(raw["excluded_calls"]),
        "tools_train": base.tool_histogram(raw["tools_train"]),
        "tools_val": base.tool_histogram(raw["tools_val"]),
        "unexpected_roles": counter_dict(raw["unexpected_roles"]),
        "outputs": {
            "train": str(train_path),
            "val": str(val_path),
            "train_bytes": train_path.stat().st_size,
            "val_bytes": val_path.stat().st_size,
            "train_sha256": base.sha256_file(train_path),
            "val_sha256": base.sha256_file(val_path),
        },
    }


def curated_overlap(data_root: Path, extra_ids: set[str]) -> dict:
    curated_dir = data_root / "datasets" / "curated"
    report = {"checked": [], "overlap": 0}
    for name in ("train.jsonl", "val.jsonl"):
        path = curated_dir / name
        if not path.is_file():
            report["checked"].append({"file": str(path), "present": False})
            continue
        overlap: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                session_id = json.loads(line)["session_id"]
                if session_id in extra_ids:
                    overlap.add(session_id)
        report["checked"].append(
            {"file": str(path), "present": True, "overlapping_sessions": len(overlap)}
        )
        report["overlap"] += len(overlap)
    return report


def write_repo_artifacts(payload: dict, sample_lines: list[str], results_path: Path) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(text, encoding="utf-8")
    repo_dir = base.REPO_ROOT / "data" / "extra"
    repo_dir.mkdir(parents=True, exist_ok=True)
    stats_path = repo_dir / "stats.json"
    sample_path = repo_dir / "sample.jsonl"
    stats_path.write_text(text, encoding="utf-8")
    sample_body = "\n".join(sample_lines) + ("\n" if sample_lines else "")
    if len(sample_body.encode("utf-8")) >= 1_000_000:
        raise SystemExit("Sample would be at least 1MB. Not written.")
    sample_path.write_text(sample_body, encoding="utf-8")
    sample_bytes = sample_path.stat().st_size
    if sample_bytes >= 1_000_000:
        sample_path.unlink()
        raise SystemExit(f"Sample is {sample_bytes} bytes, not under 1MB. Not written.")
    print(f"stats {stats_path} ({stats_path.stat().st_size} bytes)", file=sys.stderr)
    print(f"sample {sample_path} ({sample_bytes} bytes, {len(sample_lines)} rows)", file=sys.stderr)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Overrides NCA_DATA_ROOT")
    args = parser.parse_args()
    data_root = base.resolve_data_root(args.data_root)
    (data_root / "datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "results").mkdir(parents=True, exist_ok=True)
    hf_home = base.configure_hf(data_root)
    print(f"data root {data_root}", file=sys.stderr)
    print(f"HF_HOME {os.environ['HF_HOME']}", file=sys.stderr)
    print(f"HF_DATASETS_CACHE {os.environ['HF_DATASETS_CACHE']}", file=sys.stderr)
    ensure_pyarrow(data_root)

    cache = hf_home / "hub"
    filtered_path = download(FILTERED_REPO, FILTERED_FILENAME, FILTERED_REVISION, cache)
    filtered_ids = load_ids(filtered_path)
    print(f"filtered ids {len(filtered_ids)} from {filtered_path}", file=sys.stderr)

    kimi_path = download(LAMBDA_REPO, KIMI_FILENAME, LAMBDA_REVISION, cache)
    glm_path = download(LAMBDA_REPO, GLM_FILENAME, LAMBDA_REVISION, cache)
    print(f"kimi {kimi_path} ({kimi_path.stat().st_size} bytes)", file=sys.stderr)
    print(f"glm {glm_path} ({glm_path.stat().st_size} bytes)", file=sys.stderr)

    kimi_ids = read_parquet_ids(kimi_path)
    glm_ids = read_parquet_ids(glm_path)
    kimi_id_set = {item for item in kimi_ids if item}
    glm_id_set = {item for item in glm_ids if item}
    filtered_in_kimi = filtered_ids & kimi_id_set
    filtered_in_glm = filtered_ids & glm_id_set
    kept_kimi_ids = kimi_id_set - filtered_ids
    kept_glm_ids = glm_id_set - filtered_ids
    cross = kept_kimi_ids & kept_glm_ids
    print(
        f"kimi rows {len(kimi_ids)} unique {len(kimi_id_set)} "
        f"filtered overlap {len(filtered_in_kimi)} keep {len(kept_kimi_ids)}",
        file=sys.stderr,
    )
    print(
        f"glm rows {len(glm_ids)} unique {len(glm_id_set)} "
        f"filtered overlap {len(filtered_in_glm)} keep {len(kept_glm_ids)}",
        file=sys.stderr,
    )
    if cross:
        raise SystemExit(
            f"{len(cross)} session ids are in both kept slices. "
            "Refusing to emit colliding session ids."
        )
    missing_filtered = filtered_ids - kimi_id_set - glm_id_set
    if missing_filtered:
        print(
            f"warning: {len(missing_filtered)} filtered ids are in neither parquet",
            file=sys.stderr,
        )

    root = extra_root(data_root)
    raw_dir = root / "raw"
    slices = [
        ("lambda-kimi-remainder", kimi_path, raw_dir / "lambda-kimi-remainder.jsonl"),
        ("lambda-glm-5.1", glm_path, raw_dir / "lambda-glm-5.1.jsonl"),
    ]
    slice_reports = []
    summaries = []
    sample: list[str] = []
    train_parts: list[Path] = []
    val_parts: list[Path] = []
    for name, parquet_path, raw_path in slices:
        assert_not_curated(raw_path, data_root)
        print(f"selecting {name}", file=sys.stderr)
        selection = write_kept_jsonl(parquet_path, raw_path, filtered_ids)
        train_path = root / name / "train.jsonl"
        val_path = root / name / "val.jsonl"
        assert_not_curated(train_path, data_root)
        assert_not_curated(val_path, data_root)
        raw_stats = base.curate(raw_path, train_path, val_path)
        for line in raw_stats["sample_lines"]:
            if len(sample) < SAMPLE_ROWS:
                sample.append(line)
        summary = summarize_source(name, raw_stats, train_path, val_path)
        summary["selection"] = selection
        summary["source_parquet"] = {
            "dataset": LAMBDA_REPO,
            "revision": LAMBDA_REVISION,
            "license": LAMBDA_LICENSE,
            "filename": KIMI_FILENAME if name.endswith("remainder") else GLM_FILENAME,
            "bytes": parquet_path.stat().st_size,
            "sha256": base.sha256_file(parquet_path),
            "local_path": str(parquet_path),
        }
        summaries.append(summary)
        slice_reports.append(selection)
        train_parts.append(train_path)
        val_parts.append(val_path)
        print(
            f"{name} rows train={summary['counts']['rows_train']} "
            f"val={summary['counts']['rows_val']}",
            file=sys.stderr,
        )

    combined_train = root / "train.jsonl"
    combined_val = root / "val.jsonl"
    assert_not_curated(combined_train, data_root)
    assert_not_curated(combined_val, data_root)
    concat_files(train_parts, combined_train)
    concat_files(val_parts, combined_val)
    extra_ids = session_ids_in_rows(combined_train) | session_ids_in_rows(combined_val)
    overlap = curated_overlap(data_root, extra_ids)
    if overlap["overlap"]:
        raise SystemExit(f"Extra sessions overlap the curated split: {overlap}")
    per_source_ids = []
    for train_path, val_path in zip(train_parts, val_parts):
        per_source_ids.append(session_ids_in_rows(train_path) | session_ids_in_rows(val_path))
    if per_source_ids[0] & per_source_ids[1]:
        raise SystemExit("Kimi remainder and GLM share a kept session id")

    payload = {
        "filters": {
            "kept_tools": sorted(base.KEEP_TOOLS),
            "respond_label": "respond",
            "dropped": "tool names starting with browser_, plus any name outside the kept set",
            "aliases": dict(base.ALIASES),
            "goal_chars": base.GOAL_CHARS,
            "result_chars": base.RESULT_CHARS,
            "recent_steps": base.RECENT_STEPS,
            "pending_result": base.PENDING,
            "split": "sha256(session_id) first 8 bytes % 10 == 0 is val, otherwise train",
            "already_curated_excluded": {
                "dataset": FILTERED_REPO,
                "revision": FILTERED_REVISION,
                "filename": FILTERED_FILENAME,
                "ids": len(filtered_ids),
                "ids_found_in_kimi": len(filtered_in_kimi),
                "ids_found_in_glm": len(filtered_in_glm),
                "ids_missing_from_both_parquets": len(missing_filtered),
            },
            "local_hermes_in_train": False,
            "browser_star_dropped": True,
        },
        "id_audit": {
            "kimi_rows": len(kimi_ids),
            "kimi_unique_ids": len(kimi_id_set),
            "glm_rows": len(glm_ids),
            "glm_unique_ids": len(glm_id_set),
            "kept_id_intersection": len(cross),
            "curated_overlap": overlap,
        },
        "sources": summaries,
        "combined": {
            "train": str(combined_train),
            "val": str(combined_val),
            "train_bytes": combined_train.stat().st_size,
            "val_bytes": combined_val.stat().st_size,
            "train_sha256": base.sha256_file(combined_train),
            "val_sha256": base.sha256_file(combined_val),
            "rows_train": sum(item["counts"]["rows_train"] for item in summaries),
            "rows_val": sum(item["counts"]["rows_val"] for item in summaries),
            "sessions_train": sum(item["counts"]["sessions_train"] for item in summaries),
            "sessions_val": sum(item["counts"]["sessions_val"] for item in summaries),
        },
    }
    write_repo_artifacts(payload, sample, data_root / "results" / "extra-stats.json")
    print(
        f"combined rows train={payload['combined']['rows_train']} "
        f"val={payload['combined']['rows_val']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
