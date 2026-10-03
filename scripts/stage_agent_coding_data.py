#!/usr/bin/env python3
"""Stage multi-step coding-agent trajectories onto the external SSD.

Downloads four public trajectory sets whose actions are shell commands or
file edits, pins the Hub revision, and writes measured row counts. Hugging
Face caches stay under the data root. This script does not touch
datasets/extra, datasets/curated, or the SWE-bench test set.

Rerun from the repo root:

    python3 scripts/stage_agent_coding_data.py
    NCA_DATA_ROOT=/Volumes/Other/nca_research python3 scripts/stage_agent_coding_data.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/Volumes/Extreme SSD/nca_research")
REPO_ROOT = Path(__file__).resolve().parents[1]
REJECTED_VOLUME_NAMES = {"Macintosh HD", "SanDisk Unlocker"}

# Revisions are the Hub commit SHAs read from the dataset API on 2026-10-03.
STAGED = [
    {
        "repo": "SWE-Gym/OpenHands-SFT-Trajectories",
        "revision": "4aaa5a4a4b5861f4799d2336908760c190ac3b17",
        "license": "mit",
        "kind": "coding",
        "dirname": "openhands-sft",
        "allow_patterns": None,
        "shape": "multi-step",
        "parser": "xml_function",
    },
    {
        "repo": "SWE-bench/SWE-smith-trajectories",
        "revision": "08e109b4a59eaeebf80e4675cd125d42e7ac99a4",
        "license": "mit",
        "kind": "coding",
        "dirname": "swe-smith-trajectories-tool",
        "allow_patterns": ["data/tool-*.parquet", "README.md", ".gitattributes"],
        "shape": "multi-step",
        "parser": "smith_tool_calls",
    },
    {
        "repo": "nebius/SWE-rebench-openhands-trajectories",
        "revision": "35455389ab51bf5e2306bfd436ef72d0f98bf882",
        "license": "cc-by-4.0",
        "kind": "coding",
        "dirname": "swe-rebench-openhands",
        "allow_patterns": None,
        "shape": "multi-step",
        "parser": "openhands_tool_calls",
    },
    {
        # The trajectory card omits `license`. This label follows the Apache-2.0
        # LICENSE in github.com/R2E-Gym/R2E-Gym and the apache-2.0 cards on
        # R2E-Gym/R2E-Gym-Subset and R2E-Gym/R2E-Gym-V1.
        "repo": "R2E-Gym/R2EGym-SFT-Trajectories",
        "revision": "63ab4eb37668f8be0104133c21d896bedbcf8404",
        "license": "apache-2.0",
        "kind": "coding",
        "dirname": "r2e-gym-sft",
        "allow_patterns": None,
        "shape": "multi-step",
        "parser": "xml_function",
    },
]

BLOCK_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=command>([^<]*)</parameter>")


def resolve_data_root() -> Path:
    raw = os.environ.get("NCA_DATA_ROOT") or str(DEFAULT_DATA_ROOT)
    root = Path(raw)
    if not root.is_absolute():
        raise SystemExit(f"NCA_DATA_ROOT must be absolute, got {raw}")
    parts = root.parts
    if len(parts) < 3 or parts[1] != "Volumes" or parts[2] in REJECTED_VOLUME_NAMES:
        raise SystemExit(
            "Refusing to write datasets on the internal disk. "
            f"NCA_DATA_ROOT must be /Volumes/<external>/..., got {root}"
        )
    volume = Path("/Volumes", parts[2])
    if not volume.exists():
        raise SystemExit(f"External volume is not mounted: {volume}")
    resolved = root.resolve()
    home = Path.home().resolve()
    if resolved == home or home in resolved.parents:
        raise SystemExit(f"Refusing to write datasets under the home directory: {resolved}")
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise SystemExit(f"Refusing to write datasets inside the repo: {resolved}")
    return root


def configure_hf(data_root: Path) -> None:
    hf_home = data_root / "hf-cache"
    datasets_cache = hf_home / "datasets"
    tmp = data_root / "tmp"
    for path in (hf_home / "hub", datasets_cache, tmp):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)


def dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if not item.is_file() or item.name.startswith("._"):
            continue
        if ".cache" in item.parts:
            continue
        total += item.stat().st_size
    return total


def parquet_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.parquet") if not p.name.startswith("._"))


def load_pyarrow(data_root: Path):
    tools = data_root / "py-tools"
    if tools.exists():
        sys.path.insert(0, str(tools))
    import pyarrow.parquet as pq

    return pq


def parse_json_maybe(value):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def note_call(counter: Counter, name: str | None, arguments) -> None:
    if not name:
        return
    arguments = parse_json_maybe(arguments)
    command = None
    if isinstance(arguments, dict):
        command = arguments.get("command")
    if name in {"str_replace_editor", "file_editor"} and isinstance(command, str) and command:
        counter[f"{name}:{command}"] += 1
    else:
        counter[name] += 1


def count_xml(text: str, counter: Counter) -> None:
    if not text or "<function=" not in text:
        return
    for name, body in BLOCK_RE.findall(text):
        command = None
        if name in {"str_replace_editor", "file_editor"}:
            match = PARAM_RE.search(body)
            if match:
                command = match.group(1).strip()
        if command:
            counter[f"{name}:{command}"] += 1
        else:
            counter[name] += 1


def count_tool_calls(calls, counter: Counter) -> None:
    calls = parse_json_maybe(calls)
    if not isinstance(calls, list):
        return
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        note_call(counter, function.get("name") or call.get("name"), function.get("arguments"))


def scan_messages(messages, counter: Counter, parser: str) -> None:
    messages = parse_json_maybe(messages)
    if isinstance(messages, str):
        if parser == "xml_function":
            count_xml(messages, counter)
        return
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"assistant", "ai"}:
            continue
        if parser == "xml_function":
            count_xml(message.get("content") or "", counter)
        else:
            count_tool_calls(message.get("tool_calls"), counter)
            if parser == "xml_function":
                count_xml(message.get("content") or "", counter)


def measure_one(pq, dest: Path, spec: dict) -> dict:
    files = parquet_files(dest)
    rows = 0
    actions: Counter = Counter()
    resolved = Counter()
    message_column = {
        "xml_function": "messages",
        "smith_tool_calls": "messages",
        "openhands_tool_calls": "trajectory",
    }[spec["parser"]]
    for path in files:
        pf = pq.ParquetFile(path)
        rows += pf.metadata.num_rows
        names = set(pf.schema_arrow.names)
        columns = [message_column] if message_column in names else []
        if "resolved" in names:
            columns.append("resolved")
        if not columns:
            continue
        # Large OpenHands trajectories are counted from the message column
        # alone. A combined to_pydict of 64 full traces was killed mid-scan.
        for batch in pf.iter_batches(batch_size=128, columns=columns):
            data = batch.to_pydict()
            for messages in data.get(message_column, []):
                scan_messages(messages, actions, spec["parser"])
            for flag in data.get("resolved", []):
                resolved[str(flag)] += 1
    return {
        "rows": rows,
        "parquet_files": len(files),
        "bytes": dir_bytes(dest),
        "actions": dict(actions.most_common()),
        "resolved": dict(resolved),
    }


def download_one(snapshot_download, data_root: Path, spec: dict) -> Path:
    dest = data_root / "datasets" / spec["kind"] / spec["dirname"]
    blocked = {
        (data_root / "datasets" / "extra").resolve(),
        (data_root / "datasets" / "curated").resolve(),
    }
    if dest.resolve() in blocked or any(parent in blocked for parent in dest.resolve().parents):
        raise SystemExit(f"Refusing to write into a reserved dataset directory: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": spec["repo"],
        "repo_type": "dataset",
        "revision": spec["revision"],
        "local_dir": str(dest),
    }
    if spec["allow_patterns"]:
        kwargs["allow_patterns"] = spec["allow_patterns"]
    snapshot_download(**kwargs)
    return dest


def main() -> None:
    data_root = resolve_data_root()
    configure_hf(data_root)
    from huggingface_hub import snapshot_download

    pq = load_pyarrow(data_root)
    (data_root / "datasets" / "agent").mkdir(parents=True, exist_ok=True)
    (data_root / "datasets" / "coding").mkdir(parents=True, exist_ok=True)

    report = []
    for spec in STAGED:
        print(f"downloading {spec['repo']} @ {spec['revision']}", flush=True)
        dest = download_one(snapshot_download, data_root, spec)
        print(f"measuring {dest}", flush=True)
        measured = measure_one(pq, dest, spec)
        entry = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "license": spec["license"],
            "shape": spec["shape"],
            "path": str(dest),
            **measured,
        }
        report.append(entry)
        print(json.dumps(entry, indent=2), flush=True)

    manifest = data_root / "datasets" / "coding" / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    note = data_root / "datasets" / "agent" / "NOT_STAGED.txt"
    note.write_text(
        "No general-agent corpus was staged here.\n"
        "The public agent sets that were opened (AgentBank, AgentGym, AgentTrek,\n"
        "ToolBench, Mind2Web) use web, embodied, or API actions rather than\n"
        "shell / read_file / write_file / patch. See docs/agent-coding-datasets.md.\n"
    )
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
