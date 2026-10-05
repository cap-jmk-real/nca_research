"""Stage FineEnvs common-task tool calls as Hermes next-tool rows.

The source is FineEnvs/SmolDataEnvs-multiharness-sft, the verifier-correct
rollouts from Qwen3.8-27B across OpenCode, Claude Code, Codex, and
Mini-SWE-Agent. Only tasks solved in all four harnesses are kept. A prose
completion is dropped: this mix is already 37% respond, and curated Hermes
val is 5% respond. Tool names that do not map onto the 13-way head are dropped
too, so the class count stays 13.

All four harnesses of one task share a split. sha256(task_id) % 10 == 0 is
the side eval file. It is not the curated validation set.

Writes under the external datasets directory. Refuses to run if that volume
is missing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nca_text.data import assert_external, datasets_dir  # noqa: E402

REPO = "FineEnvs/SmolDataEnvs-multiharness-sft"
HARNESSES = ("opencode", "claude-code", "codex", "mini-swe-agent")
API = "https://huggingface.co/api/datasets"
RESOLVE = "https://huggingface.co/datasets"
CAP = 500
RECENT = 4

KEEP = {
    "delegate_task",
    "execute_code",
    "memory",
    "patch",
    "process",
    "read_file",
    "respond",
    "search_files",
    "session_search",
    "skill_view",
    "terminal",
    "todo",
    "write_file",
}

# Lowercase harness tool name -> Hermes label. Names already in KEEP pass through.
TOOL_MAP = {
    "bash": "terminal",
    "shell": "terminal",
    "shell_command": "terminal",
    "local_shell": "terminal",
    "exec": "terminal",
    "exec_command": "terminal",
    "execute": "terminal",
    "execute_bash": "terminal",
    "read": "read_file",
    "view": "read_file",
    "cat": "read_file",
    "write": "write_file",
    "create": "write_file",
    "edit": "patch",
    "str_replace": "patch",
    "apply_patch": "patch",
    "grep": "search_files",
    "glob": "search_files",
    "search": "search_files",
    "find": "search_files",
    "python": "execute_code",
    "ipython": "execute_code",
    "todowrite": "todo",
    "todo_write": "todo",
    "task_tracker": "todo",
    "agent": "delegate_task",
    "task": "delegate_task",
}


def map_tool(name: str) -> str | None:
    key = (name or "").strip()
    if key in KEEP and key != "respond":
        return key
    mapped = TOOL_MAP.get(key.lower())
    if mapped in KEEP:
        return mapped
    return None


def _text(content: object) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        content = " ".join(parts)
    return " ".join(str(content or "").split())


def _cap(content: object) -> str:
    return _text(content)[:CAP]


def _as_messages(value: object) -> list:
    if isinstance(value, str):
        value = json.loads(value) if value else []
    return value if isinstance(value, list) else []


def _call_names(message: dict) -> list[str]:
    names: list[str] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        name = function.get("name") or call.get("name")
        if name:
            names.append(str(name))
    if names:
        return names
    blob = _text(message.get("content"))
    start = 0
    while True:
        tag = blob.find("<function=", start)
        if tag < 0:
            break
        end = blob.find(">", tag)
        if end < 0:
            break
        names.append(blob[tag + len("<function=") : end])
        start = end + 1
    return names


def history_and_goal(prompt: object) -> tuple[str, list[tuple[str, str]]]:
    goal = ""
    pending: list[tuple[str, str]] = []
    for message in _as_messages(prompt):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role == "system":
            continue
        if role in {"user", "human"}:
            text = _cap(message.get("content"))
            if not goal:
                goal = text
            elif pending and pending[-1][1] == "[pending]":
                pending[-1] = (pending[-1][0], text)
            continue
        if role == "tool":
            text = _cap(message.get("content"))
            for i in range(len(pending) - 1, -1, -1):
                if pending[i][1] == "[pending]":
                    pending[i] = (pending[i][0], text)
                    break
            continue
        if role not in {"assistant", "gpt", "ai"}:
            continue
        for name in _call_names(message):
            mapped = map_tool(name)
            if mapped:
                pending.append((mapped, "[pending]"))
    return goal, pending


def target_tool(completion: object) -> tuple[str | None, str | None]:
    """First mapped tool in the completion, and the raw name if nothing mapped."""
    raw_unmapped = None
    for message in _as_messages(completion):
        if not isinstance(message, dict):
            continue
        for name in _call_names(message):
            mapped = map_tool(name)
            if mapped:
                return mapped, None
            raw_unmapped = name
    return None, raw_unmapped


def is_val_task(task_id: str) -> bool:
    digest = hashlib.sha256(task_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def row_from_record(record: dict) -> dict | None:
    if not record.get("common_task"):
        return None
    task_id = str(record.get("task_id") or "")
    if not task_id:
        return None
    target, _raw = target_tool(record.get("completion"))
    if target is None:
        return None
    goal, pending = history_and_goal(record.get("prompt"))
    recent = [{"tool": tool, "result": result} for tool, result in pending[-RECENT:]]
    harness = str(record.get("harness") or "")
    return {
        "session_id": f"fineenvs-{task_id}",
        "step_index": str(record.get("turn_id") or ""),
        "goal": goal,
        "recent_steps": recent,
        "target_tool": target,
        "source": f"fineenvs-{harness}",
        "harness": harness,
    }


def _self_test() -> None:
    bash = {
        "common_task": True,
        "task_id": "task-a",
        "turn_id": "t1",
        "harness": "opencode",
        "prompt": [
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "How many rows are malignant?"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "569 rows"},
        ],
        "completion": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}]}
        ],
    }
    row = row_from_record(bash)
    assert row is not None
    assert row["target_tool"] == "read_file"
    assert row["recent_steps"] == [{"tool": "terminal", "result": "569 rows"}]
    assert row["goal"].startswith("How many")
    prose = dict(bash)
    prose["completion"] = [{"role": "assistant", "content": "The answer is 212."}]
    assert row_from_record(prose) is None
    unknown = dict(bash)
    unknown["completion"] = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "WebFetch"}}]}
    ]
    assert row_from_record(unknown) is None
    assert map_tool("apply_patch") == "patch"
    assert map_tool("respond") is None
    same = [is_val_task("task-a") for _ in range(4)]
    assert same[0] == same[-1]
    print("self-test ok", flush=True)


def _list_parquets(harness: str) -> list[str]:
    url = f"{API}/{REPO}/tree/main/conversations/{harness}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode())
    paths = [
        item["path"]
        for item in payload
        if item.get("type") == "file" and str(item.get("path", "")).endswith(".parquet")
    ]
    if not paths:
        raise SystemExit(f"no parquet files listed for {harness}")
    return paths


def _download(path: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RESOLVE}/{REPO}/resolve/main/{path}"
    tmp = dest.with_suffix(dest.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _records(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["task_id", "turn_id", "harness", "common_task", "prompt", "completion"],
    )
    yield from table.to_pylist()


def stage() -> None:
    root = datasets_dir()
    assert_external(root)
    raw = root / "fineenvs" / "raw"
    out_dir = root / "extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    dropped = Counter()
    per_harness: Counter[str] = Counter()
    train_path = out_dir / "fineenvs_common_train.jsonl"
    val_path = out_dir / "fineenvs_common_val.jsonl"
    train_tasks: set[str] = set()
    val_tasks: set[str] = set()
    with train_path.open("w") as train_f, val_path.open("w") as val_f:
        for harness in HARNESSES:
            for rel in _list_parquets(harness):
                dest = raw / rel
                print(f"fetch {rel}", flush=True)
                _download(rel, dest)
                for record in _records(dest):
                    if not record.get("common_task"):
                        dropped["not_common_task"] += 1
                        continue
                    target, raw_name = target_tool(record.get("completion"))
                    if target is None:
                        dropped["prose" if raw_name is None else f"unmapped:{raw_name}"] += 1
                        continue
                    row = row_from_record(record)
                    if row is None:
                        dropped["empty"] += 1
                        continue
                    handle = val_f if is_val_task(str(record["task_id"])) else train_f
                    if handle is val_f:
                        val_tasks.add(str(record["task_id"]))
                    else:
                        train_tasks.add(str(record["task_id"]))
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[target] += 1
                    per_harness[str(record.get("harness") or harness)] += 1
    overlap = train_tasks & val_tasks
    if overlap:
        raise SystemExit(f"task split overlap: {len(overlap)}")
    stats = {
        "dataset": REPO,
        "kept": dict(counts),
        "kept_rows": int(sum(counts.values())),
        "per_harness": dict(per_harness),
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "dropped": dict(dropped),
        "train": str(train_path),
        "val": str(val_path),
        "note": (
            "Prose completions are dropped. Val is a task holdout, not curated Hermes val. "
            "Training draws these rows at the curated Hermes label prior."
        ),
    }
    stats_path = out_dir / "fineenvs_common_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test()
        return
    for extra in (
        Path("/Volumes/Extreme SSD/nca_research/py-tools"),
        Path("/Volumes/Extreme SSD/nca_research/datasets/extra/pydeps"),
    ):
        if extra.is_dir():
            sys.path.insert(0, str(extra))
    stage()


if __name__ == "__main__":
    main()
