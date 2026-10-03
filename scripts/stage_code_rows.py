"""Turn staged coding trajectories into next-tool rows on the external SSD.

OpenHands and the SWE-smith shards that are already on disk. Incomplete
SWE-smith shards are included and named as partial. Output is train-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nca_text.data import datasets_dir  # noqa: E402

SSD_CODING = Path("/Volumes/Extreme SSD/nca_research/datasets/coding")
OUT_DIR = Path("/Volumes/Extreme SSD/nca_research/datasets/extra")
CAP = 500
RECENT = 4

_FN = re.compile(r"<function=([a-zA-Z0-9_]+)>")
_PARAM = re.compile(r"<parameter=command>(.*?)</parameter>", re.DOTALL)

HERMES = {
    "bash": "terminal",
    "execute_bash": "terminal",
    "terminal": "terminal",
    "execute_code": "execute_code",
    "python": "execute_code",
    "search": "search_files",
    "search_files": "search_files",
    "web_search": "web_search",
    "finish": "respond",
    "submit": "respond",
}


def _cap(text: object) -> str:
    if isinstance(text, list):
        text = " ".join(str(part) for part in text)
    return " ".join(str(text or "").split())[:CAP]


def _map_call(blob: str) -> str | None:
    match = _FN.search(blob or "")
    if not match:
        return None
    name = match.group(1)
    if name in HERMES:
        return HERMES[name]
    if name != "str_replace_editor":
        return name
    command = _PARAM.search(blob)
    verb = (command.group(1).strip().lower() if command else "")
    if verb in {"view", "open"}:
        return "read_file"
    if verb in {"create", "write"}:
        return "write_file"
    if verb in {"str_replace", "insert", "undo_edit"}:
        return "patch"
    return "patch" if verb else "read_file"


def _rows_from_messages(session_id: str, messages: list[dict], source: str) -> list[dict]:
    goal = ""
    pending: list[tuple[str, str]] = []
    rows: list[dict] = []
    step = 0
    for message in messages:
        role = message.get("role") or ""
        content = message.get("content") or ""
        if role == "user" and not goal:
            goal = _cap(content)
        if role != "assistant":
            if role == "user" and pending:
                tool, _ = pending[-1]
                pending[-1] = (tool, _cap(content))
            continue
        target = _map_call(content)
        if target is None:
            target = "respond"
        recent = [{"tool": tool, "result": result} for tool, result in pending[-RECENT:]]
        rows.append(
            {
                "session_id": session_id,
                "step_index": step,
                "goal": goal,
                "recent_steps": recent,
                "target_tool": target,
                "source": source,
            }
        )
        step += 1
        pending.append((target, "[pending]"))
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def stage_openhands() -> None:
    import pyarrow.parquet as pq

    src = SSD_CODING / "openhands-sft" / "data" / "train.success.oss-00000-of-00001.parquet"
    if not src.is_file():
        print(f"skip missing {src}", flush=True)
        return
    rows: list[dict] = []
    table = pq.read_table(src, columns=["messages"])
    for i, messages in enumerate(table.column("messages").to_pylist()):
        rows.extend(_rows_from_messages(f"openhands-{i}", messages, "openhands-sft"))
    _write(OUT_DIR / "openhands_train.jsonl", rows)


def stage_swe_smith() -> None:
    import pyarrow.parquet as pq

    data = SSD_CODING / "swe-smith-trajectories-tool" / "data"
    shards = sorted(p for p in data.glob("tool-*.parquet") if not p.name.startswith("._"))
    if not shards:
        print("skip swe-smith: no shards", flush=True)
        return
    complete = len(shards) >= 8
    out = OUT_DIR / ("swe_smith_train.jsonl" if complete else "swe_smith_partial_train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as handle:
        for shard in shards:
            table = pq.read_table(shard, columns=["messages", "traj_id"])
            ids = table.column("traj_id").to_pylist()
            messages = table.column("messages").to_pylist()
            for traj_id, raw in zip(ids, messages, strict=True):
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                for row in _rows_from_messages(f"swe-smith-{traj_id}", parsed, shard.name):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
            print(f"swe-smith {shard.name} running_rows={n}", flush=True)
    print(f"wrote {out} ({n} rows)", flush=True)
    print(f"swe-smith shards used: {[p.name for p in shards]}", flush=True)


def _editor_verb(arguments: object) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("command") or "").strip().lower()


def _map_named(name: str, arguments: object = None) -> str | None:
    """Hermes name for a coding-agent call. None means this call is not a training label."""
    if name in {"think", "undo_edit"}:
        return None
    if name == "task_tracker":
        return "todo"
    if name in HERMES:
        return HERMES[name]
    if name in {"str_replace_editor", "file_editor"}:
        verb = _editor_verb(arguments)
        if verb in {"view", "open"}:
            return "read_file"
        if verb in {"create", "write"}:
            return "write_file"
        if verb in {"str_replace", "insert"}:
            return "patch"
        if verb == "undo_edit":
            return None
        return "read_file"
    return name or None


def _write_rows(path: Path, rows_iter) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as handle:
        for row in rows_iter:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {path} ({n} rows)", flush=True)
    return n


def _rows_from_tool_messages(session_id: str, messages: list, source: str):
    goal = ""
    pending: list[tuple[str, str]] = []
    step = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or ""
        if role == "system":
            continue
        if role == "user" and not goal:
            goal = _cap(message.get("content"))
            continue
        if role == "tool":
            text = _cap(message.get("content"))
            for i in range(len(pending) - 1, -1, -1):
                if pending[i][1] == "[pending]":
                    pending[i] = (pending[i][0], text)
                    break
            continue
        if role not in {"assistant", "ai"}:
            continue
        calls = message.get("tool_calls") or []
        targets: list[str] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            mapped = _map_named(str(function.get("name") or call.get("name") or ""), function.get("arguments"))
            if mapped:
                targets.append(mapped)
        if not targets and not calls:
            blob = message.get("content") or ""
            if isinstance(blob, str) and "<function=" in blob:
                mapped = _map_call(blob)
                targets = [mapped] if mapped else ["respond"]
            else:
                targets = ["respond"]
        for target in targets:
            recent = [{"tool": tool, "result": result} for tool, result in pending[-RECENT:]]
            yield {
                "session_id": session_id,
                "step_index": step,
                "goal": goal,
                "recent_steps": recent,
                "target_tool": target,
                "source": source,
            }
            step += 1
            pending.append((target, "[pending]"))


def stage_r2e() -> None:
    import pyarrow.parquet as pq

    src = SSD_CODING / "r2e-gym-sft" / "data" / "train-00000-of-00001.parquet"
    if not src.is_file():
        print(f"skip missing {src}", flush=True)
        return
    table = pq.read_table(src, columns=["messages"])

    def rows():
        for i, messages in enumerate(table.column("messages").to_pylist()):
            yield from _rows_from_messages(f"r2e-{i}", messages, "r2e-gym-sft")

    _write_rows(OUT_DIR / "r2e_train.jsonl", rows())


def stage_rebench() -> None:
    import pyarrow.parquet as pq

    src = SSD_CODING / "swe-rebench-openhands" / "trajectories.parquet"
    if not src.is_file():
        print(f"skip missing {src}", flush=True)
        return
    pf = pq.ParquetFile(src)

    def rows():
        n = 0
        for batch in pf.iter_batches(batch_size=16, columns=["trajectory_id", "trajectory"]):
            for rec in batch.to_pylist():
                session = f"rebench-{rec.get('trajectory_id') or n}"
                yield from _rows_from_tool_messages(session, rec.get("trajectory") or [], "swe-rebench")
                n += 1
            if n and n % 2000 == 0:
                print(f"rebench trajectories {n}", flush=True)

    _write_rows(OUT_DIR / "rebench_train.jsonl", rows())


def main() -> None:
    if not str(datasets_dir()).startswith("/Volumes/"):
        raise SystemExit("datasets dir is not on an external volume")
    for extra in (
        Path("/Volumes/Extreme SSD/nca_research/py-tools"),
        Path("/Volumes/Extreme SSD/nca_research/datasets/extra/pydeps"),
    ):
        if extra.is_dir():
            sys.path.insert(0, str(extra))
    chosen = sys.argv[1:] or ["openhands", "swe-smith"]
    dispatch = {
        "openhands": stage_openhands,
        "swe-smith": stage_swe_smith,
        "r2e": stage_r2e,
        "rebench": stage_rebench,
    }
    for name in chosen:
        if name not in dispatch:
            raise SystemExit(f"unknown set {name}")
        dispatch[name]()


if __name__ == "__main__":
    main()
