"""Turn staged Pi session jsonl into next-tool rows on the external SSD.

Maps the common Pi tools onto the Hermes names. Less common tools keep their
own names. All rows are train; these sessions have no official split.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_ROOT = Path("/Volumes/Extreme SSD/nca_research")
OUT = DATA_ROOT / "datasets" / "extra" / "pi_train.jsonl"
CAP = 500
RECENT = 4
ROOTS = [
    DATA_ROOT / "datasets" / "agent" / "pi-maxdevv",
    DATA_ROOT / "datasets" / "agent" / "pi-julien-c",
    DATA_ROOT / "datasets" / "agent" / "pi-grfwings",
    DATA_ROOT / "datasets" / "agent" / "pi-abidlabs",
]
MAP = {
    "bash": "terminal",
    "read": "read_file",
    "write": "write_file",
    "edit": "patch",
    "grep": "search_files",
    "find": "search_files",
    "ls": "search_files",
    "read_url": "web_search",
    "synthetic_web_search": "web_search",
}


def cap(text: object) -> str:
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        text = " ".join(parts)
    return " ".join(str(text or "").split())[:CAP]


def tool_name(part: dict) -> str | None:
    raw = part.get("name")
    if not raw:
        return None
    return MAP.get(str(raw), str(raw))


def rows_for_file(path: Path) -> list[dict]:
    goal = ""
    pending: list[tuple[str, str]] = []
    rows: list[dict] = []
    session_id = path.stem
    step = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "session" and event.get("id"):
                session_id = f"pi-{event['id']}"
                continue
            if event.get("type") != "message":
                continue
            message = event.get("message") or {}
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                text = cap(content)
                if text and not goal:
                    goal = text
                if pending and text:
                    tool, _ = pending[-1]
                    pending[-1] = (tool, text)
                continue
            if role == "toolResult":
                text = cap(content)
                name = MAP.get(str(message.get("toolName") or ""), str(message.get("toolName") or ""))
                for i in range(len(pending) - 1, -1, -1):
                    if pending[i][1] == "[pending]" and (not name or pending[i][0] == name):
                        pending[i] = (pending[i][0], text)
                        break
                continue
            if role != "assistant":
                continue
            calls = []
            if isinstance(content, list):
                calls = [part for part in content if isinstance(part, dict) and part.get("type") == "toolCall"]
            targets = [tool_name(part) for part in calls]
            targets = [name for name in targets if name]
            if not targets:
                targets = ["respond"]
            for target in targets:
                recent = [{"tool": tool, "result": result} for tool, result in pending[-RECENT:]]
                rows.append(
                    {
                        "session_id": session_id,
                        "step_index": step,
                        "goal": goal,
                        "recent_steps": recent,
                        "target_tool": target,
                        "source": "pi",
                    }
                )
                step += 1
                pending.append((target, "[pending]"))
    return rows


def main() -> None:
    files = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        files.extend(
            p for p in root.rglob("*.jsonl")
            if not p.name.startswith("._") and ".cache" not in p.parts and "_upload_staging" not in p.parts
        )
    if not files:
        raise SystemExit("no pi jsonl on the SSD yet")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w") as handle:
        for path in files:
            for row in rows_for_file(path):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {OUT} ({n} rows from {len(files)} files)", flush=True)


if __name__ == "__main__":
    main()
