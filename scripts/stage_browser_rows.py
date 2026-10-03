"""Pull browser_* tool decisions out of the Hermes parquets already on the SSD.

The curated and extra jsonl dropped names starting with browser_. This pass
keeps them, under the same session split, so a browser call is never in both
train and val.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.curate_traces import (  # noqa: E402
    PENDING,
    RECENT_STEPS,
    extract_tool_calls,
    extract_tool_results,
    message_text,
    role_of,
    truncate,
)

SNAP = Path(
    "/Volumes/Extreme SSD/nca_research/datasets/hf_home/hub"
    "/datasets--lambda--hermes-agent-reasoning-traces/snapshots"
    "/b92885e4f0161d4b2536512710e004d4892cac6e"
)
PARQUETS = [
    SNAP / "data/kimi/train.parquet",
    SNAP / "data/glm-5.1/train.parquet",
]
OUT_DIR = Path("/Volumes/Extreme SSD/nca_research/datasets/extra")


def split_is_val(session_id: str) -> bool:
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10 == 0


def rows_for_session(session_id: str, conversations: list) -> list[dict]:
    goal = ""
    history: list[dict] = []
    pending: list[int | None] = []
    out: list[dict] = []
    step = 0

    def emit(target: str) -> None:
        nonlocal step
        recent = [
            {"tool": item["tool"], "result": item["result"]}
            for item in history[-RECENT_STEPS:]
        ]
        out.append(
            {
                "session_id": session_id,
                "step_index": step,
                "goal": goal,
                "recent_steps": recent,
                "target_tool": target,
            }
        )
        step += 1

    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = role_of(message)
        text = message_text(message)
        if role in {"human", "user"}:
            goal = truncate(text.strip(), 500)
            continue
        if role == "system":
            continue
        if role == "tool":
            results, _errors = extract_tool_results(text)
            for result in results:
                if not pending:
                    continue
                slot = pending.pop(0)
                if slot is None:
                    continue
                history[slot]["result"] = result
            pending.clear()
            continue
        if role not in {"gpt", "assistant"}:
            continue
        pending.clear()
        names = extract_tool_calls(text)
        if not names:
            continue
        for raw_name in names:
            if not raw_name or not raw_name.startswith("browser_"):
                if raw_name:
                    history.append({"tool": raw_name, "result": PENDING})
                    pending.append(len(history) - 1)
                else:
                    pending.append(None)
                continue
            emit(raw_name)
            history.append({"tool": raw_name, "result": PENDING})
            pending.append(len(history) - 1)
    return out


def main() -> None:
    import pyarrow.parquet as pq

    train_path = OUT_DIR / "browser_train.jsonl"
    val_path = OUT_DIR / "browser_val.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    n_train = 0
    n_val = 0
    with train_path.open("w") as train_f, val_path.open("w") as val_f:
        for path in PARQUETS:
            if not path.is_file():
                print(f"missing {path}", flush=True)
                continue
            print(f"scan {path.name} in {path.parent.name}", flush=True)
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=64, columns=["id", "conversations"]):
                for rec in batch.to_pylist():
                    session_id = str(rec.get("id") or "")
                    rows = rows_for_session(session_id, rec.get("conversations") or [])
                    handle = val_f if split_is_val(session_id) else train_f
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[row["target_tool"]] += 1
                        if split_is_val(session_id):
                            n_val += 1
                        else:
                            n_train += 1
    print(f"browser train {n_train} val {n_val}", flush=True)
    print(counts.most_common(30), flush=True)


if __name__ == "__main__":
    main()
