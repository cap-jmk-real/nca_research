"""Map public BFCL function names to the 13-label Hermes-style vocabulary (labels.json)."""

from __future__ import annotations

import json
import re
from pathlib import Path

LABELS = json.load(open(Path(__file__).with_name("labels.json")))
LABEL_SET = set(LABELS)

# Heuristic name→tool mapping for BFCL AST categories (unmapped names → execute_code).
_BFCL_PREFIX_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(get_|fetch_|retrieve_|find_|search_|list_|query_)"), "search_files"),
    (re.compile(r"^(read_|open_)"), "read_file"),
    (re.compile(r"^(write_|save_|create_|update_|edit_|patch_)"), "write_file"),
    (re.compile(r"^(run_|exec_|shell_|cmd_|terminal_)"), "terminal"),
    (re.compile(r"^(send_|email_|message_)"), "respond"),
    (re.compile(r"^(todo_|task_)"), "todo"),
    (re.compile(r"^(memory_|remember_)"), "memory"),
]

EXACT: dict[str, str] = {
    "respond": "respond",
    "answer": "respond",
    "clarify": "respond",
}


def map_bfcl_function(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return "respond"
    if n in LABEL_SET:
        return n
    if n in EXACT:
        return EXACT[n]
    low = n.lower()
    for pat, tool in _BFCL_PREFIX_RULES:
        if pat.search(low):
            return tool
    return "execute_code"


def gold_label_from_answer(answer: dict | None, *, irrelevance: bool) -> str:
    if irrelevance:
        return "respond"
    if not answer:
        return "respond"
    for item in answer.get("ground_truth") or []:
        if isinstance(item, dict) and item:
            fn = next(iter(item.keys()))
            return map_bfcl_function(str(fn))
    return "respond"
