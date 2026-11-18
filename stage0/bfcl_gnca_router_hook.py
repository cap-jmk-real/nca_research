"""GNCA System-1 hook for the official bfcl-eval run (engine).

The GNCA router predicts one of 13 Hermes action labels (stage0/labels.json); it cannot
emit BFCL function-call JSON. In the official protocol it acts only as the paper's dual-rule
gate (E2/E2v2): before every actuator query, the trace so far (latest user text + previous
tool calls mapped to the Hermes vocabulary via bfcl_label_map + their results) is routed;
if the router's top-1 is ``respond``, the tool schema is withheld for that query so the
actuator must answer in text (abstain). Otherwise the actuator sees the full official
tool list unchanged. Scoring is always the unmodified official bfcl checker.

Environment:
  GNCA_ROUTER_ENABLED=1         enable the gate (handler: local/qwen3.5-9b-FC-gnca)
  GNCA_ROUTER_URL               default http://127.0.0.1:18777 (runner/gnca_router_server.py,
                                which loads GNCA_CHECKPOINT in the paper env)
  GNCA_GATE_LOG                 optional jsonl path; one decision per actuator query
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request

try:
    from bfcl_label_map import map_bfcl_function
except ImportError:  # pragma: no cover
    from stage0.bfcl_label_map import map_bfcl_function

_LOCK = threading.Lock()


def router_enabled() -> bool:
    return os.environ.get("GNCA_ROUTER_ENABLED", "0").strip() in {"1", "true", "yes"}


def checkpoint_path() -> str | None:
    return os.environ.get("GNCA_CHECKPOINT")


def _get(m, k, default=None):
    if isinstance(m, dict):
        return m.get(k, default)
    return getattr(m, k, default)


def trace_from_messages(messages: list) -> tuple[str, list[tuple[str, str]]]:
    """Return (goal_text, [(hermes_label, result_text), ...]) from an OpenAI-style history."""
    goal = ""
    steps: list[tuple[str, str]] = []
    pending: dict[str, str] = {}
    for m in messages:
        role = _get(m, "role")
        if role == "user":
            goal = str(_get(m, "content") or "")
        elif role == "assistant":
            for tc in _get(m, "tool_calls") or []:
                fn = _get(tc, "function")
                pending[str(_get(tc, "id"))] = map_bfcl_function(str(_get(fn, "name") or ""))
        elif role == "tool":
            label = pending.pop(str(_get(m, "tool_call_id")), "execute_code")
            steps.append((label, str(_get(m, "content") or "")[:200]))
    return goal, steps


def route(goal: str, steps: list[tuple[str, str]]) -> dict:
    url = os.environ.get("GNCA_ROUTER_URL", "http://127.0.0.1:18777") + "/route"
    req = urllib.request.Request(url, data=json.dumps({"goal": goal, "steps": steps}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def gate(messages: list, entry_id: str | None = None) -> bool:
    """True -> withhold tools for this query (router top-1 == respond)."""
    goal, steps = trace_from_messages(messages)
    out = route(goal, steps)
    withhold = out["top1"] == "respond"
    log = os.environ.get("GNCA_GATE_LOG")
    if log:
        with _LOCK, open(log, "a") as f:
            f.write(json.dumps({"id": entry_id, "n_steps": len(steps), "top1": out["top1"],
                                "p_respond": out["probs"].get("respond"), "withhold_tools": withhold}) + "\n")
    return withhold
