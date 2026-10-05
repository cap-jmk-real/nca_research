#!/usr/bin/env python3
"""One-screen auto-research status for agent supervision ticks.

Prints ≤12 short lines. No docs, no full logs, no experiment dumps.
Prefer this over reading daemon.log / console.log during monitor ticks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SSD = Path("/Volumes/Extreme SSD/nca_research/results")
STATE = SSD / "auto_research" / "daemon" / "state.json"
HEARTBEAT = SSD / "auto_research" / "daemon" / "heartbeat.json"
LATEST = SSD / "auto_research" / "latest.json"
CONSOLE = SSD / "auto_research" / "daemon" / "console.log"
ROOT = Path(__file__).resolve().parents[1]


def _j(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _last_console() -> str:
    if not CONSOLE.is_file():
        return "-"
    try:
        with CONSOLE.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 800))
            chunk = handle.read().decode("utf-8", "replace")
        line = [ln for ln in chunk.splitlines() if ln.strip()][-1]
        return line[:160]
    except (OSError, IndexError):
        return "-"


def _procs() -> str:
    try:
        out = subprocess.check_output(
            ["pgrep", "-fl", "hypotheses|auto_research|compile_specialist|distill"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"
    names = []
    for line in out.splitlines()[:4]:
        if "hypotheses" in line:
            names.append("hypotheses")
        elif "compile_specialist" in line:
            names.append("compile")
        elif "distill" in line:
            names.append("distill")
        elif "run_auto_research_daemon" in line:
            names.append("daemon")
        elif "auto_research" in line:
            names.append("scoreboard")
    return ",".join(names) or "other"


def _branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "?"


def _dirty() -> str:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "?"
    return "dirty" if out.strip() else "clean"


def main() -> None:
    state = _j(STATE)
    hb = _j(HEARTBEAT)
    latest = _j(LATEST)
    qi = state.get("queue_index", "?")
    done = len(state.get("completed") or [])
    failed = len(state.get("failed") or [])
    fail_gates = latest.get("failing_gates") or []
    block_gates = latest.get("blocked_gates") or []
    lines = [
        f"branch={_branch()} git={_dirty()}",
        f"deadline={state.get('deadline_iso', '?')}",
        f"queue_index={qi} completed={done} failed={failed} cycles={state.get('cycles', 0)}",
        f"procs={_procs()}",
        f"heartbeat={hb.get('ts', '-')}",
        f"failing_gates={len(fail_gates)} blocked={len(block_gates)}",
        f"console={_last_console()}",
    ]
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)
    lean_path = SSD / "auto_research" / "daemon" / "status_lean.txt"
    lean_path.parent.mkdir(parents=True, exist_ok=True)
    lean_path.write_text(text)


if __name__ == "__main__":
    main()
