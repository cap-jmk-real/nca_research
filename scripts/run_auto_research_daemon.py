#!/usr/bin/env python3
"""3-day auto-research daemon for nca_research.

Runs on the M1 (MPS training). When engine (192.168.10.29) is reachable,
offloads Ollama/Klaut/LLM baseline scoring. State + logs stay on the SSD.

Stop when wall clock passes --days, or all non-blocked gates pass, or
TOUCH_STOP file appears.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSD = Path("/Volumes/Extreme SSD/nca_research/results")
STATE_DIR = SSD / "auto_research" / "daemon"
LOG_PATH = STATE_DIR / "daemon.log"
STATE_PATH = STATE_DIR / "state.json"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.json"
STOP_PATH = STATE_DIR / "STOP"
PYTHON = os.environ.get("NCA_PYTHON", "/opt/homebrew/bin/python3.14")
ENGINE = os.environ.get("NCA_ENGINE_HOST", "192.168.10.29")
OLLAMA = f"http://{ENGINE}:11434"
KLAUT = f"http://{ENGINE}:30487"

# Long-training floor (see docs/auto-research-loop.md §Long training).
# Short runs (8k–20k) are plumbing only; architectural claims need ≥150k,
# matching the mix13 recipe and recent Sudoku/iterative-reasoning work
# (long imitation on structured traces before the policy settles).
LONG_STEPS = os.environ.get("NCA_HYP_STEPS", "150000")

# Ordered experiment queue (hypothesis id -> command args).
QUEUE = (
    {
        "id": "scoreboard",
        "kind": "scoreboard",
        "cmd": [PYTHON, str(ROOT / "scripts" / "auto_research.py")],
    },
    {
        "id": "H6_long",
        "kind": "train",
        "cmd": [
            PYTHON,
            "-m",
            "nca_text.hypotheses",
            "--steps",
            LONG_STEPS,
            "--only",
            "long_range",
            "baseline",
            "flat",
        ],
    },
    {
        "id": "H2b_carry_long",
        "kind": "train",
        "cmd": [
            PYTHON,
            "-m",
            "nca_text.hypotheses",
            "--steps",
            LONG_STEPS,
            "--only",
            "carry",
            "baseline",
        ],
    },
    {
        "id": "H_halt_long",
        "kind": "train",
        "cmd": [
            PYTHON,
            "-m",
            "nca_text.hypotheses",
            "--steps",
            LONG_STEPS,
            "--only",
            "halt",
            "baseline",
        ],
    },
    {
        "id": "scoreboard_2",
        "kind": "scoreboard",
        "cmd": [PYTHON, str(ROOT / "scripts" / "auto_research.py")],
    },
    {
        "id": "H16_mix_prior",
        "kind": "train_main",
        "cmd": [PYTHON, "-m", "nca_text.train"],
        "note": "Continue/resume mix13-scale train (150k class). Do not use --fresh.",
        "optional": False,
    },
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{_now()} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as handle:
        handle.write(line + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {
        "started_at": _now(),
        "queue_index": 0,
        "completed": [],
        "failed": [],
        "engine_probes": [],
        "cycles": 0,
    }


def _env() -> dict:
    env = os.environ.copy()
    env["NCA_DATA_ROOT"] = str(SSD)
    env["TMPDIR"] = str(SSD / "tmp")
    env["PYTHONPYCACHEPREFIX"] = str(SSD / "pycache")
    env["HF_HOME"] = str(SSD / "hf-home")
    env["HF_HUB_CACHE"] = str(SSD / "hf-home" / "hub")
    env.setdefault("PYTHONPATH", str(ROOT))
    return env


def _reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:  # noqa: BLE001
        return False


def _probe_engine() -> dict:
    ollama = _reachable(f"{OLLAMA}/api/tags", timeout=4.0)
    klaut = False
    try:
        req = urllib.request.Request(
            f"{KLAUT}/v1/research/papers/search",
            data=json.dumps(
                {"query": "neural cellular automata", "sources": ["arxiv"], "limit": 1}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=8.0)
        klaut = True
    except Exception:  # noqa: BLE001
        klaut = False
    return {"ts": _now(), "ollama": ollama, "klaut": klaut, "host": ENGINE}


def _run(cmd: list[str], timeout: int | None = None) -> int:
    _log(f"RUN {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=_env(),
            timeout=timeout,
            check=False,
        )
        _log(f"EXIT {proc.returncode} for {cmd[-1] if cmd else '?'}")
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        _log(f"TIMEOUT {cmd}")
        return 124


def _engine_jobs(state: dict) -> None:
    probe = _probe_engine()
    state.setdefault("engine_probes", []).append(probe)
    _log(f"engine probe ollama={probe['ollama']} klaut={probe['klaut']}")
    if probe["klaut"]:
        _run(
            [
                PYTHON,
                str(ROOT / "scripts" / "auto_research.py"),
                "--klaut-search",
            ],
            timeout=300,
        )
    if probe["ollama"]:
        # Local LLM baseline on Hermes (transformers path may still use M1;
        # prefer engine Ollama when the score script supports it later).
        score = ROOT / "scripts" / "score_local_next_tool.py"
        if score.is_file():
            _run([PYTHON, str(score)], timeout=3600)


def _gates_all_pass() -> bool:
    latest = SSD / "auto_research" / "latest.json"
    if not latest.is_file():
        return False
    data = json.loads(latest.read_text())
    failing = data.get("failing_gates") or []
    return len(failing) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=3.0)
    parser.add_argument("--sleep-between", type=int, default=120)
    parser.add_argument("--once", action="store_true", help="Run one queue item then exit")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    # Preserve an existing mission deadline on restart (supervision ticks).
    if state.get("deadline_unix") and float(state["deadline_unix"]) > time.time():
        deadline = float(state["deadline_unix"])
    else:
        deadline = time.time() + args.days * 86400
        state["started_at"] = state.get("started_at") or _now()
    state["deadline_unix"] = deadline
    state["deadline_iso"] = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
    _write_json(STATE_PATH, state)
    _log(f"daemon start days={args.days} deadline={state['deadline_iso']} queue_index={state.get('queue_index', 0)}")

    while time.time() < deadline:
        _write_json(
            HEARTBEAT_PATH,
            {
                "ts": _now(),
                "pid": os.getpid(),
                "queue_index": state["queue_index"],
                "cycles": state["cycles"],
                "deadline_iso": state["deadline_iso"],
            },
        )
        if STOP_PATH.is_file():
            _log("STOP file present; exiting")
            break
        if _gates_all_pass():
            _log("all non-blocked gates pass; sleeping until deadline for rechecks")
            _engine_jobs(state)
            state["cycles"] += 1
            _write_json(STATE_PATH, state)
            if args.once:
                break
            time.sleep(max(args.sleep_between, 1800))
            continue

        # Engine offload attempt each cycle
        _engine_jobs(state)

        if state["queue_index"] >= len(QUEUE):
            # Restart scoreboard + short ablations until deadline
            state["queue_index"] = 0
            state["rollovers"] = int(state.get("rollovers") or 0) + 1
            _log(f"queue rollover #{state['rollovers']}")

        item = QUEUE[state["queue_index"]]
        if item.get("optional") and state["queue_index"] > 0 and not args.once:
            # Skip optional long jobs on rollover unless explicitly needed
            if int(state.get("rollovers") or 0) == 0:
                pass
            else:
                _log(f"skip optional {item['id']}")
                state["queue_index"] += 1
                _write_json(STATE_PATH, state)
                continue

        code = _run(item["cmd"], timeout=None if item["kind"] == "train_main" else 6 * 3600)
        record = {"id": item["id"], "ts": _now(), "exit": code}
        if code == 0:
            state["completed"].append(record)
        else:
            state["failed"].append(record)
        state["queue_index"] += 1
        state["cycles"] += 1
        _write_json(STATE_PATH, state)

        # Scoreboard after each train
        if item["kind"].startswith("train"):
            _run([PYTHON, str(ROOT / "scripts" / "auto_research.py")], timeout=600)

        if args.once:
            break
        time.sleep(args.sleep_between)

    state["finished_at"] = _now()
    state["reason"] = "deadline" if time.time() >= deadline else "stop_or_once"
    _write_json(STATE_PATH, state)
    _log(f"daemon end reason={state['reason']}")


if __name__ == "__main__":
    main()
