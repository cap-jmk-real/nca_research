"""Closed-set next-tool accuracy of a small instruct model on local Hermes.

Reads ~/.hermes/state.db in memory, same rows as the 1D NCA test. Writes
counts only. Does not write goals, tool results, or model text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SSD = Path("/Volumes/Extreme SSD/nca_research")
os.environ.setdefault("HF_HOME", str(SSD / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(SSD / "hf-cache" / "hub"))
os.environ.setdefault("TMPDIR", str(SSD / "tmp"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from nca_text.data import load_local_test  # noqa: E402

LABELS = [
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
]
LABEL_SET = set(LABELS)
WORD = re.compile(r"[a-z_]+")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OLLAMA_MODEL = os.environ.get("NCA_BASELINE_MODEL", "qwen2.5:1.5b")
OLLAMA = os.environ.get("NCA_OLLAMA", "http://192.168.10.29:11434")
OUT = SSD / "results" / "local_next_tool_qwen25_1_5b.json"


# Match nca_text.data.TEXT_CAP so the transformer baseline is not fed a
# longer observation than the NCA hashed bag (fair + cheaper tokens).
_OBS_CAP = 160


def _prompt(goal: str, steps: list[tuple[str, str]]) -> str:
    lines = [f"Goal: {(goal or '(none)')[:_OBS_CAP]}"]
    if steps:
        lines.append("Recent tools, oldest first:")
        for name, text in steps[-5:]:
            lines.append(f"- {name}: {(text or '')[:_OBS_CAP]}")
    else:
        lines.append("Recent tools: none")
    lines.append("Next tool:")
    return "\n".join(lines)


def _parse(text: str) -> str | None:
    tokens = WORD.findall((text or "").lower())
    found = [tok for tok in tokens if tok in LABEL_SET]
    if len(found) == 1:
        return found[0]
    if found and all(tok == found[0] for tok in found):
        return found[0]
    return None


def _engine_models() -> set[str]:
    req = urllib.request.Request(f"{OLLAMA.rstrip('/')}/api/tags")
    with urllib.request.urlopen(req, timeout=8) as resp:
        payload = json.load(resp)
    return {str(row.get("name") or "") for row in payload.get("models") or []}


def _engine_reply(system: str, user: str, model: str) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8},
    }
    req = urllib.request.Request(
        f"{OLLAMA.rstrip('/')}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    return str((payload.get("message") or {}).get("content") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rescore even if the SSD baseline JSON already exists",
    )
    args = parser.parse_args()
    if OUT.is_file() and not args.force:
        print(f"baseline already on SSD, skip ({OUT})", flush=True)
        return

    examples, note = load_local_test()
    rows = [ex for ex in examples if ex.label in LABEL_SET]
    print(f"local rows in vocab {len(rows)} of {len(examples)}", flush=True)
    print(f"teacher={OLLAMA_MODEL} via {OLLAMA} (not this Mac)", flush=True)
    try:
        names = _engine_models()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"engine Ollama unreachable, refusing local teacher load: {exc}", flush=True)
        return
    if OLLAMA_MODEL not in names and f"{OLLAMA_MODEL}:latest" not in names:
        print(
            f"{OLLAMA_MODEL} is not on engine; refusing to download or load it on this Mac",
            flush=True,
        )
        return

    system = (
        "Reply with exactly one Hermes tool name and no other words. "
        "Choose from: " + ", ".join(LABELS) + "."
    )
    correct = 0
    unparsed = 0
    pred_hist: Counter[str] = Counter()
    gold_hist: Counter[str] = Counter()
    hit: Counter[str] = Counter()

    for i, ex in enumerate(rows, start=1):
        try:
            text = _engine_reply(system, _prompt(ex.goal, ex.steps), OLLAMA_MODEL)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"engine teacher failed, stopping without a local load: {exc}", flush=True)
            return
        pred = _parse(text)
        gold_hist[ex.label] += 1
        if pred is None:
            unparsed += 1
            pred_hist["unparsed"] += 1
        else:
            pred_hist[pred] += 1
            if pred == ex.label:
                correct += 1
                hit[ex.label] += 1
        if i % 50 == 0 or i == len(rows):
            print(f"n={i} acc={correct / i:.4f} unparsed={unparsed}", flush=True)

    n = len(rows)
    payload = {
        "model": OLLAMA_MODEL,
        "model_ref": MODEL_ID,
        "endpoint": OLLAMA,
        "note": note,
        "protocol": (
            "Same local ~/.hermes rows as the 1D NCA test, restricted to the "
            "13 training labels. Greedy decode, max 8 new tokens. A reply "
            "counts only when it contains exactly one of those names."
        ),
        "n": n,
        "correct": correct,
        "top1": correct / n if n else None,
        "unparsed": unparsed,
        "gold": dict(gold_hist),
        "pred": dict(pred_hist),
        "per_label_recall": {
            name: (hit[name] / gold_hist[name] if gold_hist[name] else None)
            for name in LABELS
        },
        "always_terminal": gold_hist["terminal"] / n if n else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
