"""Closed-set next-tool accuracy of a small instruct model on local Hermes.

Reads ~/.hermes/state.db in memory, same rows as the 1D NCA test. Writes
counts only. Does not write goals, tool results, or model text.
"""

from __future__ import annotations

import json
import os
import re
import sys
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


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples, note = load_local_test()
    rows = [ex for ex in examples if ex.label in LABEL_SET]
    print(f"local rows in vocab {len(rows)} of {len(examples)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

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
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _prompt(ex.goal, ex.steps)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[-1] :]
        pred = _parse(tokenizer.decode(new_tokens, skip_special_tokens=True))
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
        "model": MODEL_ID,
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
