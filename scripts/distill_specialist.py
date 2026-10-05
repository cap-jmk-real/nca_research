#!/usr/bin/env python3
"""Distill a ≤9B teacher into a specialist student (labels / SFT text only).

Default teacher: qwen3.5:9b on engine Ollama. This script:
1) Samples specialist jsonl rows.
2) Asks the teacher for a short specialist completion or route label.
3) Writes a distill jsonl on the SSD for later LoRA/SFT (training not launched here
   unless --train is passed and mlx/transformers env is ready).

Does not distill into TextNCA hidden states.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

SSD = Path("/Volumes/Extreme SSD/nca_research")
SPEC = SSD / "datasets" / "specialists"
OUT = SSD / "results" / "specialists" / "distill"
TEACHER = os.environ.get("NCA_TEACHER", "qwen3.5:9b")
OLLAMA = os.environ.get("NCA_OLLAMA", "http://192.168.10.29:11434")


def _ollama_chat(prompt: str, model: str, timeout: float = 120.0) -> str:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }
    req = urllib.request.Request(
        f"{OLLAMA.rstrip('/')}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    return str((payload.get("message") or {}).get("content") or "")


def _prompt(lang: str, row: dict) -> str:
    paths = ", ".join(row.get("path_hints") or [])
    return (
        f"You are a {lang} coding specialist (≤9B teacher distillation target).\n"
        f"Goal: {row.get('goal') or '(none)'}\n"
        f"Paths: {paths or '(none)'}\n"
        f"Write a short plan (3 bullets) then the next tool name from: "
        f"terminal, read_file, write_file, patch, execute_code, respond.\n"
        f"Format:\nPLAN:\n- ...\nTOOL: <name>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["python", "typescript", "shell", "orchestrator"])
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--teacher", default=TEACHER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_dir = SPEC / args.lang
    if not src_dir.is_dir():
        raise SystemExit(f"missing specialist data: {src_dir} (run compile_specialist_datasets.py)")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.lang}_from_{args.teacher.replace(':', '_')}.jsonl"
    rows = []
    for path in sorted(src_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= args.limit:
                    break
        if len(rows) >= args.limit:
            break

    print(f"distill lang={args.lang} n={len(rows)} teacher={args.teacher} out={out_path}", flush=True)
    if args.dry_run:
        print(_prompt(args.lang, rows[0] if rows else {}))
        return

    written = 0
    with out_path.open("w") as out:
        for row in rows:
            try:
                text = _ollama_chat(_prompt(args.lang, row), args.teacher)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"engine unreachable: {exc}", flush=True)
                break
            out.write(
                json.dumps(
                    {
                        "specialist": args.lang,
                        "teacher": args.teacher,
                        "source_row": {
                            "session_id": row.get("session_id"),
                            "goal": row.get("goal"),
                            "path_hints": row.get("path_hints"),
                            "source": row.get("source"),
                        },
                        "teacher_text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written % 10 == 0:
                print(f"wrote {written}", flush=True)
    meta = {
        "lang": args.lang,
        "teacher": args.teacher,
        "written": written,
        "path": str(out_path),
        "cap_b": 9,
    }
    (OUT / f"{args.lang}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
