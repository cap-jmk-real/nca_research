#!/usr/bin/env python3
"""Run the shell blocks of the documentation that are marked for CI.

Single source of truth for commands: the README and the docs pages contain the
commands, and this script executes them. A fenced block is executed when the
line directly above the opening fence is the HTML comment ``<!-- ci:run -->``.
All marked blocks of all listed files run in order inside one bash process
(``set -euo pipefail``), so a virtual environment activated in the first block
stays active in later blocks. A block whose text is identical to an earlier
one runs only once. Blocks without the marker (for example full GPU
runs) are never executed.

Usage:
    python tools/run_readme_blocks.py README.md docs/reproduce.md
    python tools/run_readme_blocks.py --list README.md     # print, do not run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MARKER = "<!-- ci:run -->"
FENCE = re.compile(r"^(```|~~~)(\w*)\s*$")


def extract(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == MARKER:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            m = FENCE.match(lines[j]) if j < len(lines) else None
            if not m or m.group(2) not in ("bash", "sh", "shell", ""):
                raise SystemExit(f"{path}:{i + 1}: marker must precede a bash code block")
            fence = m.group(1)
            k = j + 1
            body: list[str] = []
            while k < len(lines) and lines[k].strip() != fence:
                body.append(lines[k])
                k += 1
            if k >= len(lines):
                raise SystemExit(f"{path}:{j + 1}: unterminated code block")
            blocks.append("\n".join(body))
            i = k
        i += 1
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--list", action="store_true", help="print the script instead of running it")
    args = ap.parse_args()

    parts: list[str] = []
    seen: set[str] = set()
    for f in args.files:
        found = extract(f)
        if not found:
            raise SystemExit(f"{f}: no block marked with {MARKER}")
        for n, b in enumerate(found, 1):
            if b in seen:
                continue
            seen.add(b)
            parts.append(f"echo '--- {f} block {n} ---'\n{b}")
    script = "set -euo pipefail\n" + "\n".join(parts) + "\n"
    if args.list:
        print(script)
        return 0
    return subprocess.run(["bash", "-c", script]).returncode


if __name__ == "__main__":
    sys.exit(main())
