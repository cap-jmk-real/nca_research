#!/usr/bin/env python3
"""Fail if files in the supplement manifest contain author-identifying strings."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "paper" / "tmlr" / "supplement_manifest.txt"

PAT = re.compile(
    r"github\.com/cap-jmk|cap-jmk-real|kleber|julian\.m\.|"
    r"orcid|0000-0001-5518|/home/s4il0r|s4il0r@|"
    r"julian\.m\.kleber@gmail",
    re.I,
)


def main() -> int:
    if not MANIFEST.is_file():
        print(f"missing {MANIFEST}; run write_supplement_manifest.py", file=sys.stderr)
        return 1
    paths = [REPO / line.strip() for line in MANIFEST.read_text().splitlines() if line.strip()]
    bad: list[str] = []
    for p in paths:
        if not p.is_file():
            bad.append(f"missing: {p.relative_to(REPO)}")
            continue
        if PAT.search(p.as_posix()):
            bad.append(f"path: {p.relative_to(REPO)}")
            continue
        if p.suffix.lower() in {".png", ".pdf", ".pt", ".npz", ".pkl", ".zip"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = PAT.search(text)
        if m:
            bad.append(f"content {p.relative_to(REPO)}: …{m.group(0)}…")
    if bad:
        print("Supplement anonymity check failed:\n", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"check_supplement_anonymity: ok ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())