#!/usr/bin/env python3
"""Write paper/tmlr/supplement_manifest.txt for the anonymised supplementary ZIP."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "tmlr" / "supplement_manifest.txt"


def main() -> None:
    include: list[str] = []

    def add(p: Path) -> None:
        if p.is_file():
            include.append(p.relative_to(REPO).as_posix())

    for pattern in (
        "paper/tmlr/supplement.tex",
        "paper/tmlr/supplement_manifest.txt",
        "paper/tmlr/Makefile",
        "paper/tmlr/README.md",
        "paper/tmlr/SUBMISSION_PACKAGE.md",
        "paper/tmlr/TMLR_STYLE_LICENSE_APACHE2.txt",
        "paper/tmlr/tmlr.sty",
        "paper/tmlr/tmlr.bst",
        "paper/tmlr/fancyhdr.sty",
        "paper/generated",
        "paper/figures",
        "paper/si",
        "scripts/build_",
        "scripts/check_tmlr_bare_floats.py",
        "scripts/build_protocol_pins_1124.py",
        "scripts/write_supplement_manifest.py",
        "stage0/stage0_results",
        "stage0/bfcl_official_runner",
        "formal_proofs",
        "data/curated/stats.json",
    ):
        path = REPO / pattern
        if pattern.endswith("/") or pattern in ("paper/generated", "paper/figures", "paper/si", "stage0/stage0_results", "formal_proofs"):
            if path.is_dir():
                for f in sorted(path.rglob("*")):
                    if f.is_file() and ".git" not in f.parts:
                        add(f)
        elif pattern == "scripts/build_":
            for f in sorted((REPO / "scripts").glob("build_*.py")):
                add(f)
        elif path.is_file():
            add(path)
    include = [p for p in include if not p.endswith("main.pdf")]
    OUT.write_text("\n".join(sorted(set(include))) + "\n")
    print(f"manifest: {len(include)} files -> {OUT}")


if __name__ == "__main__":
    main()