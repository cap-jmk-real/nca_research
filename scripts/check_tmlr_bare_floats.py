#!/usr/bin/env python3
"""Fail if main_tmlr.tex has a bare decimal in a results paragraph.

Hyperparameters, software versions, layout lengths, Holm/coverage thresholds and
search ranges are allowlisted. Result decimals must use \\num... or \\bf... macros
from paper/generated/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "paper" / "main_tmlr.tex"

# Whole-line skips
LINE_SKIP = re.compile(
    r"\\(definecolor|usepackage|bibitem|includegraphics|setlength|newcommand|input|"
    r"lstinputlisting|footnotesize|scriptsize|fontsize|hypersetup|ifdefined|else|fi|fi)\b"
    r"|\\begin\{(thebibliography|itemize|tabular|tabularx|document)\}"
    r"|\\end\{(thebibliography|itemize)\}"
    r"|leftmargin|itemsep|tabcolsep|textwidth|columnwidth|arraybackslash"
    r"|arXiv|doi\.org"
)

# Decimal allowed in-line (thresholds, versions, hyperparams, layout)
INLINE_ALLOW = re.compile(
    r"(Lean~4\.\d+|bfcl-eval|Qwen3\.5|Gemma-4|"
    r"learning rate|weight decay|label smoothing|dropout|depth \d+|maximum depth|"
    r"learning rate 0\.|presence penalty|temperature |Holm|coverage|p<0\.05|"
    r"\\theta\\in|threshold|\\beta\{=\\}|tabcolsep|0\.\d+\\,\\textwidth|"
    r"width=0\.|fontsize|pt\}\{|rows initialised|softmax.*\\kappa|"
    r"contractive variant|update \$0\.25|spectrally normalised|"
    r"Appendix~|search range|rate \$\\\[|decay \$\\\[|below 0\.5 on)"
)


def body_lines(text: str) -> list[tuple[int, str]]:
    if "\\begin{document}" not in text:
        return []
    body = text.split("\\begin{document}", 1)[1]
    if "\\appendix" in body:
        body = body.split("\\appendix", 1)[0]
    if "\\begin{thebibliography}" in body:
        body = body.split("\\begin{thebibliography}", 1)[0]
    out: list[tuple[int, str]] = []
    start = text.index("\\begin{document}") + len("\\begin{document}")
    for i, line in enumerate(body.splitlines()):
        n = start + i + 1
        raw = line.split("%", 1)[0]
        if not raw.strip():
            continue
        out.append((n, raw))
    return out


def main() -> int:
    text = TEX.read_text(encoding="utf-8")
    bad: list[str] = []
    float_re = re.compile(r"(?<![\\$A-Za-z_/])(\d+\.\d+)(?![A-Za-z%])")
    for lineno, line in body_lines(text):
        if LINE_SKIP.search(line):
            continue
        if "\\num" in line or "\\bf" in line:
            # still flag if a bare float remains outside macros
            stripped = re.sub(r"\\(?:num|bf)[A-Za-z0-9]+\{[^{}]*\}", "", line)
        else:
            stripped = line
        for m in float_re.finditer(stripped):
            ctx = line.strip()
            if INLINE_ALLOW.search(ctx):
                continue
            if "Figure~" in ctx or "Table~" in ctx and "0.05" in m.group(1):
                continue
            bad.append(f"line {lineno}: {m.group(1)} in {ctx[:120]}")
    if bad:
        print("Bare result floats in main_tmlr.tex (use paper/generated/ macros):\n", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"check_tmlr_bare_floats: ok ({TEX.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())