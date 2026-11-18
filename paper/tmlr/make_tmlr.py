#!/usr/bin/env python3
"""Derive the TMLR-format source (paper/main_tmlr.tex) from paper/main.tex.

Body text, tables and numbers are untouched (they stay generated from result files).
Only the preamble, title block, bibliography style hooks and one internal appendix
(the change ledger vs. an earlier internal version) are changed.
Run from paper/:  python3 tmlr/make_tmlr.py
Build (from paper/): tectonic -X compile -Z search-path=tmlr main_tmlr.tex
"""
import re, pathlib
src = pathlib.Path("main.tex").read_text(encoding="utf-8")
lines = src.split("\n")
pre_end = next(i for i, l in enumerate(lines) if l.startswith("\\begin{document}"))
pre, body = "\n".join(lines[:pre_end]), "\n".join(lines[pre_end:])

# ---- preamble: keep packages/macros, drop two-column layout hacks -------------
pre = re.sub(r"\\documentclass\[[^\]]*\]\{article\}", "", pre)
pre = re.sub(r"\\usepackage\[margin=[^\]]*\]\{geometry\}", "", pre)
for pkg in ("cite", "inputenc", "orcidlink"):
    pre = re.sub(r"\\usepackage(\[[^\]]*\])?\{%s\}" % pkg, "", pre)
pre = re.sub(r"\\setlength\{\\(textfloatsep|floatsep|intextsep|dbltextfloatsep|dblfloatsep|abovecaptionskip|belowcaptionskip)\}[^\n]*", "", pre)
pre = re.sub(r"\\makeatletter\n\\renewcommand\{\\@seccntformat\}.*?\\makeatother", "", pre, flags=re.S)
pre = re.sub(r"\\setcounter\{dbltopnumber\}\{2\}\n(\\renewcommand\{\\[a-z]*fraction\}\{[0-9.]*\}\n?)*", "", pre)
pre = re.sub(r"\\title\{.*?\}\n\\author\{.*?\n\}\n\\date\{[^}]*\}", "", pre, flags=re.S)
pre = re.sub(r"\\hypersetup\{.*?\n\}", "\\\\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}", pre, flags=re.S)
pre = pre.replace("\\usepackage{hyperref}", "\\usepackage{hyperref}")

head = r"""\documentclass[10pt]{article}
% TMLR submission build. Default = anonymous review version.
% Non-anonymous (preprint / camera-ready) build: compile main_tmlr_preprint.tex.
\ifdefined\NONANON
  \usepackage[preprint]{tmlr}
\else
  \usepackage{tmlr}
\fi
\usepackage[utf8]{inputenc}
\setcitestyle{numbers,square,comma}
"""
title = r"""
\title{Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection}
\ifdefined\NONANON
\author{\name Julian M. Kleber \email julian.m.kleber@gmail.com \\
        \addr Independent Researcher \\
        ORCID: 0000-0001-5518-0932}
\def\month{MM}\def\year{YYYY}\def\openreview{\url{https://openreview.net/forum?id=XXXX}}
\newcommand{\repolink}{\url{https://github.com/cap-jmk-real/nca_research}}
\else
\author{Anonymous authors}
\def\month{MM}\def\year{YYYY}\def\openreview{\url{https://openreview.net/forum?id=XXXX}}
\newcommand{\repolink}{\textit{[repository URL withheld for double-blind review; the public repository and its MIT licence are added in the camera-ready version]}}
\fi
"""

# ---- body -----------------------------------------------------------------
# title block: replace the twocolumn wrapper by plain maketitle + abstract
m = re.search(r"\\twocolumn\[\s*\\begin\{@twocolumnfalse\}.*?\\end\{@twocolumnfalse\}\s*\]", body, flags=re.S)
blk = m.group(0)
abs_ = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", blk, flags=re.S).group(0)
abs_ = abs_.replace("\\vspace{-1.0em}\n", "").replace("\\noindent ", "")
body = body.replace(blk, "\\maketitle\n" + abs_)
# repository URL -> macro (anonymous in review build)
body = body.replace("\\url{https://github.com/cap-jmk-real/nca_research} (MIT licence)", "\\repolink{} (MIT licence)")
# drop the internal change-ledger half of the last appendix (mentions earlier internal/preprint versions)
body = body.replace("\\section{Result Sources and Changes from Version 7.6}\\label{app:ledger}", "\\section{Result Sources}\\label{app:ledger}")
body = re.sub(r"Table~\\ref\{tab:sources\} names the result sources used in this paper\..*?exact file and key for every entry\.",
              r"Table~\\ref{tab:sources} names the result sources used in this paper.", body, flags=re.S)
body = re.sub(r"\\begin\{table\*\}\[p\]\n\\centering\n\\scriptsize\n\\setlength\{\\tabcolsep\}\{3pt\}\n\\caption\{Ledger of Version.*?\\end\{table\*\}\n", "", body, flags=re.S)
body = body.replace(" Version~7.6 of this manuscript claimed more than the result files support; Appendix~\\ref{app:ledger} lists what was removed or replaced.", "")
assert "tab:ledger" not in body and "Version~7.6" not in body, "ledger removal failed"
assert "github.com" not in body.replace("\\repolink", ""), "repo URL still in body"
out = head + pre.strip() + "\n" + title + body
pathlib.Path("main_tmlr.tex").write_text(out, encoding="utf-8")
pathlib.Path("main_tmlr_preprint.tex").write_text("\\def\\NONANON{1}\n\\input{main_tmlr.tex}\n", encoding="utf-8")
print("wrote main_tmlr.tex")
