#!/usr/bin/env python3
"""Derive the TMLR-format source (paper/main_tmlr.tex) from paper/main.tex.

Citations are converted to the template's natbib author-year style (labels below).

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
\newcommand{\repolink}{\textit{[anonymized code mirror: URL inserted when the mirror is built; the same files are in the supplementary ZIP]}}
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

# ---- citations: TMLR template style is natbib author-year (tmlr.sty) -------------
body = body.replace("\\cite{", "\\citep{")
LABELS = {
 "mordvintsev2020growing": "Mordvintsev et~al.(2020)", "grattarola2021learning": "Grattarola et~al.(2021)",
 "gala2023equivariant": "Gala et~al.(2024)", "etcheverry2026reasoning": "Etcheverry et~al.(2026)",
 "pio2026brainca": "Pio-Lopez et~al.(2026)", "mittal2026textnca": "Mittal et~al.(2026)",
 "lee2026training": "Lee et~al.(2026)", "li2016ggnn": "Li et~al.(2016)", "gilmer2017mpnn": "Gilmer et~al.(2017)",
 "dehghani2019universal": "Dehghani et~al.(2019)", "bai2019deq": "Bai et~al.(2019)",
 "vaswani2017attention": "Vaswani et~al.(2017)", "cover1967nearest": "Cover and Hart(1967)",
 "geifman2017selective": "Geifman and El-Yaniv(2017)", "kontschieder2015deep": "Kontschieder et~al.(2015)",
 "frosst2017distilling": "Frosst and Hinton(2017)", "yang2018deep": "Yang et~al.(2018)",
 "chen2023frugalgpt": "Chen et~al.(2024)", "ong2024routellm": "Ong et~al.(2025)",
 "singla2026agentweave": "Singla et~al.(2026)", "li2026fastmodels": "Li(2026)",
 "chen2024octopus": "Chen and Li(2024)", "erdogan2024tinyagent": "Erdogan et~al.(2024)",
 "lin2024hammer": "Lin et~al.(2025)", "chen2026oats": "Chen et~al.(2026)", "qu2024colt": "Qu et~al.(2024)",
 "patil2023gorilla": "Patil et~al.(2024)", "gorilla-bfcl": "Patil et~al.(2025)", "qin2023toolllm": "Qin et~al.(2024)",
 "li2023apibank": "Li et~al.(2023)", "hermestraces": "{Hermes traces}(2026)", "chung1997spectral": "Chung(1997)",
 "katz2019marabou": "Katz et~al.(2019)", "wang2021beta": "Wang et~al.(2021)",
 "katz2017reluplex": "Katz et~al.(2017)", "bentkamp2017expressiveness": "Bentkamp et~al.(2017)",
 "chow1970optimum": "Chow(1970)", "breiman2001random": "Breiman(2001)", "friedman2001greedy": "Friedman(2001)", "hendrycks2017baseline": "Hendrycks and Gimpel(2017)",
}
bib_m = re.search(r"(\\begin\{thebibliography\}\{[^}]*\}[^\n]*\n)(.*?)(\\end\{thebibliography\})", body, flags=re.S)
items = re.findall(r"\\bibitem\{([^}]+)\}([^\n]*(?:\n(?!\\bibitem|\\end\{thebibliography\})[^\n]*)*)", bib_m.group(2))
missing = [k for k, _ in items if k not in LABELS]
assert not missing, f"no author-year label for {missing}"
items.sort(key=lambda kv: re.sub(r"[^a-z0-9]", "", LABELS[kv[0]].lower()))
new_bib = "".join("\\bibitem[%s]{%s}%s\n" % (LABELS[k], k, t.rstrip("\n")) for k, t in items)
body = body[:bib_m.start(2)] + new_bib + body[bib_m.end(2):]

# ---- LLM-use footnote on the first page (TMLR FAQ). Wording approved by Julian (2026-11-18, 09:01); keep verbatim. ----
AI_NOTE = ("The author used AI coding, research and writing assistants (Cursor, Grok Bot, Hermes Agent, Antigravity and SciSpace) to write code, run experiments, search and read literature, and draft and edit text. The research questions, study design and conclusions are the author's. The author checked every number against the released result files and every citation against its source, and takes full responsibility for the content.")
# the footnote replaces the body paragraph "Use of AI tools" in the TMLR builds (no duplicate statement)
body, n_ai = re.subn(r"\\paragraph\{Use of AI tools\.\}[^\n]*\n", "", body)
assert n_ai == 1, "AI-use paragraph not found"
body = body.replace("\\end{abstract}", "\\footnote{" + AI_NOTE + "}\n\\end{abstract}", 1)
# two appendix tables were laid out for the wider two-column page (overfull in the 6.5in TMLR text block)
body, n1 = re.subn(r"(\\section\{Default Configurations and the Single-Seed Search\}\\label\{app:earlier\}\n\\begin\{table\*\}\[t\]\n\\centering\n)\\small", r"\1\\footnotesize\\setlength{\\tabcolsep}{2.5pt}", body)
body, n2 = re.subn(r"(\\scriptsize\n)\\setlength\{\\tabcolsep\}\{3\.5pt\}(\n\\caption\{Tuned models: test top-1)", r"\1\\setlength{\\tabcolsep}{2.5pt}\2", body)
assert n1 == 1 and n2 == 1, (n1, n2)
assert "tab:ledger" not in body and "Version~7.6" not in body, "ledger removal failed"
assert "github.com" not in body.replace("\\repolink", ""), "repo URL still in body"
out = head + pre.strip() + "\n" + title + body
pathlib.Path("main_tmlr.tex").write_text(out, encoding="utf-8")
pathlib.Path("main_tmlr_preprint.tex").write_text("\\def\\NONANON{1}\n\\input{main_tmlr.tex}\n", encoding="utf-8")
print("wrote main_tmlr.tex")
