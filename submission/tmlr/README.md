# TMLR submission package (NCA / GNCA micro-routing paper)

**Status: prepared, NOT submitted.** No OpenReview account was created and nothing was uploaded.

| File | Purpose |
|------|---------|
| `main.tex` | Anonymous (double-blind) manuscript in the TMLR style, generated from `paper/main.tex` by `make_tmlr.py` |
| `nca_tmlr_anonymous_review.pdf` | Compiled review PDF (tectonic 0.15.0, 33 pages: body to p. 30, references p. 31, appendix pp. 32-33) |
| `tmlr.sty` | Official TMLR style file (copied unmodified from https://github.com/JmlrOrg/tmlr-style-file, last commit 7bf90ef, 2023-06-30) |
| `figures/` | The five PDF figures used by `main.tex` |
| `make_tmlr.py` | Reproducible converter. `python3 submission/tmlr/make_tmlr.py` rewrites `main.tex`; add `--accepted` for the camera-ready (non-anonymous) variant `main_accepted.tex` |
| `cover_letter.md` | Draft text for the OpenReview submission fields / comments to the Action Editor |

Build: `cd submission/tmlr && tectonic -X compile main.tex` (or `latexmk -pdf main.tex`).

## What the converter changes (and what it does not)
Changed: preamble (TMLR style, no `geometry`, `cite`, custom section formats), title/author block (anonymous), abstract placement, `\cite` -> `\citep`/`\citet` with authoryear bib labels (14 bibitems), starred floats -> plain floats, `\twocolumn[...]` appendix header, `\footnotesize` on wide tables, and the two repository/dataset links (withheld for review).
Not changed: every sentence, number, table, and figure of the scientific body.

## Known formatting leftovers
* One table (`tab:feature-eval`) is still 31.9 pt wider than the text block (Overfull hbox in `main.log`); fix by wrapping the first column in `p{...}` before camera-ready.
* Body length is 30 pages single-column. TMLR accepts any length, but the FAQ says reviews are requested within 2 weeks for up to 12 pages of main content and 4 weeks above that.
