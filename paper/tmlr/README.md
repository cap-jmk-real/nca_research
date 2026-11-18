# TMLR submission package (draft, NOT submitted)

Source: `paper/main.tex` at head `0964149` of `research/rigour-novelty-1110` (PR #11, draft). This branch only adds the TMLR packaging; the manuscript text and generated tables are unchanged except for the items listed below.

| File | What |
|---|---|
| `../main_tmlr.tex` / `../main_tmlr.pdf` | anonymous review build, TMLR style (what would be uploaded) |
| `../main_tmlr_preprint.tex` / `../main_tmlr_preprint.pdf` | same text, non-anonymous (`[preprint]` option): author, ORCID, repository link |
| `make_tmlr.py` | regenerates both `.tex` files from `paper/main.tex` |
| `tmlr.sty`, `fancyhdr.sty`, `tmlr.bst` | TMLR style files (Apache-2.0, `TMLR_STYLE_LICENSE_APACHE2.txt`) from <https://github.com/JmlrOrg/tmlr-style-file> |
| `COVER_LETTER.md` | cover letter draft |
| `AUTHOR_INFO.md` | author and form data |
| `VENUES.md` | verified shortlist, exclusions, and what changes for other venues |

Build (from `paper/`): `python3 tmlr/make_tmlr.py && tectonic -X compile -Z search-path=tmlr main_tmlr.tex`

## Changes against `main.tex` (all in `make_tmlr.py`)

1. Preamble: TMLR style instead of the two-column `article` layout; numeric citations kept via `\setcitestyle{numbers}`.
2. Title block anonymised in the review build; the repository URL is a macro that is empty (explained in text) in the review build.
3. Removed the internal change ledger (appendix "Changes from Version 7.6") and the intro sentence that points to it, because they refer to earlier public versions and would de-anonymise the paper. The result-source table (S1-S8) stays as appendix "Result Sources". The cover letter discloses that earlier public preprint versions overclaimed.

## Open before submission

- Julian's own yes to the venue and to the submit step; Julian creates the OpenReview profile.
- Build is clean for undefined references; remaining TeX warnings are a few overfull/underfull boxes and duplicate hyperref anchor names (cosmetic). The page count is 26 (main text about 17 pages).
- The repository-side numbers are those of PR #11 (unmerged draft); if that PR changes, rerun `make_tmlr.py`.
- Self-review score in the repo for this text: paper-review 9.0 (novelty 7.0), klaut 10.0 with 0 findings (PR #11 body); not re-run here.
