# TMLR submission package (draft, NOT submitted)

Consolidated package. It replaces the two duplicate packages of PR #13 (`paper/main_tmlr.*`, this folder) and PR #14 (`submission/tmlr/`). The manuscript is `paper/main.tex` of the five-seed re-analysis (PR #11 lineage). PR #14's manuscript (the v7.7 text of PR #12) is not used.

| File | What |
|---|---|
| `../main_tmlr.tex` / `../main_tmlr.pdf` | anonymous review build, TMLR style (what would be uploaded) |
| `../main_tmlr_preprint.tex` / `../main_tmlr_preprint.pdf` | same text, named (`[preprint]` option), for arXiv |
| `make_tmlr.py` | regenerates both `.tex` files from `paper/main.tex` |
| `tmlr.sty`, `fancyhdr.sty`, `tmlr.bst` | TMLR style files (Apache-2.0, `TMLR_STYLE_LICENSE_APACHE2.txt`) from <https://github.com/JmlrOrg/tmlr-style-file> |
| `COVER_LETTER.md` | cover note and OpenReview form fields (draft) |
| `AUTHOR_INFO.md` | author and form data |
| `VENUES.md` | verified venue shortlist, including the entries ported from PR #14 |

Build (from `paper/`): `python3 tmlr/make_tmlr.py && tectonic -X compile -Z search-path=tmlr main_tmlr.tex`

CI (`.github/workflows/paper-build.yml`) reruns the five table generators and fails if any macro or table in `paper/generated/` changes, checks that `main_tmlr.tex` is the current `make_tmlr.py` output, builds the three PDFs, fails on overfull boxes or undefined references in the review build, and greps the review PDF text and metadata for identifying strings.

## What `make_tmlr.py` changes against `main.tex`

1. Preamble: TMLR style instead of the two-column `article` layout.
2. Citations: natbib author-year as the template defines it, with one label per bibitem. The bibliography is sorted by label.
3. Title block anonymised in the review build. The repository URL is a macro that shows a placeholder for the anonymized mirror in the review build.
4. The "Use of AI tools" paragraph becomes a footnote on the first page (TMLR FAQ). Wording pending Julian's OK.
5. The internal change ledger appendix and the intro sentence pointing to it are removed, because they refer to earlier public versions.
6. Two appendix tables get a smaller column separation so they fit the 6.5 in text block.

## Open before submission (needs Julian)

- His yes to the venue, the errata list, the AI-use wording, the data statement on the private sessions, and the cover letter.
- His own OpenReview profile and the submit click.
- The anonymized mirror URL (built after this PR and the hygiene PR are merged).
- Length: about 17 pages of main text (references start on page 18). TMLR asks 4-week reviews above 12 pages. No cuts made.
