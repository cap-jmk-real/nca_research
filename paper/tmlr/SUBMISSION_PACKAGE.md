# TMLR submission package notes (draft — not submitted)

## Builds

From `paper/`:

```bash
python3 tmlr/make_tmlr.py
tectonic -X compile -Z search-path=tmlr main_tmlr.tex          # anonymous review PDF
tectonic -X compile -Z search-path=tmlr main_tmlr_preprint.tex  # named preprint
make -f tmlr/Makefile supplement-zip                           # anonymised ZIP (≤100 MB)
tectonic -X compile -Z search-path=tmlr tmlr/supplement.tex     # supplementary PDF
```

## Anonymity

- **Review PDF** (`main_tmlr.pdf`): `\repolink` points reviewers to the submitted supplementary ZIP, not the public GitHub URL.
- **Supplementary ZIP**: must pass `scripts/check_supplement_anonymity.py` (no author paths or account names). After acceptance, restore the public repository URL in the camera-ready PDF.

## Licences (camera-ready)

| Artifact | Licence |
|---|---|
| Manuscript at TMLR submission | CC BY 4.0 (TMLR default on acceptance; applies to the submitted PDF) |
| Code repository | MIT (as stated in the paper) |

## OpenReview / profile

Julian must complete his OpenReview profile and submit; this package does not touch OpenReview fields beyond `tmlr/COVER_LETTER.md` and `tmlr/AUTHOR_INFO.md` drafts.