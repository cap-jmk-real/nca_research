Anonymised supplementary bundle for TMLR double-blind review
============================================================

This ZIP is submitted with the manuscript. It contains:
  - Result JSON/NPZ files, table generators, Lean sources, figures
  - Stage-0 training/evaluation Python modules (no author home-directory paths)
  - stats_public.json (Hermes dataset pin; local disk paths removed)

Rebuild from a full checkout:
  cd paper && make -f tmlr/Makefile supplement-zip

CI runs scripts/check_supplement_anonymity.py on every file in the manifest.

Licence: MIT for code; result files as in the paper data statement.
After acceptance, the public GitHub repository link is restored in the camera-ready PDF.