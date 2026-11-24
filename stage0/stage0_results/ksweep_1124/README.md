# ksweep_1124: recurrence-depth sweep of the tuned tree GNCA (source S8b)

- Producer: `stage0/run_ksweep_1124.py` at nca_research b65c102, run on the engine host (RTX 3060, torch, 2 threads)
  on 2026-11-24 11:12-11:51 CET, in two passes (K = 1,2,8,16, then K = 4,6). Selected configurations read from the
  rigour_1110 selection (`RG.chosen(bench, "gnca_tree")`); only K changes. Seeds 0-4; test split scored once.
- `ksweep_summary.json`: per (benchmark, K, seed) val/test top-1, parameter count, wall time, device.
- `ksweep_correct.npz`: per-turn test correctness, arrays of shape (5 seeds, n_test) keyed `<bench>/K<k>`.
- `stats_ksweep.json`: paired bootstrap vs the re-run of the selected K, Holm over 20 pairs, DerSimonian-Laird pooling,
  and the re-run-vs-rigour-final reproducibility check. Rebuild: `python scripts/build_ksweep_1124.py`.
