# Input-parity audit: next-tool classification (stage0), 2026-10-26

Code paths: `~/nca_research/stage0` and `~/nca_research/nca_text/data.py` on engine; LLM prompts in `/workspace/nca_global/run.py` and `/workspace/nca_sol/sol_teacher.py` on the box.
Data: each curated row is capped at goal ≤500 and result ≤500 characters (`scripts/curate_traces.py:31-32`). On the curated test set (3,530 rows; 3,339 have at least one step), the last result has a median of 141 characters and 13 whitespace tokens. 19.6% of last results are longer than 48 tokens.

| Model | Goal | Previous tools | Last tool-result text | Older results | Representation of the result text | Source |
|---|---|---|---|---|---|---|
| NCA-1D, GraphNCA, GRU, tiny transformer (`tx`) | first 48 tokens → 48-d crc32 hash bag | last 4, one-hot | **yes**, but only the first 48 lowercased whitespace tokens, hashed together with the tool name into a 48-d unigram bag (L2-normalised) | yes, steps 1–4 handled the same way | ~48 buckets with heavy collisions, no bigrams, no IDF, no length or error feature | `train_eval.py:65-66`, `data.py:304-316` (hash_features, `split()[:48]`), `data.py:601-626` (build_trace_obs, HASH_DIM=48, TRACE_STEPS=4) |
| NCA-2D (`nca2d`) | 64-d hash | last 5, one-hot | same, but 64-d (HASH_C·GRID_W = 8·8) | last 5 | as above | `data.py:318-340`, `data.py:20-22` |
| Flat MLP (`flat`) | same input as NCA-1D, flattened | last 4 | same as NCA-1D | yes | as above | `train_eval.py:69-70` (FlatPolicy on build_trace_obs) |
| HistGB (`hgb`) | full goal (≤500 chars) | last 4, one-hot, plus count | **yes, first 200 chars** of the **last** step only | **no** | TF-IDF 1–2-grams (50k vocab, min_df 2, sublinear) → SVD-128, plus scalars log(len(last result)), "error" in last result, log(len goal) | `sota_v2.py:16-23` (feats), `:27` (text, `[:200]`), `:29-33`; same in `sota_gbdt.py:38` |
| HistGB, no result (`hgb_noresult`) | full goal | same | text **no**; the scalars len(last result) and the error flag remain (leak of result info) | no | goal TF-IDF only | `sota_v2.py:21-22, 27, 29` |
| Logistic regression | full goal | same | yes, 200 chars | no | sparse full TF-IDF (no SVD) + same scalars | `sota_v2.py:35-37` |
| Sol (teacher) | **first 160 chars** | last 6 tool names | **yes, first 100 chars**, last step only | no | raw text in the prompt, few-shot shows only tool sequences | `nca_sol/sol_teacher.py:18-21, 29` |
| DeepSeek V4.1 Flash, Maverick | via `sol_teacher.body` → same as Sol: 160-char goal, **100-char** result | same | yes, 100 chars | no | raw text | `nca_global/run.py:2, 39, 41` |
| (old teacher prompts, not in the headline) | 500 / 300 chars | last 4 | 300 / 200 chars of each step | yes | raw text | `score_teachers.py:13-16`, `score_teachers_v2.py:28-29`; local v2c: 160/100 like Sol (`score_local_v2c.py:18-19`) |

## Verdict
The inputs were **not matched**. The comparison is not fair to the neural models or to the LLMs.
1. All neural models (NCA, flat, GRU, tx) do see the last result text, plus older results, which HistGB does not get. But they see it only as a 48-d hash bag of at most 48 tokens, mixed with the tool name. HistGB gets 200 characters as TF-IDF bigrams → SVD-128 plus explicit length and error features. The representation is much richer, so the 2.8-point gap may come from the features rather than the architecture.
2. `hgb_noresult` still sees result information through the scalars len(last result) and the "error" flag. The "removing the result text barely hurts HistGB" ablation is therefore not clean.
3. The LLMs get less than HistGB: goal 160 vs ≤500 chars, result 100 vs 200 chars.
4. Logistic regression gets the same text as HistGB, so the logreg ≈ NCA tie is not affected.

## Change spec for the cloud agent (repo nca_research, PR only, no merge)
Goal: every model gets the **same** text input: goal ≤500 chars plus the last tool result, first 200 chars, plus the same scalars.
1. New module `stage0/features.py`: one shared function `shared_text(ex)` = goal + " || " + last result[:200], identical to `sota_v2.text(e, True)`, and `shared_scalars(ex)` = the 4 scalars from `sota_v2.feats`. Change sota_v2/sota_gbdt to use it as well (results must stay bit-identical; add a test for this).
2. Neural input variant `--text_feat {hash48 (default, old), tfidf_svd128}`. For `tfidf_svd128`, fit TF-IDF (1–2-gram, 50k, min_df 2, sublinear) + SVD-128 **on train only** per seed, attach the 128-d vector plus 4 scalars as an extra conditioning cell or channel in build_trace_obs (goal cell), keep the tool one-hots and step cells. Cache the features per split/seed under `~/nca_data/stage0_feats/`.
3. Optional third variant `--text_feat hash_wide`: hash_features with dim 512, tokens up to 200 chars, unigrams + bigrams (tests capacity separately from TF-IDF).
4. Fix the ablation: `hgb_noresult` also without len(result) and error flag (`hgb_noresult_clean`); keep the old variant only for reference.
5. LLM parity (prompt change only, **no new run without Julian's OK and budget**): `body()` variant with goal[:500] and result[:200], as a separate function; do not touch the running Azure jobs.
6. Runs: archs nca1d, graph, flat, gru, tx (nca2d optional) × text_feat {hash48, tfidf_svd128} × seeds {0,1,2}, same hyperparameters/selection as the current best runs (val only), test scored once. Evaluate on curated test and hermes_ood/test (1,208 rows).
7. Analysis: mean ± sample std over 3 seeds; paired bootstrap 10k over rows (seed-averaged correctness) for (a) each neural model tfidf vs hash48, (b) best neural tfidf vs HistGB, (c) vs the floors 52.9% / Hermes floor. Write per-row predictions to `rowpreds_<date>/`.
8. Append to the queue file only (append-only, IDs = line+hash), run on engine via queue.sh, do not interrupt q_train/Hermes FT. Restic backup of results, tag paper_src.

Runtime estimate (engine RTX 3060): previous curated runs took ~3–20 min each (run dirs 03:05–07:31). 5 archs × 2 variants × 3 seeds = 30 runs, of which 15 are new (hash48 already exists for most of them) → ~4–6 h GPU, plus ~15 min CPU for the TF-IDF/SVD caches and ~10 min for HistGB/analysis. With hash_wide, +2–3 h. If the GPU is shared with the Hermes FT, expect up to 2×.
