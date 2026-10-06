# SI Table S2 — Next-tool top-1 (primary / preliminary)

Smoke runs are listed but **not** for main-text claims.

| ID | Status | Setting | NCA / model | Flat | Always-terminal | Local NCA | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `EXP-20261003-001` | primary | 1D TextNCA mix13 primary 150k (Hermes tools) | 65.18% | 60.71% | 54.30% | 40.09% | step 26000 |
| `EXP-20261004-001` | primary | Baseline Qwen2.5-1.5B-Instruct local Hermes next | 6.29% | — | 49.47% | 6.29% | Qwen local baseline |
| `EXP-20261005-001` | primary | 1D TextNCA FineEnvs-prior continuation 150k | 58.41% | 49.54% | 54.30% | 40.33% | step 60000 |
| `EXP-20261005-002` | smoke | ablation `baseline` @ 8000 | 63.77% | — | — | 28.83% | best step 5500 |
| `EXP-20261005-002` | smoke | ablation `halt` @ 8000 | 63.29% | — | — | 33.69% | best step 6500 |
| `EXP-20261005-002` | smoke | ablation `long_range` @ 8000 | 64.00% | — | — | 18.15% | best step 6500 |
| `EXP-20261005-002` | smoke | ablation `carry` @ 8000 | 62.03% | — | — | 37.60% | best step 5000 |
| `EXP-20261005-002` | smoke | ablation `flat` @ 8000 | 62.96% | — | — | 42.94% | best step 4000 |
| `EXP-20261006-001` | primary | ablation `long_range` @ 150000 | 63.55% | — | — | 23.01% | best step 10000 |
| `EXP-20261006-001` | primary | ablation `baseline` @ 150000 | 62.60% | — | — | 25.86% | best step 10000 |
| `EXP-20261006-001` | primary | ablation `flat` @ 150000 | 62.49% | — | — | 37.37% | best step 15000 |
| `EXP-20261006-002` | primary | ablation `long_range` @ 150000 | 63.55% | — | — | 23.01% | best step 10000 |
| `EXP-20261006-002` | primary | ablation `baseline` @ 150000 | 62.60% | — | — | 25.86% | best step 10000 |
| `EXP-20261006-002` | primary | ablation `flat` @ 150000 | 62.49% | — | — | 37.37% | best step 15000 |
| `EXP-20261006-003` | primary | ablation `carry` @ 150000 | 62.77% | — | — | 39.03% | best step 5000 |
| `EXP-20261006-003` | primary | ablation `baseline` @ 150000 | 62.60% | — | — | 25.86% | best step 10000 |
| `EXP-20261006-004` | primary | ablation `carry` @ 150000 | 62.77% | — | — | 39.03% | best step 5000 |
| `EXP-20261006-004` | primary | ablation `baseline` @ 150000 | 62.60% | — | — | 25.86% | best step 10000 |
| `EXP-20261006-005` | primary | ablation `halt` @ 150000 | 62.67% | — | — | 29.77% | best step 10000 |
| `EXP-20261006-005` | primary | ablation `baseline` @ 150000 | 62.60% | — | — | 25.86% | best step 10000 |
| `EXP-20261006-006` | primary | ablation `halt` @ 150000 | 62.67% | — | — | 29.77% | best step 10000 |
| `EXP-20261006-006` | primary | ablation `baseline` @ 150000 | 62.60% | — | — | 25.86% | best step 10000 |
