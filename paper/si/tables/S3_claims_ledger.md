# SI Table S3 — Claims ledger

| ID | Status | Supported claims | Not supported |
| --- | --- | --- | --- |
| `EXP-20261003-001` | primary | 1D TextNCA beats flat linear readout on curated Hermes val at 150k steps<br>Always-terminal is stronger than NCA on held-out local Hermes sessions | Local Hermes generalization without prior matching |
| `EXP-20261004-001` | primary | Closed-set Qwen2.5-1.5B greedy next-tool top-1 is far below always-terminal and below TextNCA on the same local rows | — |
| `EXP-20261005-001` | primary | Verifier-correct FineEnvs rows can be staged into the 13 Hermes labels | — |
| `EXP-20261005-002` | smoke | At 8k steps, cellular baseline beats flat on curated val<br>Long-range goal edge slightly beats baseline at 8k | 8k-step carry and halt do not beat baseline on reset-val metric<br>8k results are not final architectural verdicts (need ≥150k) |
