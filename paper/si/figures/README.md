# SI figures

Captions and SSD paths. Export plots into this folder (or keep only paths) when preparing the PDF.

| Fig | Caption (draft) | Source |
| --- | --- | --- |
| S1 | Curated val top-1 vs step for mix13 NCA vs flat | `results/logs/text_nca_1d_mix13.jsonl` / TensorBoard `tb/` |
| S2 | Equal-budget ablation curves (150k) | `results/hypothesis_tests/*_history.json` |
| S3 | Regeneration: clean vs damaged top-1 vs update depth \(k\) | hypotheses `regeneration` block in EXP record |
| S4 | Local Hermes confusion / per-label recall vs Qwen1.5B | `local_next_tool_qwen25_1_5b.json` + NCA local eval |

Generate TensorBoard screenshots with `scripts/tensorboard.sh` when writing the camera-ready SI.
