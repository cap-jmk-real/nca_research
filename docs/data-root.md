# Data root

Large datasets, Hugging Face caches, and training artifacts stay off the internal disk.

| | Path |
| --- | --- |
| Volume | `/Volumes/Extreme SSD` |
| Data root | `/Volumes/Extreme SSD/nca_research` |
| Datasets | `/Volumes/Extreme SSD/nca_research/datasets` |
| Results | `/Volumes/Extreme SSD/nca_research/results` |

`scripts/curate_traces.py` defaults to this data root. Override with `NCA_DATA_ROOT` (must be a directory on `/Volumes/`, and not `Macintosh HD`). Training, the auto-research daemon, and `scripts/env.sh` refuse a data root whose device is the internal disk, including a `/Volumes/Extreme SSD` folder created while the drive is unmounted. They do not fall back to `results/` inside this repo. `scripts/env.sh` replaces an inherited internal `TMPDIR`, `HF_HOME`, `TORCH_HOME`, or pip cache with the Results directory above. The script sets these before any Hugging Face download:

- `HF_HOME=/Volumes/Extreme SSD/nca_research/datasets/hf_home`
- `HF_HUB_CACHE` and `HF_XET_CACHE` under that `HF_HOME`
- `HF_DATASETS_CACHE=/Volumes/Extreme SSD/nca_research/datasets/hf_datasets_cache`

Curated train and validation jsonl are written to `/Volumes/Extreme SSD/nca_research/datasets/curated/`. The repo keeps only `data/curated/stats.json` and a small `data/curated/sample.jsonl`.

Chosen 2026-10-03. `df -h` showed 411 GiB free on Extreme SSD (932 GiB volume, 520 GiB used). `SanDisk Unlocker` had no free space (3.3 MiB helper volume) and was ignored. `Macintosh HD` is the system volume and was ignored.
