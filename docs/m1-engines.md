# M1 engines for the text NCA

**Date:** 2026-10-03. Measured on this machine, not copied from a leaderboard.

## Machine

`sysctl` on this Mac: `hw.optional.arm64=1`, `machdep.cpu.brand_string=Apple M1`, `hw.memsize=17179869184` (16 GiB), `hw.ncpu=8` (4 performance + 4 efficiency). PyTorch 2.13.0, MPS available. No teacher model was loaded. Engine (`192.168.10.29`, the 3060 running Ollama `qwen3.5:9b`) was not used.

## What was checked

The Klaut gateway at `POST http://192.168.10.29:30487/v1/research/papers/search` answered. Records below are from that response or from pages opened the same day. Preprints are marked. Speeds from papers are their claims, not re-runs on this M1.

- **MLX** ([ml-explore/mlx](https://github.com/ml-explore/mlx) README). Apple's array framework for Apple silicon. Arrays sit in unified memory, so a CPU op and a GPU op do not copy between devices. Lazy evaluation, NumPy-like API, `mlx.nn` for training. Not installed here.
- **mlx-lm** ([ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) README). Generation and fine-tuning on top of MLX: 4-bit quantization, LoRA on quantized weights, a rotating KV cache (`--max-kv-size`), and a prefill step size. Default chat model is a 4-bit Llama 3.2 3B. This is the path for a small teacher on a Mac, not for a 6×8 conv update.
- **llama.cpp** ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) README). Metal is a listed backend. Apple silicon is targeted with ARM NEON, Accelerate, and Metal. GGUF integer quants from 1.5-bit through 8-bit. Inference runtime. No GGUF model was run.
- **Ollama.** Search of the Ollama repo shows an MLX backend beside the GGML/Metal runner ([ollama/ollama#1730](https://github.com/ollama/ollama/issues/1730)). Not timed on this Mac.
- **PyTorch MPS** ([MPS backend](https://docs.pytorch.org/docs/main/notes/mps.html), updated 2026-09-09). Training device for Macs. Metal has no `double`; `float64` cannot be placed on MPS, and `PYTORCH_ENABLE_MPS_FALLBACK` does not fix that. This run is float32. Time with `torch.mps.synchronize()` after a warmup, because MPS launches are async. Small kernels can lose to CPU launch overhead; a larger conv can go the other way. The device below was chosen by a microbench, not by that folklore.
- **Gateway, limited-memory and speculative decode.** Alizadeh et al., "LLM in a flash: Efficient Large Language Model Inference with Limited Memory," ACL 2024 (DOI `10.18653/v1/2024.acl-long.678`); the 2023 arXiv:2312.11514 record is the preprint. Svirschevski et al., SpecExec, NeurIPS 2024 (DOI `10.52202/079017-0522`). Fonal, "Cross-Family Speculative Decoding for Polish Language Models on Apple Silicon," preprint arXiv:2604.16368 (2026): the abstract says unified-memory verification often fails to pay for itself. None of these change a closed-label conv step, so none were implemented.
- **Gateway, Apple silicon runtimes.** Rathnayaka, Waschkowski, and Wesemann, "BaseRT: Best-in-Class LLM Inference on Apple Silicon via Native Metal," preprint arXiv:2607.00501 (2026). The abstract claims up to 1.56× decode throughput versus llama.cpp and up to 1.35× versus MLX on M3 and M4 Pro, not on an M1, and not measured here. Dziedzic, Nytko, and Kruzel, "Benchmarking transformer-based time series models: pytorch vs mlx on apple silicon," ECMS 2026 (DOI `10.7148/2026-0638`): the abstract reports MLX training at 1.02×–3.12× PyTorch on an M4 Pro. Chaplia and Klym, "Evaluating small quantized language models on apple silicon," 2025 (DOI `10.23939/acps2025.01.034`): the gateway abstract has no speeds, so none are repeated here.

## What this run uses

PyTorch eager on **MPS**, batch **1024**. The task is next-tool classification on a 6×8 grid (8 hash channels + one class channel per label, 8 hidden channels, 4 residual conv steps, float32). A flat linear map on the same observation tensor is the non-spatial baseline. No distillation.

Random-grid train-step microbench (20 steps after 6 warmup steps, synchronized):

| Device | Batch | ms/step | samples/sec |
| --- | ---: | ---: | ---: |
| CPU | 64 | 47.484 | 1347.8 |
| CPU | 256 | 174.881 | 1463.9 |
| CPU | 1024 | 697.002 | 1469.1 |
| MPS | 64 | 4.751 | 13470.2 |
| MPS | 256 | 15.552 | 16461.2 |
| MPS | 1024 | 60.519 | 16920.4 |

MPS at batch 1024 is the winner, about 11× the best CPU samples/sec on this grid. `torch.compile` on that same MPS batch measured 173.530 ms/step (5901.0 samples/sec) and printed Inductor's "Not enough SMs to use max_autotune_gemm mode". Training stayed eager.

Real NCA steps on the training rows (24 steps, batch 1024, weights already on device): **66.122 ms/step, 15486.5 samples/sec**. Inference over the validation grids: **55335.1 samples/sec, 0.0181 ms/example**.

## Useful result

Public data: the first 400 rows of `lambda/hermes-agent-reasoning-traces`, config `kimi`, from the datasets-server rows API (the parquet HEAD reported `x-linked-size: 508262558`, so the full file was not downloaded). 3603 decision rows, 400 sessions. Session split by CRC of the session id: **2905 train rows / 322 sessions**, **698 val rows / 78 sessions**. This is the head of the file, not a random draw of the 7646, and not the `glm-5.1` config. Targets whose names start with `browser_` were dropped.

Local `~/.hermes/state.db` is test only (read-only, transcripts not written out). 33 sessions, 867 non-browser decision rows, **848** whose label is in the 17-way train vocabulary, 19 left out. The vocabulary is `clarify`, `delegate_task`, `execute_code`, `list`, `memory`, `patch`, `process`, `read_file`, `respond`, `search_files`, `session_search`, `skill_manage`, `skill_view`, `skills_list`, `terminal`, `todo`, `write_file`.

Top-1, checkpoint matches this readout (hidden state of the whole grid, flattened):

| Split | Rows | NCA | Flat | Train-majority |
| --- | ---: | ---: | ---: | ---: |
| Public val (held-out sessions) | 698 | 269/698 (38.5%) | 273/698 (39.1%) | 135/698 (19.3%, always `terminal`) |
| Local test | 848 | 117/848 (13.8%) | 411/848 (48.5%) | 417/848 (49.2%, always `terminal`) |

On the public session split the next-tool signal is real: both models are about double the majority class. The cellular update does not beat the flat readout. On the local pilot, `terminal` is 417 of 848 rows, and neither model beats always-`terminal`. The flat model is close (411/848). The NCA overfits the public mix.

An earlier readout that pooled only the last row, same split and same seed schedule, scored 221/698 on public val and 323/848 on local test. That checkpoint was overwritten. The file on disk is the full-grid readout above. MPS is not bitwise stable; a rerun can move these counts.

Loss was sqrt-inverse-frequency cross-entropy, so top-1 is not the training objective. Sixteen epochs, AdamW, best validation checkpoint.

## How to rerun

Artifacts go to the Results path in [data-root.md](data-root.md) (`/Volumes/Extreme SSD/nca_research/results`). The loader refuses a path whose parent is `/Volumes` itself.

```bash
export NCA_DATA_ROOT="/Volumes/Extreme SSD/nca_research/results"
export PYTHONPYCACHEPREFIX="$NCA_DATA_ROOT/pycache"
export TMPDIR="$NCA_DATA_ROOT/tmp"
export TORCH_HOME="$NCA_DATA_ROOT/torch-home"
export TORCHINDUCTOR_CACHE_DIR="$NCA_DATA_ROOT/inductor-cache"
export HF_HOME="$NCA_DATA_ROOT/hf-home"
python3 -m nca_text.train
```

A truncated public cache is reused from `$NCA_DATA_ROOT/cache/hermes_kimi_head_truncated.json` (3.3 MB). The checkpoint is `$NCA_DATA_ROOT/checkpoints/text_nca.pt` (138 KB). Metrics: `$NCA_DATA_ROOT/text_nca_metrics.json`. If the rows API fails, the run falls back to `nca_text/fixtures/synthetic_traces.json` and says that real rows were not loaded. This measured run did load real public rows.

## What not to use on this M1 for this run

- float64 tensors on MPS.
- `torch.compile` for this conv. The probe was slower than eager.
- llama.cpp, Ollama, GGUF 4-bit, speculative decoding, a KV cache, or continuous batching. Those serve an autoregressive teacher. This label set does not need one. A later cleanup pass can call the 3060; do not load a 7B–9B model into the 16 GiB beside the dataset cache.
- MLX, until a teacher or a larger train actually runs on the Mac. Installing it was skipped so the internal disk would not take another framework.
- Browser label alignment. Local `browser_exec` and public `browser_navigate` / `click` / `snapshot` stay out.
- Writing checkpoints, Hugging Face caches, or the trace cache under the home directory or `/tmp`.

## Not verified

MLX, llama.cpp, and Ollama were not timed here. Paper and abstract speeds above were not re-measured. The full Hermes parquet and the `glm-5.1` config were not loaded. Local message text was not copied to the SSD.
