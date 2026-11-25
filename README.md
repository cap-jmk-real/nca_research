# nca_research

Research code and manuscript for **Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection**.

- Paper workspace: [`paper/README.md`](paper/README.md)
- Project handoff and file map: [`HANDOFF.md`](HANDOFF.md)
- How to cite: [`CITATION.cff`](CITATION.cff)

## License

This repository uses a **split license**. Paths not listed under [CC BY 4.0](#cc-by-40) are under the [MIT License](LICENSE) (SPDX: `MIT`, copyright Julian M. Kleber).

### MIT (software and documentation code)

Applies to source code, build scripts, Lean proofs, configuration, and prose documentation outside the manuscript and committed datasets, including:

| Paths | Notes |
| --- | --- |
| `scripts/`, `nca_text/`, `formal_proofs/`, `tests/` | Python, Lean, and tests |
| `stage0/` (except `stage0/stage0_results/`) | Training and evaluation **code** |
| `config/`, `configs/`, `docs/`, `.github/` | Config and docs |
| Root files such as `install.sh`, `requirements.txt`, `EXPERIMENT_PLAN.md`, `HANDOFF.md` | Repo tooling and planning |

Full terms: [`LICENSE`](LICENSE).

### CC BY 4.0

Applies to the **manuscript**, **supporting paper material**, and **committed data and result artifacts** (SPDX: `CC-BY-4.0`). Full legal text: [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0).

| Paths | Contents |
| --- | --- |
| `paper/` | LaTeX, PDFs, figures, SI registry, methods drafts (includes `paper/tmlr/`) |
| `data/` | Curated and extra trace samples and stats checked into git |
| `stage0/stage0_results/` | Benchmark summaries, JSON/NPZ result exports in the repo |

Large checkpoints, jsonl training logs, and TensorBoard runs referenced from the paper are **not** in git; when published separately they follow the same CC BY 4.0 policy as the in-repo result files.

**Attribution (CC BY 4.0):** Julian M. Kleber, *Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection*, https://github.com/cap-jmk-real/nca_research, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### SPDX in metadata

| File | Field | SPDX |
| --- | --- | --- |
| [`CITATION.cff`](CITATION.cff) | `license` (preferred citation / paper) | `CC-BY-4.0` |
| [`LICENSE`](LICENSE) | (repository default for software) | `MIT` |

Third-party datasets and benchmarks retain their upstream licenses (see `docs/` and dataset cards).