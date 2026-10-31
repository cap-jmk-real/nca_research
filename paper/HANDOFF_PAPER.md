# Research Handoff: Next-Tool NCA Paper

**Working Title:** *A One-Dimensional Neural Cellular Automaton for Next-Tool Prediction in Agent Traces*  
**Authors:** Julian M. Kleber (`0000-0001-5518-0932`)  
**Target Venue:** *arXiv (cs.AI / cs.SE)* / *ICLR / NeurIPS Agentic Workshop*  
**Current Status:** **Active Experiments & Draft Phase**. Outlined manuscript exists; training queue running on `engine` GPU / local daemons.

---

## 1. Executive Summary & Objective

This paper investigates whether **one-dimensional Neural Cellular Automata (1D NCAs)** with local update rules and message passing can outperform small autoregressive LLMs (e.g. Qwen2.5-1.5B / Qwen-9B) on **next-tool prediction** within agentic execution traces (Hermes, mix13, specialist coder datasets) under a bounded parameter and compute budget.

### Current Scoreboard Snapshot
* **Curated Val Top-1:** $65.2\%$ (Step 26k mix13) vs. Flat Baseline $60.7\%$ (Pass).
* **Local Hermes Top-1:** $40.1\%$ vs. Always-Terminal baseline $49.5\%$ (Need convergence) and Qwen2.5-1.5B $6.3\%$ (Pass).
* **Gate Policy:** Minimum $150\text{k}$ AdamW steps required for any architectural claim (avoiding short $<8\text{k}$ step smoke artifacts).
* **Gates Config:** [`config/benchmark_gates.json`](../config/benchmark_gates.json).

---

## 2. Directory Layout & Key Files

### Paper Directory (`nca_research/paper/`)
* [`outline.md`](outline.md): Living IMRaD outline with exact section goals and required experiment mappings (`EXP-*`).
* [`methods.md`](methods.md): Architectural equations for 1D cell states, update functions, and message-passing ablations.
* [`availability.md`](availability.md): Open data/model release plan (Hugging Face / Zenodo).
* [`si/`](si/): Supporting Information tables (`tables/`) and experiment run cards (`experiments/`).

### Docs & Release Specifications (`nca_research/docs/`)
* [`paper-release-plan.md`](../docs/paper-release-plan.md): 6-phase protocol for locking gates, preparing datasets, and publishing artifacts.
* [`auto-research-loop.md`](../docs/auto-research-loop.md): Scoreboard tracking, stopping conditions, and long-training requirements.
* [`specialist-coder-stack.md`](../docs/specialist-coder-stack.md): Python / TypeScript / Shell student trace pipelines.

### Scripts & Runners (`nca_research/scripts/`)
* `train_mix13.py` / `train_long_range.py`: Core training runners.
* `prepare_hf_dataset.py`: Privacy and secret scrubbing for Hugging Face trace release.
* `select_top4_artifacts.py`: Top-4 candidate model selector (Slots A, B, C, D).
* `publish_hf.py`: Hugging Face automated publishing script.

---

## 3. Remote Engine Integration (`engine` / `192.168.10.29`)

* **Remote Path:** `/home/s4il0r/nca_research/` and `/home/s4il0r/nca_data/`
* **Cron & Sync:** `17 */3 * * * $HOME/nca_research/stage0/sync.sh` creates encrypted backups under `nca_backups/`.
* **Daemon Sessions:**
  * `tmux attach -t q_train`: Main training queue for H6 150k steps.
  * `tmux attach -t q_teacher`: Teacher evaluation and distillation.

---

## 4. Next Steps for Incoming Collaborators / Agents

1. **Verify H6 150k Runs:**
   Check whether the 150k-step runs (`long_range`, `baseline`, `flat`) have completed on `engine` or the SSD, and record their `val_top1` and confusion matrices into `si/tables/`.
2. **Dataset Scrubbing:**
   Run `python scripts/prepare_hf_dataset.py --check` on the specialist traces (Python, TypeScript) to confirm zero secret leakage.
3. **Artifact Selection:**
   Execute `python scripts/select_top4_artifacts.py --write` to freeze slots A (Best NCA), B (Flat twin), C (Specialist student), and D (Dataset).
4. **Draft Results & Discussion:**
   Using the template in [`paper/outline.md`](outline.md), write §4 (Experimental Setup), §5 (Results), and §6 (Discussion).
