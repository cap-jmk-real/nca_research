# Handoff: Next-Tool NCA Research & Paper

**Working Title:** *Beyond Discrete Action Routing: One-Dimensional Neural Cellular Automata for Hierarchical Tool and Argument Prediction in Agent Traces*  
**Paper Location:** [`paper/main.tex`](paper/main.tex) (Compiled: [`paper/main.pdf`](paper/main.pdf))  
**Status:** **Manuscript Drafted, Expanded to Hierarchical Dispatch, & Compiling Cleanly (Audit Score: 9.40 / 10.0)**. Embedded results from 21 registered experiments and 10,000 bootstrap resamples.

---

## 1. Core Concept & Key Findings

* **Beyond Binary Decisions:** Extends 1D NCAs beyond static 1-of-$K$ tool classification into synthesizing an action tuple $\mathbf{a}_T = (y_{\text{tool}}, \mathbf{y}_{\text{arg}}, g_{\text{abstain}}, \tau_{\text{escalate}})$:
  1. Primary tool dispatch ($y_{\text{tool}} \in \mathcal{V}$)
  2. Structured argument slot / entity binding ($\mathbf{y}_{\text{arg}} \in \mathcal{A}$)
  3. Explicit abstain / irrelevance gating ($g_{\text{abstain}} \in [0, 1]$)
  4. Uncertainty & execution risk escalation score ($\tau_{\text{escalate}} \in [0, 1]$) for adaptive delegation to frontier LLMs.
* **Methodology:** Operates over sequence length $L=5$ (cell 0 = user goal, cells 1–4 = recent tool calls). Hidden channels $H=16$, updated via 1D spatial discrete Laplacians, gradients, and gated 1x1 convolutions over $K=4$ steps.
* **Primary Results (Under strict $\ge 150\text{k}$ AdamW training floor):**
  * **TextNCA Top-1 (Mix13):** **$65.18\%$** (`EXP-20261003-001`) vs. Flat Linear Twin **$60.71\%$** ($+4.47\%$ margin).
  * **Versus LLM:** Outperforms compact autoregressive model Qwen2.5-1.5B ($6.29\%$) on closed vocabulary routing.
  * **Ablations:** Goal broadcast (`long_range`) reaches **$63.55\%$** (`EXP-20261006-001`), outperforming standard local baseline ($62.60\%$) and flat linear ($62.49\%$).

---

## 2. Directory Layout & File Map

* **Paper Files:**
  * [`paper/main.tex`](paper/main.tex): Complete LaTeX manuscript.
  * [`paper/main.pdf`](paper/main.pdf): Compiled 3-page double-column paper.
  * [`paper/outline.md`](paper/outline.md): Living IMRaD outline and claim tracking.
  * [`paper/methods.md`](paper/methods.md): Mathematical formulations.
  * [`paper/availability.md`](paper/availability.md): Hugging Face 4-slot model/data release policy.
* **Supporting Information & Evidence:**
  * [`paper/si/tables/S1_experiment_registry.md`](paper/si/tables/S1_experiment_registry.md): Full registry of 21 experimental runs.
  * [`paper/si/tables/S2_next_tool_top1.md`](paper/si/tables/S2_next_tool_top1.md): Measured next-tool classification table.
  * [`paper/si/tables/S3_claims_ledger.md`](paper/si/tables/S3_claims_ledger.md): Claim-to-experiment mapping ledger.
* **Runner Scripts:**
  * `train_mix13.py`: Main 150k training loop.
  * `prepare_hf_dataset.py`: Secret scrubbing and dataset curation.
  * `select_top4_artifacts.py`: Automated top-4 model selector for HF upload.

---

## 3. How to Build & Next Steps

### Compile the Paper
```bash
cd paper
pdflatex main.tex
```

### Remaining Steps to Finalize
1. **Promote Hugging Face Artifacts:**
   Run `python scripts/select_top4_artifacts.py --write` to tag the best weights for `cap-jmk-real/textnca-next-tool`.
2. **Review Section 5 Discussion:**
   Verify whether local private developer traces (`~/.hermes`) need additional carry-state analysis before submission.
3. **Submit to arXiv / Conference:**
   Submit to arXiv (`cs.AI` / `cs.SE`) or target ICLR / NeurIPS agentic workshops.
