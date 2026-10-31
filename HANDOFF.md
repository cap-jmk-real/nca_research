# Handoff: Next-Tool NCA Research & Paper

**Working Title:** *Beyond 1D Grid Automata: Architecture Meta-Learning for Graph Neural Cellular Automata in Agent Execution Traces*  
**Paper Location:** [`paper/main.tex`](paper/main.tex) (Compiled: [`paper/main.pdf`](paper/main.pdf))  
**Status:** **Expanded to Graph NCAs, Architecture Meta-Learning, & Machine-Checked Lean 4 Proof Suite (Audit Score: 9.65 / 10.0)**. Embedded results from 21 registered experiments and 10,000 bootstrap resamples.

---

## 1. Core Concept & Key Findings

* **Beyond 1D Topologies via Architecture Meta-Learning:** Extends rigid 1D regular lattices ($L=5$) to **Graph Neural Cellular Automata (GNCA)** governed by continuous architecture meta-learning over spatial topologies $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A_\alpha)$:
  $$\min_{\alpha \in \mathcal{A}} \mathcal{L}_{\text{val}}(\theta^*(\alpha); A_\alpha) \quad \text{s.t.} \quad \theta^*(\alpha) = \arg\min_\theta \mathcal{L}_{\text{train}}(\theta; A_\alpha)$$
  Bilevel meta-learning dynamically discovers optimal spatial interaction structures that continuously interpolate between 1D linear chains, hierarchical call-tree DAGs, and star goal-broadcasting.
* **Hierarchical Action Dispatch:** Jointly synthesizes the action tuple $\mathbf{a}_T = (y_{\text{tool}}, \mathbf{y}_{\text{arg}}, g_{\text{abstain}}, \tau_{\text{escalate}})$:
  1. Primary tool dispatch ($y_{\text{tool}} \in \mathcal{V}$)
  2. Structured argument slot / entity binding ($\mathbf{y}_{\text{arg}} \in \mathcal{A}$)
  3. Explicit abstain / irrelevance gating ($g_{\text{abstain}} \in [0, 1]$)
  4. Uncertainty & execution risk escalation score ($\tau_{\text{escalate}} \in [0, 1]$) for adaptive delegation to frontier LLMs.
* **Machine-Checked Lean 4 Proof Suite:** Formalized in `formal_proofs/NCAMetaLearning` and compiled with zero unproven axioms:
  * **Theorem 1 (Topological Subsumption):** Proves 1D lattices are strict submanifolds of GNCAs (`theorem_subsumption_1d`).
  * **Theorem 2 (Receptive Field Separation):** Proves $O(1)$ diameter for meta-learned star topologies vs $\Omega(V)$ for 1D chains (`theorem_receptive_field_separation`).
  * **Theorem 3 (Permutation Equivariance):** Proves equivariance of cellular message passing under node permutations (`neighborhood_equivariance`, `path_equivariance`).
* **Primary Results (Under strict $\ge 150\text{k}$ AdamW training floor):**
  * **Meta-Learned GNCA Top-1 (Mix13):** **$65.18\%$** (`EXP-20261003-001`) vs. Flat Linear Twin **$60.71\%$** ($+4.47\%$ margin).
  * **Versus LLM:** Outperforms compact autoregressive model Qwen2.5-1.5B ($6.29\%$) on closed vocabulary routing.
  * **Ablations:** Goal broadcast (`long_range`) reaches **$63.55\%$** (`EXP-20261006-001`), outperforming standard 1D local baseline ($62.60\%$).

---

## 2. Directory Layout & File Map

* **Paper Files:**
  * [`paper/main.tex`](paper/main.tex): Complete 6-page double-column LaTeX manuscript with Supplementary Information.
  * [`paper/main.pdf`](paper/main.pdf): Compiled PDF (Zero overfull hboxes, Audit Score: 9.65 / 10.0).
  * [`paper/outline.md`](paper/outline.md): Living IMRaD outline and claim tracking.
  * [`paper/methods.md`](paper/methods.md): Mathematical formulations.
* **Formal Lean 4 Verification Suite:**
  * [`formal_proofs/lakefile.toml`](formal_proofs/lakefile.toml): Lake build manifest.
  * [`formal_proofs/NCAMetaLearning/Basic.lean`](formal_proofs/NCAMetaLearning/Basic.lean): Graph definitions.
  * [`formal_proofs/NCAMetaLearning/Topology.lean`](formal_proofs/NCAMetaLearning/Topology.lean): Subsumption and Receptive Field Diameter theorems.
  * [`formal_proofs/NCAMetaLearning/Equivariance.lean`](formal_proofs/NCAMetaLearning/Equivariance.lean): Permutation equivariance theorems.
  * [`formal_proofs/Main.lean`](formal_proofs/Main.lean): CLI verification runner (`lake build`).
* **Supporting Information & Evidence:**
  * [`paper/si/tables/S1_experiment_registry.md`](paper/si/tables/S1_experiment_registry.md): Full registry of 21 experimental runs.
  * [`paper/si/tables/S2_next_tool_top1.md`](paper/si/tables/S2_next_tool_top1.md): Measured next-tool classification table.
  * [`paper/si/tables/S3_claims_ledger.md`](paper/si/tables/S3_claims_ledger.md): Claim-to-experiment mapping ledger.
* **Runner Scripts:**
  * `train_mix13.py`: Main 150k training loop.
  * `prepare_hf_dataset.py`: Secret scrubbing and dataset curation.
  * `select_top4_artifacts.py`: Automated top-4 model selector for HF upload.

---

## 3. How to Build & Verify

### Build and Verify Formal Lean Proofs
```bash
cd formal_proofs
lake build
.lake/build/bin/ncametalearning
```

### Compile the Paper
```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

### Remaining Steps to Finalize
1. **Promote Hugging Face Artifacts:**
   Run `python scripts/select_top4_artifacts.py --write` to tag the best weights for `cap-jmk-real/textnca-next-tool`.
2. **Review Section 5 Discussion:**
   Verify whether local private developer traces (`~/.hermes`) need additional carry-state analysis before submission.
3. **Submit to arXiv / Conference:**
   Submit to arXiv (`cs.AI` / `cs.SE`) or target ICLR / NeurIPS agentic workshops.
