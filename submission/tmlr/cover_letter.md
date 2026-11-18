# Draft cover note for the TMLR submission (not sent)

TMLR has no separate cover-letter upload; the text below is meant for the OpenReview submission form fields
(Action Editor recommendation, "comments", human-subjects/IRB, funding, competing interests, LLM disclosure).
Julian reviews and edits before anything is pasted into OpenReview.

---

**Title:** Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection

**Author:** Julian M. Kleber (ORCID 0000-0001-5518-0932), Independent Researcher, julian.m.kleber@gmail.com
(identity is withheld in the anonymous PDF; OpenReview stores it in the profile)

**Dear Action Editor,**

Please consider the attached manuscript for TMLR.

**What the paper claims.** A 35.3k to 59.2k parameter Graph Neural Cellular Automaton (GNCA) router selects the next tool for an LLM agent from a relational trace graph. On four tool-routing benchmarks (3 seeds each) it beats a Random Forest baseline on all four and HistGradientBoosting on API-Bank, trails HistGradientBoosting on Hermes by 0.55 pp, and uses up to 307x fewer parameters than the Random Forest; batch-1 CPU forward passes take 0.35 to 0.66 ms. Ten structural theorems are machine-checked in Lean 4. A risk-gated System-1/System-2 harness is evaluated with a selective risk-coverage analysis.

**What the paper does not claim.** The BFCL v4 number in the abstract is a 692-turn routing split, not the official leaderboard protocol. In the official BFCL v4 harness the Hermes-trained gate is cheaper but not more accurate: overall accuracy drops from 45.88% (baseline, no GNCA) to 35.70% (GNCA gate); multi-turn accuracy falls from 28.88% to 11.88% (Table "bfcl-official" in the manuscript). The live dual-system pilots (N=10, N=30, N=3 episodes) are labelled exploratory. These limits are reported in the main text so that the claims in the abstract stay inside the evidence, in line with TMLR's first acceptance criterion.

**Fit with TMLR.** The submission is a sound-empirical-validation paper with a formal component and honest negative end-to-end results. It does not rely on state-of-the-art claims or novelty beyond what the evidence supports.

**Related submissions / preprints.** A version of this work is publicly visible as a preprint on aiXiv (aixiv.261101.000002), which ran its own automated review. It is not published or accepted at any archival venue and is not under review elsewhere. The TMLR PDF is anonymized and does not link to the named version.

**Code and data.** MIT-licensed code, splits, checkpoints, and Lean proofs are public (link withheld in the anonymous PDF; an anonymized mirror will be provided in the supplementary ZIP).

**Human subjects / IRB:** none. **Funding:** none. **Competing interests:** none.

**LLM use (TMLR LLM policy):** AI tools assisted code and text preparation under the author's review; the author takes full responsibility for all content, numbers, and citations. [Julian to confirm the exact wording and tools before submission.]

**Suggested Action Editors / reviewers:** [to be chosen by Julian from the TMLR editorial board page; none proposed here]

Sincerely,
Julian M. Kleber
