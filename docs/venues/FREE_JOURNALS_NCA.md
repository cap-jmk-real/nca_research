# Free (diamond open-access) journal options for the NCA / GNCA micro-routing paper

Prepared 2026-11-18 (Europe/Berlin) on request of Julian M. Kleber. **Nothing was submitted, no journal account was created.**
Every fact below was read from the venue's own pages on 2026-11-18; the URL is next to it. Items I could not verify on an official page are marked *(unverified)*.

Paper state used: `paper/main.tex` at the head of PR #12 (`9535c5a`, canonical PR #5 content + official BFCL v4 harness result).
The paper's honest end-to-end result: on the official BFCL v4 harness the Hermes-trained gate is cheaper but not more accurate
(overall 45.88% baseline vs 35.70% gate; multi-turn 28.88% vs 11.88%). The routing-accuracy claims (API-Bank, ToolBench, BFCL 692-turn split, Hermes) and Lean results are unaffected.
Any venue must be fine with "sound evidence, modest/mixed end-to-end result". That favours venues that judge correctness rather than significance.

## Ranking

| # | Venue | APC (author) | Review time (official) | Fit | Effort |
|---|-------|--------------|------------------------|-----|--------|
| **1 (pick)** | **TMLR** | **0** | aims for a decision about 9 weeks after submission | high | low (package ready) |
| 2 | JAIR | 0 | decision in about 8-12 weeks | medium-high, higher significance bar | medium (JAIR template + checklist) |
| 3 | JMLR | 0 | not stated on the pages I read | medium-low (scope excludes "applications to other domains"; wants broad audience) | medium (JMLR style) |
| 4 | Computo | 0 | not stated on the pages I read | medium (stats/ML numerical studies; reproducibility-first) | high (re-author as Quarto + CI) |
| 5 | Journal of Formalized Reasoning | 0 | 2 reviewers within 2 months | low for this paper (only for a Lean-centred companion) | high |

### 1. TMLR - Transactions on Machine Learning Research (recommended)
* Official: https://jmlr.org/tmlr/ , policies https://jmlr.org/tmlr/editorial-policies.html , author guide https://jmlr.org/tmlr/author-guide.html , FAQ https://jmlr.org/tmlr/faq.html
* APC: "TMLR imposes no fees or payments to authors"; all content free to read; CC BY 4.0, authors keep copyright.
* Peer review: double-blind, open reviewing on OpenReview, at least 3 reviewers; reviews due in 2 weeks for up to 12 pages of main content, 4 weeks above; FAQ: "aims to deliver a final decision approximately 9 weeks after submission".
* Acceptance criteria (https://jmlr.org/tmlr/acceptance-criteria.html): (1) claims supported by accurate, convincing evidence, (2) interest to some of the audience. Explicitly *not* novelty, significance or SOTA. "Gap between claims and evidence" can be fixed by reducing claims. This matches the honest BFCL result.
* Format: PDF made with the TMLR LaTeX style file; any length (unusually long main text delays review); appendix after references; supplementary up to 100 MB (PDF/ZIP), anonymized.
* Preprints: allowed any time on arXiv or other servers, but the submission must not link to a named version. aiXiv listing is therefore compatible; the TMLR PDF must stay anonymous.
* Originality: no overlap with work published, accepted, or in parallel submission at another archival peer-reviewed venue; preprints and non-archival workshops are fine.
* LLM policy: assistive use allowed; authors fully responsible; LLMs cannot be authors.
* Quota: Generalized Harmonic Quota Rule (N_1 = 2, N_9 = 9) per year; needs OpenReview profiles with history for all authors.
* Indexing/reputation: I did not find an indexing list on tmlr pages; a secondary source (SCImago via search) lists Scopus and DBLP, no Clarivate impact factor *(unverified)*. The official news item of 2025-10-21 says TMLR joined the NeurIPS/ICML/ICLR Journal-to-Conference track.
* Needed manuscript changes: see below. Package: `submission/tmlr/` (this PR).

### 2. JAIR - Journal of Artificial Intelligence Research
* Official: https://www.jair.org/index.php/jair/about/submissions , https://www.jair.org/index.php/jair/about
* APC: "we do not charge any publication fees to authors, nor do we charge readers"; diamond OA since 1993; CC BY.
* Review: "a decision returned to the authors in approximately 8-12 weeks" (about page: "within approximately three months").
* Indexing: INSPEC, Science Citation Index, MathSciNet (about page).
* Format: PDF in the JAIR format only ("Submissions that do not use the JAIR format ... will be rejected without review"); mandatory reproducibility checklist appended to the LaTeX source; structured abstract encouraged.
* Originality: not under review or published elsewhere (extended conference versions allowed). arXiv is fine after submission ("Authors may submit their JAIR papers to arXiv").
* Bar: "originality and significance of the contribution" as judged by the AI community; the mixed official-BFCL result and small pilots are weaker against this criterion than against TMLR's.
* Changes: JAIR template, reproducibility checklist, sharper significance argument.

### 3. JMLR - Journal of Machine Learning Research
* Official: https://jmlr.org/author-info.html
* APC: "There are no publication fees associated with this journal and all papers are freely available to readers."
* Scope text: "does not publish applications of machine learning to other domains" and "may deem a paper unsuitable if the editorial board finds its audience too narrow". NCA routing for agents is an application-flavoured method paper; risk of a scope rejection.
* No simultaneous submission; previously published journal papers excluded; workshop/conference versions allowed with a substantive delta.
* Format: JMLR style file, strict formatting rules (https://www.jmlr.org/format/authors-guide.html); review time and preprint policy not stated on the pages I read *(unverified)*.

### 4. Computo
* Official: https://computo-journal.org/site/about.html , https://computo-journal.org/site/guidelines-authors.html
* APC: "free for both readers and authors", Diamond OA, CC-BY 4.0.
* Scope: new methods and numerical studies in stats/ML; reproducibility is a publication condition (Quarto notebook, frozen environment, GitHub Actions CI, data on Zenodo/OSF); reviewers run the code.
* New (2026-11-04): every submission ends with a "Use of generative AI" statement.
* Fit: strong reproducibility story, but the 28-page two-column paper with Lean 4 appendix would need re-authoring as a Quarto notebook; review time not stated *(unverified)*.

### 5. Journal of Formalized Reasoning (only for a Lean-centred companion)
* Official: https://jfr.unibo.it/about : "neither article processing charges nor submission processing fees"; CC-BY 3.0; initial assessment within 15 days, two anonymous reviewers within 2 months.
* Scope: significant formalization efforts. Ten Lean theorems are an appendix of this paper, not a stand-alone formalization contribution, so this is only a later option.

### Considered and excluded
* **JOSS** (https://joss.readthedocs.io/en/latest/submitting.html): no fees, but needs at least six months of public history; `cap-jmk-real/nca_research` was created 2026-10-03, so it is ineligible until about April 2027, and JOSS publishes the software, not this paper.
* **Venues with author fees** were not considered (APC must be 0).

## Manuscript changes needed for the recommended venue (TMLR)
1. **Anonymize** (done in `submission/tmlr/main.tex`): author block, repository and Hugging Face links. Still to do by Julian: an anonymized code mirror (e.g. anonymous.4open.science) for the supplementary ZIP; remove ORCID/URLs from supplementary files.
2. **Put the official BFCL v4 result into the abstract and conclusion.** Proposed sentence (numbers from Table `bfcl-official`): "On the official BFCL v4 harness the Hermes-trained router gate lowers overall accuracy from 45.88% to 35.70% (multi-turn 28.88% to 11.88%), so the gate saves schema tokens but does not improve accuracy." The abstract currently only says the 692-turn split is not the official protocol.
3. **Length.** 30 pages of single-column main text. TMLR allows it but review requests become 4 weeks instead of 2 and the FAQ warns of delays. Consider moving ablation tables and the exploratory pilots to the appendix.
4. **Close the open items in `paper/JULIAN_TODO.md`** that reviewers on OpenReview will ask about: same-hardware latency control for the LLM baseline, per-probe BFCL k/N ledgers with Wilson intervals, false-trigger/context-overflow instrumentation or removal of that language.
5. Fix the one overfull table (`tab:feature-eval`).
6. Add the TMLR LLM-use sentence (draft in `submission/tmlr/cover_letter.md`).
7. Decide the aiXiv relationship: TMLR accepts preprints; the cover note discloses the aiXiv listing and its automated review.

## Open decisions for Julian
* Approve TMLR as the first target (then he, not an agent, creates the OpenReview profile and submits).
* Which of the open PRs (#9 concise v7.6, #11 rigour, #12 BFCL official) is the manuscript of record. This package is built on #12 because it is the only one with the official BFCL result.
* Accept the 30-page length or trim.
* Choose suggested Action Editors.
