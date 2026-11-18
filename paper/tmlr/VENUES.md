# Free (no author fee) peer-reviewed venues for the NCA paper

Facts below were read from the journals' own pages on 2026-11-18 (Europe/Berlin) unless marked **(3rd party)** (read from an aggregator or search result, not from the journal) or **not verified**. Pages are linked so Julian can re-check. Nothing here has been submitted; no journal account was created.

## Shortlist

| # | Venue | Author fee | Access / licence | Review | Index | Format | Preprint policy |
|---|---|---|---|---|---|---|---|
| 1 | **TMLR** <https://jmlr.org/tmlr/> | none: "imposes no fees or payments to authors" ([policies](https://jmlr.org/tmlr/editorial-policies.html)) | free to read; CC BY 4.0 from submission, authors keep copyright ([author guide](https://jmlr.org/tmlr/author-guide.html)) | open + double-blind via OpenReview, >=3 reviewers; aims at a decision about 9 weeks after submission ([FAQ](https://jmlr.org/tmlr/faq.html)); main text over 12 pages gets longer timescales | DBLP records exist **(3rd party search result, dblp.org/rec/journals/tmlr/...)**; Scopus and DOAJ **(3rd party, PubScope)**; no Web of Science JIF **(3rd party)**; the journal's own pages list no index | TMLR LaTeX style file ([repo](https://github.com/JmlrOrg/tmlr-style-file)); any length, justified by content; anonymous | arXiv/preprint servers allowed at any time, with or without names; no overlap with archival venues |
| 2 | **Complex Systems** <https://www.complex-systems.com/> | none: "no publication charges" ([contribute](https://www.complex-systems.com/contribute/)) | "Platinum Open Access", CC BY ([about](https://www.complex-systems.com/about/)) | not stated on the pages read | own page: ESCI, Scopus, Inspec, MathSciNet, Computer and Control Abstracts, Crossref ([about](https://www.complex-systems.com/about/)); home page cites a 2023 JIF of 1.2 | LaTeX template and `ComplexSystems.sty` ([contribute](https://www.complex-systems.com/contribute/)); referees suggested by the author | not stated on the pages read (**not verified**) |
| 3 | **JAIR** <https://www.jair.org/> | none: "no publication fees are charged to authors" ([about](https://www.jair.org/index.php/jair/about)) | free to read; CC BY (articles since 2023-05-01) | decision in about 8-12 weeks ([submissions](https://www.jair.org/index.php/jair/about/submissions)) | own page: INSPEC, Science Citation Index, MathSciNet ([about](https://www.jair.org/index.php/jair/about)) | JAIR LaTeX template (Overleaf link on [author instructions](https://www.jair.org/index.php/jair/authorinstrs)); reproducibility checklist mandatory, else desk reject; account registration needed | not under review or published elsewhere; authors may post to arXiv and similar |
| 4 | **JMLR** <https://www.jmlr.org/> | none: "no publication fees ... freely available to readers" ([author info](https://www.jmlr.org/author-info.html)) | free to read; CC BY on acceptance | not stated on the pages read | Scopus and Web of Science SCIE **(3rd party)**; not in DOAJ **(3rd party)** | JMLR LaTeX style ([repo](https://github.com/JmlrOrg/jmlr-style-file)); papers must be concise | no simultaneous submission to other journals/conferences; prior preprint policy not stated on the pages read |

## Scope fit

1. **TMLR (recommended).** Its acceptance test is "claims supported by accurate, convincing and clear evidence", and it says modest contributions are fine. This paper is a careful null-result comparison of a new architecture against tuned baselines, with component ablations and an end-to-end negative result. That is the kind of paper this test rewards. Weak point: the topic is applied (agent tool routing), and the novelty score in the repo's own review is the limiting one (7.0).
2. **Complex Systems.** Its scope list names "cellular automata" and "applications of automata to ... machine and deep learning" ([about](https://www.complex-systems.com/about/)). Good topical match for the GNCA and the structural propositions. Weak point: the paper's core content is a tool-routing benchmark study, which is outside what most of its readers follow; review time and preprint policy are not stated on the pages I read.
3. **JAIR.** Covers all AI including agents and machine learning. Strong, established venue (SCI-indexed per its own page). Weak points: the mandatory reproducibility checklist and JAIR template, a rejected-without-review risk for papers that are "less than they might be", and the AI-tools policy ("the core contributions of a paper must come from the authors").
4. **JMLR.** It "does not publish applications of machine learning to other domains" and favours a broad ML audience. A tool-routing benchmark study is a risk here. Listed for completeness; I would not use it first.

## Checked and excluded (they are not free)

| Venue | Why excluded | Source |
|---|---|---|
| TISMIR (music; relevant to the seam paper only) | APC GBP 530 for research articles; waiver possible on request but not guaranteed | [submissions page](https://transactions.ismir.net/about/submissions/) |
| Digital Discovery (RSC) | gold OA with an APC; a third-party summary of RSC pages gives GBP 2,200 from 2026-01-01; I did not find the figure on the author-guideline page itself | [author guidelines](https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery) (APC figure: **3rd party**) |
| JOSS | would only publish the software, not this paper: "Your paper must not focus on new research results accomplished with the software" | [JOSS docs](https://joss.readthedocs.io/en/latest/submitting.html) |

## Prior publication on aiXiv

The paper is on aiXiv as `aixiv.261101.000002` (site shows "v7.7, 2026-11-10"; page fetched 2026-11-18 returns 200). aiXiv describes itself as "a free, AI- or human-peer-reviewed preprint archive" ([about](https://aixiv.science/about)). Of the journals above, TMLR states in writing that preprint servers do not count as overlap; JAIR says authors may post to arXiv and similar services. Complex Systems and JMLR do not say, so ask them or state the aiXiv version in the cover letter. The aiXiv listing shows an older, stronger-claiming abstract than this manuscript; the cover letter says so.

## What changes if Julian picks a different venue than TMLR

| Venue | Still to do |
|---|---|
| Complex Systems | switch to `ComplexSystems.sty` template (download from the contribute page); not anonymous; add 3 suggested referees; trim toward cellular-automata framing |
| JAIR | JAIR template; fill the reproducibility checklist (appendix); structured abstract encouraged; anonymisation rules not checked; Julian registers an account |
| JMLR | jmlr2e style; shorten; add a broader-ML framing; confirm preprint policy with the editors |

## TMLR-specific cautions

- TMLR's rule that authors need a complete OpenReview profile means Julian must create it himself; this package created nothing.
- Quota: TMLR limits the number of submissions per author per year ("Generalized Harmonic Quota Rule, N_1 = 2 and N_9 = 9", [policies](https://jmlr.org/tmlr/editorial-policies.html)). Check the remaining budget with the tool linked there before submitting two papers.
- LLM policy: LLMs may be used as assistive tools; authors are fully responsible and low-quality, largely LLM-generated contributions are examined closely. The manuscript discloses AI use.

## Further options (ported from the PR #14 shortlist)

Read on 2026-11-18 by the author of PR #14 from the pages linked. Not re-checked in the consolidation PR.

- **Computo** (https://computo-journal.org/site/about.html, https://computo-journal.org/site/guidelines-authors.html): "free for both readers and authors", Diamond OA, CC BY 4.0. New methods and numerical studies in statistics and ML. Reproducibility is a publication condition (Quarto notebook, frozen environment, CI, data on Zenodo or OSF), and reviewers run the code. Every submission ends with a "Use of generative AI" statement. The paper would need re-authoring as a Quarto notebook. Review time not stated.
- **Journal of Formalized Reasoning** (https://jfr.unibo.it/about): "neither article processing charges nor submission processing fees", CC BY 3.0, initial assessment within 15 days, two anonymous reviewers within 2 months. Only for a later Lean-centred companion: the 28 Lean theorems are an appendix of this paper, not a stand-alone formalization.
- Excluded: **JOSS** (https://joss.readthedocs.io/en/latest/submitting.html) needs at least six months of public history. The repository was created 2026-10-03, so it is not eligible before about April 2027, and JOSS publishes the software, not the paper.
