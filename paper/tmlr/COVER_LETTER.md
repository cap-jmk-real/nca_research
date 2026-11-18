# Cover letter draft (TMLR, via OpenReview; "comment to editors" / cover note)

DRAFT. Not sent. Julian reads and edits before anything is submitted.

---

To the Editors-in-Chief and the Action Editor,

I submit "Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection" for review at TMLR.

**What the paper does.** Agent harnesses usually pick the next tool with a multi-billion-parameter language model. The paper studies a small learned router instead: a Graph Neural Cellular Automaton (GNCA) that runs K weight-tied, gated message-passing steps over a call-tree graph. On four benchmarks (Hermes agent traces, BFCL, ToolBench, API-Bank) I tune 13 models with the same search budget, select each on multi-seed validation, and test with five seeds. The result is a measured null result: the tree GNCA is competitive with gradient boosting, an MLP, a small Transformer and an untied GNN, not better, and it is ahead of k-NN retrieval. Component ablations show no detectable effect. On held-out private sessions every model loses 11.8 to 14.4 points. An end-to-end check on the official BFCL v4 harness (local 9B actuator, unmodified checker) shows that a Hermes-trained GNCA used as a respond gate does not help: overall accuracy drops from 45.88% to 35.70% and multi-turn accuracy from 28.88% to 11.88%. These overall scores are not comparable to the public leaderboard. Lean 4 proofs cover structural properties only and are stated not to certify the trained network.

**Why TMLR.** The claims are limited to what the evidence supports, which is what TMLR's acceptance criteria ask for, and the paper is a null-and-negative-results study of a new architecture against well-tuned baselines. All numbers in the paper are generated from result files that are published with the code.

**Prior versions and originality.** [Pending Julian's OK, final wording after the arXiv upload.] A preprint of this work is public under my name (aiXiv now, arXiv planned before submission). Preprint servers are not archival peer-reviewed venues, and TMLR's policy accepts overlap with them. Earlier preprint versions made stronger claims than the result files support. The submitted manuscript is a re-analysis and rewrite that removes them. To keep the review double-blind I do not link the preprint here. The work is not under review and has not been published at any other venue.

**AI tools.** AI assistants (Cursor, Grok Bot, Hermes Agent, Antigravity and SciSpace) were used for code, literature search and text, as disclosed in a first-page footnote. I checked all numbers and citations and I am responsible for the content.

**Code and data.** The code, result files, table and figure generators, the official BFCL run scripts and the Lean proofs are public under the MIT licence. For review they are provided as an anonymized mirror and as the supplementary ZIP. The named repository link is added at camera-ready.

**Human subjects / IRB:** none. The out-of-distribution split uses agent sessions from one in-house deployment, not released (pending Julian's confirmation). **Funding:** none. **Competing interests:** none.

**Scope and format.** The manuscript uses the TMLR style file. The main text runs to about 17 pages before the references (26 pages in total with references and appendices in the current build), which is longer than 12 pages, so I understand review may take the longer timescale that TMLR describes for long submissions.

Sole author: Julian M. Kleber, ORCID 0000-0001-5518-0932, julian.m.kleber@gmail.com.

Suggested action editor: [to be chosen by Julian from the OpenReview list]

Sincerely,
Julian M. Kleber
