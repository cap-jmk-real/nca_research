# Distillation, multi-harness RL, and message passing

**Date:** 2026-10-05

No distillation has been run. The measured model is still the 1D next-tool NCA in [text-nca-architecture.md](text-nca-architecture.md): supervised cross-entropy on trace labels, 150,000 steps, best public val 65.2% against a flat readout at 60.7%. [hermes-first-run.md](hermes-first-run.md) keeps a teacher out of that comparison on purpose. `qwen3.5:9b` on the 3060 is allowed later only to shorten bad tool results and mark repeated failing calls. It is not a policy the automaton copies, and its weights and hidden states are not distillation targets.

## What the FineEnvs guide actually trains

[The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl) (Kolavi et al., Hugging Face and Liquid AI, published 1 October 2026; chapters read from the Space source the same week) is a training stack for a generative language-model policy. It is not a cellular-automaton trainer and it is not a distillation method.

Three pieces:

- **OpenEnv** exposes a Gymnasium-style environment and a capture proxy. The proxy sits on the model URL the harness already calls. It records prompt token ids, sampled token ids, per-token log probabilities, and a loss mask. Harnesses are not modified.
- **Harbor** supplies the task, the harness, and the sandbox as separate choices. Version 0.22.0 ships adapters for more than 40 harnesses.
- **TRL async GRPO** trains on those traces. Generation runs on one GPU (vLLM) while another GPU updates weights that are usually two optimizer steps behind, and never more than four.

Their own run used LiquidAI LFM2.5-2.6B, 1,000 SmolDataEnvs tasks, groups of eight rollouts, two H100s, and one E2B sandbox per rollout. OpenCode-only training took about 32 hours; training across OpenCode, Claude Code, Codex, and Mini-SWE-Agent took about 46 hours. Held-out pass@1, averaged over the four harnesses, moved from 42.2% to 54.2% for the multi-harness run and to 52.3% for OpenCode only. The multi-harness model used 31% fewer tool calls than the base model on tasks both solved. OpenCode-only gained mostly inside OpenCode (34% to 58%) and less in the harnesses it never trained in.

A small efficiency bonus, at most 0.1 and only on a correct answer, is what separates rollouts once a whole group is correct. Their earlier Qwen3.5-2B runs rewarded correctness alone. Tool calls grew, answers later hit the evaluation length cap, and accuracy rose and then fell (multi-harness peak 37.0% at step 500, back near 26% at step 1,000).

### Their teacher comparison is the distillation result

They also ran Qwen3.8-27B as a teacher through the same four harnesses and fine-tuned LFM2.5-2.6B on the successful rollouts only (completion-only loss, two epochs). That is sequence-level imitation of a teacher, which is the distillation-shaped experiment in the guide.

| Setup | Held-out pass@1 |
| --- | --- |
| LFM2.5-2.6B base | 42.2% |
| Multi-harness SFT on the 27B teacher's successes | 43.1% |
| OpenCode-only SFT on the teacher | 47.5% |
| Multi-harness RL on the student's own rollouts | 54.6% at the best checkpoint (step 700) |

Multi-harness SFT improved Claude Code, Codex, and OpenCode, then dropped Mini-SWE-Agent from 62.1% to 45.2%, so the average barely moved. The article treats this as one seed and unequal token counts, not a ranking of methods. The direction matches the first-run note: copying a stronger model's traces is a weaker target than training on the policy's own scored attempts, and mixing harnesses in the copy can erase a harness the student was already good at.

## How the NCA uses it

The published GRPO run is the wrong tool for this checkpoint. The accuracy gap is the label prior.

Curated Hermes train (`data/curated/stats.json`) is 52% `terminal` and 5% `respond`. Curated val is 54% `terminal` and 5% `respond`. Local Hermes is 49% `terminal` and 8% `respond`. The 150k mix is 37% `respond` and 29% `terminal`, because OpenHands and SWE-smith map bare assistant turns and `finish` onto `respond` (179,330 of 484,986 train rows). The NCA still beats the flat readout on curated val (65.2% vs 60.7%). On local sessions it loses to always-terminal (40.1% vs 49.5%).

FineEnvs' own SFT result says not to pour the 17,929 teacher completions into that mix. Multi-harness SFT on those completions moved LFM2.5 from 42.2% to 43.1% and dropped Mini-SWE-Agent from 62.1% to 45.2%. Prose answers would land on `respond`, the class this mix already over-samples.

What their stack does have, and this run did not, is a verifier. The common-task slice is 692 tasks solved in all four harnesses (2,768 rollouts). `scripts/stage_fineenvs_rows.py` keeps only those tasks, maps harness tool names onto the existing 13 labels (`bash`/`shell` to `terminal`, `Read` to `read_file`, `Edit`/`apply_patch` to `patch`, and the same kind of alias for search, write, code, todos, and delegation), and drops a completion that has no mapped tool. The class count stays 13. Tasks are split with `sha256(task_id) % 10`, and that holdout is scored in the log as `fineenvs_val_top1_*`. Checkpoint selection stays on curated val.

Once `datasets/extra/fineenvs_common_train.jsonl` is on the external volume, training does two things:

- A quarter of the rows drawn for each tool come from those graded calls. The other three quarters stay the current mix.
- The tool itself is drawn at the curated Hermes rates above, not at the 37% `respond` rate of the coding files.

Uniform sampling stays in place when the FineEnvs file is absent, so a rerun without it matches the 150k recipe. The volume was not mounted when this was written, so the jsonl is not staged and the 150k checkpoint is unchanged.

The stack cannot be pointed at the text NCA. GRPO here consumes sampled token ids and the log probability of each sampled token. `TextNCA` returns class logits over 13 tool names. The capture proxy also refuses training traces from an engine that cannot return token ids and processed logprobs; a hosted chat API is evaluation-only. Replaying the published recipe needs Python 3.12, OpenEnv 0.7 or later, TRL from main (the harness worker is not in the 1.14.1 release), two GPUs, and a sandbox backend. This machine is an M1 with 16 GiB. The 3060 serves `qwen3.5:9b` for inference. Neither is the training setup in the article.

What does transfer, without running that job:

- A model trained in one harness learns that harness's tool names, context layout, and stop conditions. KwaiKAT call this format, context, and control-flow overfitting. Orchard reports OpenSWE-32B falling from its OpenHands score to 3.6% on SWE-bench Verified under Kimi-CLI. Our public val mixes curated Hermes, extra Hermes, OpenHands, and SWE-smith, but the label is only the next tool name. That is a shared action vocabulary, not the same thing as training a generative policy inside each harness's loop.
- Liquid's LFM2.5 release names Hermes Agent as one of its training harnesses. Our traces are Hermes. That makes LFM a closer teacher *distribution* than a generic chat model, and it does not make the NCA a student of LFM's weights.
- If a generative teacher is trained later, the guide's result says to score it in every harness it will be used in, and to prefer its own RL over SFT on a larger model's successes. The NCA's job in this repo stays the local readout on cleaned labels.

## The 1D rule is message passing

Each update of `TextNCA` is one round of synchronous message passing on a path of five cells. Perception is a fixed radius-1 filter (identity, gradient, Laplacian). A shared 1×1 map writes a residual into the hidden channels through a sigmoid gate. Four iterations is the whole receptive field of that path: a radius-1 rule needs four steps to carry the goal cell to the far end of the trace.

Message-passing networks fail in three ways that this rule can hit, even on five cells. The names are from the graph-network literature; the mechanism is the same local average.

| Failure | What it is | What people changed |
| --- | --- | --- |
| Oversmoothing | Repeated averaging makes cell states indistinguishable. Li, Han, and Wu (2018). | Residuals, gates, and normalization that keep a cell's own state (GCNII, PairNorm, DropEdge). The NCA gate and the zero-init residual are this fix, in the growing-NCA form. |
| Oversquashing | A fixed-size vector has to carry an exponentially growing neighborhood. Alon and Yahav (2021). | Rewire the graph: diffusion edges (DIGL), curvature-based rewiring (SDRF), delayed edges (DRew). A virtual node, or attention, gives a shortcut that is not a path of local steps. |
| Underreaching | If the number of rounds is shorter than the distance, the far cell never arrives. Alon and Yahav (2021). | More depth, or a long-range edge so depth is no longer the distance. |

Errica et al. (ICML 2025, arXiv:2312.16560) put the three in one training objective. Adaptive Message Passing learns, per node, when to stop and which messages to drop, inside a variational bound. On their node and graph tasks it matches deeper fixed message passing without always paying the full depth. The analogue on this trace is a halt on the iteration count. Four is a constant copied from TextNCA's measured optimum on WikiText, not a depth this model was allowed to choose.

Graph-side selective state models are the same idea with a different mixer. MbaGCN (IJCAI 2025) replaces uniform neighbor aggregation with a Mamba selective scan. Dual Mamba GCN (AAAI 2026) adds a second scan over a global ordering so a node can keep a private trajectory across layers instead of converging to the mean. MAPN (arXiv:2502.16454) makes the propagation asynchronous for the same reason: a synchronous round forces every neighbor in at once.

BraiNCA (Pio-Lopez, Hartl, and Levin, 2026, already in [research-design.md](research-design.md)) is the cellular version of rewiring: an attention layer plus edges longer than the Moore neighborhood. TextNCA (Mittal et al., 2026) replaces the conv perception with causal windowed attention and needs a GRU before extra iterations help. Both are on the record as later variants, not as the current checkpoint.

## Transformer alternatives

Attention is one way to add the long-range edge. It is quadratic in length and stores a key-value cache that grows with the sequence. The alternatives that have held up as of 2026 replace that cache with a fixed recurrent state. They are sequence models, not cellular rules, and at length 5 the quadratic cost is irrelevant. The part that matches this project is the state: what is kept, erased, or overwritten between steps.

| Family | State update | Status in 2026 |
| --- | --- | --- |
| Selective SSM: Mamba, Mamba-2, Mamba-3 | A data-dependent linear state. Mamba-2 (Dao and Gu, ICLR 2025) is the hardware-efficient form. Mamba-3 (arXiv:2603.15569) changes the discretization and the state so parity-style state tracking is less weak. | Used inside hybrid LLMs. Pure SSMs still lose retrieval tasks that need exact recall of a distant token. |
| Delta rule: DeltaNet, Gated DeltaNet, Gated DeltaNet-2 | The state is an associative memory. The delta rule overwrites the value addressed by the current key instead of adding into it. Gated DeltaNet-2 (arXiv:2605.22791) uses separate gates for erase and write. | Stronger than Mamba-2 on retrieval and long context in the papers above. Hybrids add a short sliding window of real attention. |
| xLSTM | A matrix memory with exponential gating. Beck et al. | A 2026 comparison on code, time series, and distillation from a larger LM (arXiv:2606.12364) found xLSTM ahead of Mamba-2 and Gated DeltaNet at matched size, and attributes that to more stable memory correction. |
| Sliding-window or linear attention | Local attention, or attention rewritten as a recurrent state (Katharopoulos et al., Performers, RetNet). | The usual partner in a hybrid, not a full replacement. TextNCA's perception kernel is the local-attention end of this row. |

No pure replacement has taken the transformer's place in frontier models. The models that match transformer quality at long context are hybrids: a recurrent layer for most positions, a short attention window for local comparison. LFM2.5, the policy FineEnvs trained, is from that liquid / SSM line, which is why their result is about a small recurrent language model and not about a grid.

For the trace NCA the useful split is:

- **Do not** swap the five-cell conv for a Mamba or xLSTM block to save attention cost. There is no attention cost.
- **Do** treat the hidden channels as the recurrent state those papers are designing. They are reset on every decision row today. Carrying them across rows in one session is hypothesis 2 in [research-design.md](research-design.md), and it is the experiment that corresponds to selective erase and write. A learned halt on the four update steps is the Adaptive Message Passing analogue. A long-range edge from the goal cell is the rewiring analogue, and it removes the claim that four local steps are what carries the goal.

Those three are still ablations of the cellular rule. A teacher, and the FineEnvs GRPO stack, stay behind them.

Equal-budget measurements of carry, learned halt, the goal long-range edge, the flat baseline, and regeneration after cell damage are in [hypothesis-tests.md](hypothesis-tests.md).

## Sources opened on 2026-10-05

- Kolavi, Niklaus, Paniego Blanco, Monigatti, Dirhoussi, Burtenshaw, Tunstall, and von Werra (2026). The ultimate guide to multi-harness RL. Hugging Face Space `FineEnvs/multi-harness-rl`, published 1 October 2026. Chapters `introduction`, `basics`, `why-multi-harness`, `openenv`, `training`, `conclusions`.
- FineEnvs SFT data and script: [SmolDataEnvs-multiharness-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-multiharness-sft). Teacher rollouts: [qwen38-27b-harbor-rollouts](https://huggingface.co/datasets/FineEnvs/qwen38-27b-harbor-rollouts).
- Errica et al. Adaptive Message Passing. ICML 2025. arXiv:2312.16560.
- Alon and Yahav (2021). On the bottleneck of graph neural networks and its practical implications. ICLR.
- Li, Han, and Wu (2018). Deeper insights into graph convolutional networks for semi-supervised learning. AAAI.
- Dao and Gu. Transformers are SSMs: generalized models and efficient algorithms through structured state space duality (Mamba-2). ICLR 2025.
- Mamba-3. arXiv:2603.15569.
- Gated DeltaNet-2. arXiv:2605.22791.
- On Subquadratic Architectures (xLSTM vs Mamba-2 vs Gated DeltaNet). arXiv:2606.12364.
- MbaGCN. IJCAI 2025, DOI `10.24963/ijcai.2025/595`.
