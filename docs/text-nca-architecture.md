# Text NCA architecture

**Date:** 2026-10-03

The cellular rule for next-tool prediction is one-dimensional, along the trace. The previous 6×8 conv grid is not used for this run. Hypothesis 4 is unchanged: the same observation tensor is scored by a flat linear readout with no cellular update.

## Why the image grid is a poor fit

A tool trace has one axis that means something: the order of recent steps. The 2D model wrote hash bins across a width of 8 and applied a Sobel filter in both axes. That width is a reshaped feature vector. Neighbors along it are arbitrary hash collisions, not nearby events.

The image rule this copied is the growing NCA. Mordvintsev, Randazzo, Niklasson, and Levin (Distill, 2020, DOI `10.23915/distill.00023`) define perception as a 3×3 convolution: the cell state, plus Sobel filters for the partial derivatives in x and in y. The reference notebook in [google-research/self-organising-systems](https://github.com/google-research/self-organising-systems/blob/2580099a3d8419bafd683265adf6b82295d9e485/notebooks/growing_ca.ipynb) implements that with `Conv2D` and the same pair of Sobel kernels. Both axes are spatial axes of an image. A trace does not have the second one.

On the 400-session pilot, that 2D update did not beat the flat readout (public val 38.5% vs 39.1%). That number is from the measured run in [m1-engines.md](m1-engines.md), not a reason by itself to change the axis. The mismatch above is the reason. The pilot is only evidence that the 2D write did not pay for itself on this label set.

Mittal et al. (preprint, arXiv:2608.02050, 2026) say NCAs have been studied almost exclusively for images and textures, and they map the same primitive onto a 1D sequence of token states. Their Table 1 sets the cell state to a token hidden state and the perception kernel to causal windowed attention. On WikiText-103, at about 30M parameters and 60k steps, hierarchical TextNCA reaches 60.3 perplexity against 52.8 and 44.7 for parameter-matched transformers, so they treat it as a probe. Iteration helps only inside a narrow band, with a measured optimum at 4 shared-weight steps, and that benefit required a GRU gate and a learned per-step embedding. The Klaut gateway returned this preprint (DOI `10.48550/arxiv.2608.02050`, `is_preprint: true`) for a language-modeling query. A separate gateway query for 1D neural cellular automata did not return a conv-style text NCA; the hits were image NCAs and classical 1D automata.

Lee, Han, Kumar, and Agrawal (preprint, arXiv:2603.10055) use an NCA the other way around. The opened paper trains a transformer on synthetic trajectories from spatially local NCA rules in the Game of Life / growing-NCA family. Their project page says those trajectories are tokenized with 2×2 patches and fed to a standard transformer ([hanseungwook.github.io/blog/nca-pre-pre-training](https://hanseungwook.github.io/blog/nca-pre-pre-training/)). That is a data source for a language model, not a cellular readout over a tool trace. The gateway search for the title returned a PREreview of the preprint (Zenodo `10.5281/zenodo.18993092`) rather than the paper record itself.

This run does not adopt TextNCA's attention windows or its 30M-parameter hierarchy. The trace window is five cells (the goal plus up to four previous steps). A radius-1 rule iterated 4 times can cross that chain. Four iterations matches their measured optimum, on a different task and scale, so it is a choice, not a transferred result.

## What this run trains

Each decision row is a length-5 sequence.

| Index | Cell |
| --- | --- |
| 0 | Latest user goal, always occupied |
| 1–4 | Up to four recent steps, right-aligned. Empty slots stay zero |

A cell holds a 48-d hashed bag of tokens, a one-hot of the step's tool when that name is in the training vocabulary, and an occupied bit. Observation channels are not updated. Hidden channels (16) start at zero for every row and are carried across the 4 local updates only. Carrying them from one decision row to the next is a different experiment; each curated row already stores the recent trace.

Perception is a fixed depthwise conv1d: identity `[0, 1, 0]`, gradient `[-0.5, 0, 0.5]`, Laplacian `[0.25, -0.5, 0.25]`. A 1×1 map produces a residual delta and a sigmoid gate. A learned vector per iteration is added to the hidden state before perception, so the shared rule can differ by step. The last 1×1 map is zero-initialized, so the initial update is zero. The readout is a linear map on the flattened hidden state. The flat baseline is a linear map on the flattened observation, trained in the same loop.

Loss is sqrt-inverse-frequency cross-entropy. Top-1 is the reported accuracy, not the training objective. The constant baseline predicts `terminal` on every row.

## Sources opened

- Mordvintsev, Randazzo, Niklasson, and Levin (2020). Growing Neural Cellular Automata. *Distill*. DOI `10.23915/distill.00023`. [distill.pub/2020/growing-ca](https://distill.pub/2020/growing-ca). Gateway record, not a preprint.
- Growing-NCA notebook, `Conv2D` Sobel perception: [self-organising-systems/notebooks/growing_ca.ipynb](https://github.com/google-research/self-organising-systems/blob/2580099a3d8419bafd683265adf6b82295d9e485/notebooks/growing_ca.ipynb).
- Mittal, Anand, Kumar, Kukreja, Prasad, Dulloo, Cambria, Liu, Wang, and Shah (2026). TextNCA: Neural Cellular Automata for Language Modeling via Hierarchical Local Attention. Preprint, arXiv:2608.02050. [arxiv.org/abs/2608.02050](https://arxiv.org/abs/2608.02050).
- Lee, Han, Kumar, and Agrawal. Training Language Models via Neural Cellular Automata. Preprint, arXiv:2603.10055. [arxiv.org/abs/2603.10055](https://arxiv.org/abs/2603.10055). Project page opened the same day: [nca-pre-pre-training](https://hanseungwook.github.io/blog/nca-pre-pre-training/).
- Gateway: `POST http://192.168.10.29:30487/v1/research/papers/search` with sources `arxiv`, `openalex`, `crossref`, on 2026-10-03.

## Measured run

150,000 AdamW steps on Apple M1, PyTorch 2.13, MPS eager, batch 1024. `torch.compile` was not used for the resumed steps. The first 12,496 steps were the curated-only run (57,675 rows) and were resumed into the mix below, so the later steps are a continuation, not a fresh 150k on the mix alone.

Train rows: 484,986 across 25,304 sessions, from curated train, extra Hermes train, OpenHands train, and the SWE-smith partial train. Validation is the curated val split only (6,995 rows, 382 sessions). Local `~/.hermes` stayed out of the loss (843 rows scored, 33 sessions, 24 rows outside the 13-name vocabulary).

Step time on the mix, from the 138 validation windows at steps 13,000 through 150,000: mean 98.9 ms/step, median 97.7, min 63.0, max 206.7. Observations stayed in an SSD file and each step copied one batch onto MPS.

| Checkpoint | Val top-1 NCA | Flat | Always-terminal | Local NCA | Local flat | Local always-terminal |
| --- | --- | --- | --- | --- | --- | --- |
| Step 150,000 | 64.3% | 59.8% | 54.3% | 37.6% | 44.1% | 49.5% |
| Best val, step 26,000 | 65.2% | 60.7% | 54.3% | 40.1% | 44.7% | 49.5% |

On the public val split the 1D rule beats the flat readout. On the local Hermes test it does not: always-terminal is ahead of both learned heads.

Checkpoints and the full log are on the SSD: `results/checkpoints/text_nca_1d_mix13_last.pt`, `results/checkpoints/text_nca_1d_mix13_best.pt`, `results/logs/text_nca_1d_mix13.jsonl`. The small metrics copy is [text_nca_1d_mix13_metrics.json](text_nca_1d_mix13_metrics.json).

## Benchmarks

No suite score was produced. The checkpoint is a 13-way Hermes tool label. BFCL non-live AST and irrelevance need function-call ASTs, and the official files are eval categories. NESTFUL `data_v2` is the 1,861-row benchmark of nested math APIs. τ2 airline and retail need those booking APIs plus a user simulator. AppWorld dev was not installed; a published train row uses a single `appworld` tool, and `test_normal` was not run. The blocker record is `/Volumes/Extreme SSD/nca_research/results/benchmarks_1d.json`.
