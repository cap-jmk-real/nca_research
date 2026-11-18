# nca_research

Code, results and manuscript sources for *Micro-Routing with Graph Neural Cellular Automata for Sub-Millisecond Agent Tool Selection*.

An agent harness picks the next tool at every turn. This repository tests whether a small graph neural cellular automaton (GNCA, 35k to 59k parameters) can make that choice from a graph of the recent trace on CPU in under a millisecond, and how it compares with Random Forest and HistGradientBoosting on identical splits.

Where to start:

- [Install](install.md) the dependencies.
- Read the [repository layout](layout.md) to find code, results and paper sources.
- [Reproduce](reproduce.md) the Hermes table from the stored results, or run the smoke pipeline.
- Browse the [API reference](api.md) for `nca_text` and the `stage0` helpers.
- See [data and licence](data-and-licence.md) for where data lives and how the material is licensed.
