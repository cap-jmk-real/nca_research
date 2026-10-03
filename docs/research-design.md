# Research design: multimodal neural cellular automata as an agent substrate

**Date:** 2026-10-03

This note proposes a research program. It does not report a trained model. Every citation below is a record returned by the Klaut research gateway on 2026-10-03 (search or the focused synthesis). Preprints are marked as such.

## Question

Can one learned local update rule, running on a shared multi-channel grid, bind a visual observation and an instruction into a state that remembers across time and emits an action?

## What the retrieved literature already shows

The gateway searches split the field into four clusters. They do not yet contain a paper that occupies all four at once.

### A local rule can grow, hold, and repair a pattern

Mordvintsev, Randazzo, Niklasson, and Levin (2020) train a neural cellular automaton (NCA) whose every cell applies the same local rule, and show that the rule can grow a target pattern from a seed on a grid. Later papers keep that rule and change the space it lives in: isotropic updates (Mordvintsev, Randazzo, and Fouts, 2022), steerable updates (Randazzo, Mordvintsev, and Fouts, 2023), ultra-compact texture rules (Mordvintsev and Niklasson, 2021, preprint), and 3D artefacts that include functional machines (Sudhakaran et al., 2021). Horibe, Walker, and Risi (2021, preprint) put the same regenerative growth onto soft-robot morphologies. Masumori, Sato, and Ikegami (2026) analyse a trained growing NCA and treat fluctuations in its hidden channels as part of how the pattern maintains itself.

These papers are the reason to prefer a cellular state over a flat vector when damage and persistence matter. They train the grid to *be* a pattern. They do not train it to *act* from a fused percept.

### The same kind of grid can perceive

Randazzo, Mordvintsev, Niklasson, Levin, and Greydanus (2020) train an NCA that classifies MNIST digits by settling the grid into a digit-specific state. Sandler et al. (2020, preprint) pose image segmentation as a cellular process. Kalkhof, González, and Mukhopadhyay (2023, preprint) scale that idea to lightweight medical segmentation (Med-NCA). Tesfaldet, Nowrouzezahrai, and Pal (2022) put attention inside the local rule. Pajouheshgar, Xu, Zhang, and Süsstrunk (2023) synthesize dynamic texture in real time (DyNCA), so the state is a trajectory. Richardson, Antal, Blythe, and Schumacher (2024) learn spatio-temporal patterns. Walker et al. (2022, preprint) classify 2D shapes with a physical NCA. Menta, Archetti, and Matteucci (2024) run restoration in a latent NCA. Xu, Zhang, and Süsstrunk (2024, preprint) use an NCA as an adaptor around a vision transformer (AdaNCA): there the cellular rule sits beside a large network.

Perception in this cluster is visual and single-purpose. The instruction, when one exists, is the training target, not a second input the rule must read.

### A goal or a body can be written into the update

Sudhakaran, Najarro, and Risi (2022, preprint) train goal-guided NCAs that control self-organising systems. A 2021 paper in *Innovations in Machine Intelligence* uses an NCA as a robust controller for a cart-pole agent (DOI `10.54854/imi2021.01`; the gateway record has a title, year, and DOI, and an empty author list). Nadizar, Medvet, Nichele, and Pontes-Filho (2022, preprint) control modular soft robots with embodied spiking NCAs. Sato and Masumori (2024) couple developing morphology to motor behavior. Pio-Lopez, Hartl, and Levin (2026) introduce BraiNCA, an NCA with an attention layer and long-range edges on a richer topology than a Moore neighborhood. Their abstract, as returned by the gateway, reports better robustness and learning speed than a vanilla NCA on morphogenesis and on motor control. Xu and Miikkulainen (2025) apply NCAs to ARC-AGI, a grid task where the input is a visual puzzle and the output is a structured grid. Zhang et al. (2023) use NCAs to generate environments rather than policies. Hamon et al. (2025) search for sensorimotor agency inside cellular automata; that paper studies classical automata, and it is the retrieved work that treats agency as a property to find in a grid.

### A second signal can condition the same rule

Stovold (2023) shows that an NCA can respond to an external signal. Catrina, Plajer, and Băicoianu (2025) condition multi-texture synthesis on a signal. Dacre et al. (2025) use an NCA for decentralized sensing across a soft inductive sensor array, so several sensor streams already share one cellular state. These three are the retrieved precedents for writing an extra modality into the grid.

Pande and Grattarola (2023) stack NCAs into a hierarchy. Hernandez Ruiz, Vilalta, and Moreno-Noguer (2021) learn a manifold of NCAs. Najarro et al. (2022, preprint) grow the weights of a network with an NCA (HyperNCA). Béna, Faldor, Goodman, and Cully (2025, preprint) look for a more universal NCA. Lemke et al. (2026, preprint) survey applications, challenges, and opportunities. None of those papers is the multimodal agent substrate described below. They bound how the rule might later be scaled, shared, or surveyed.

## Gap this design fills

Across the records above, perception, control, and extra signals are separate papers. No retrieved paper trains one NCA whose channels are partitioned into a visual field, an instruction, and an action, with hidden channels carried from one environment step to the next. Signal-conditioned texture, a sensor array, and a goal channel are the nearest pieces. The program below is the join.

## Substrate

One grid is the agent's working state.

- **Vision channels** receive a spatially aligned encoding of the current observation. A small convolutional encoder, or a direct copy of a low-resolution view, writes into a fixed slice of the cell vector. The write is local: a patch lands on the cells that correspond to that patch.
- **Instruction channels** receive a discrete command or a short embedding, broadcast or written onto a designated set of cells. Stovold (2023) and Catrina et al. (2025) are the precedent for a signal that the rule can read. Here the signal is an instruction the action depends on.
- **Hidden channels** are not written by either encoder. They are the memory. They persist from the end of one environment step to the start of the next. Masumori et al. (2026) motivate treating those channels as functional, which this design takes as a hypothesis about cross-step memory.
- **One local rule** updates every cell from its neighborhood for K steps after each write. The first experiments use a shared local rule of the growing-NCA type (Mordvintsev et al., 2020). BraiNCA's long-range edges (Pio-Lopez et al., 2026) are a later variant. Starting local keeps the regeneration hypothesis testable.
- **Readout** pools a fixed set of cells, or a learned pool over the action channels, into the next discrete action.

Proprioception, if a task has it, is another write slice. It is optional in the first experiment.

## Hypotheses

1. On a task that is solvable only when both the image and the instruction are present, a grid with both write slices outperforms the same rule with either slice removed.
2. On a task whose correct action depends on an earlier observation, carrying hidden channels across environment steps outperforms resetting the grid after every step.
3. After a contiguous block of cells is zeroed, further update steps restore readout accuracy without retraining. Growing-NCA regeneration (Mordvintsev et al., 2020; Horibe et al., 2021) is the precedent. In this program it is a hypothesis about a policy state, and it needs its own measurement.
4. A baseline with the same encoders and the same readout, and with no spatial update, is the comparison that attributes binding to the cellular rule.

## Experiment ladder

The first run is text traces from Hermes, not images. See [hermes-first-run.md](hermes-first-run.md). The ladder below is the multimodal continuation after that run has a number.

**Step 1. Image plus instruction, one step.** A low-resolution image and a discrete instruction share the grid. Two instructions on the same pixels require different labels (for example digit identity versus parity, in the spirit of the self-classifying grid of Randazzo et al., 2020). Train the rule and both the write and the readout. Measure accuracy per instruction, then under each channel ablation from hypothesis 1.

**Step 2. The grid persists.** A small gridworld gives a local view and a goal token. The hidden channels at the end of step t are the starting state at step t+1. The readout emits one move. Hypothesis 2 is the comparison against a reset grid. Goal-guided NCA (Sudhakaran et al., 2022) and the cart-pole controller (2021, DOI `10.54854/imi2021.01`) are the control precedents; ARC-AGI NCAs (Xu and Miikkulainen, 2025) are the precedent for a visual grid task with a structured answer.

**Step 3. Damage and the flat baseline.** Repeat step 2 after zeroing a block of cells, and against the non-spatial baseline in hypothesis 4. Report recovery over K, not a single post-damage number.

## What this pass does not do

No training code, no new architecture implementation, and no measured result. Language in step 1 is a discrete instruction, not open dialogue. Long-range connectivity, hierarchical NCAs (Pande and Grattarola, 2023), and a manifold of rules (Hernandez Ruiz et al., 2021) stay outside the first experiment so a failure is attributable to the shared local substrate.

## References

Records are from the Klaut research gateway, 2026-10-03. Preprint means the gateway record is an arXiv or otherwise unpublished version.

- Béna, G., Faldor, M., Goodman, D. F. M., and Cully, A. (2025). A Path to Universal Neural Cellular Automata. Preprint, arXiv:2505.13058. DOI `10.48550/arxiv.2505.13058`.
- Catrina, M.-M., Plajer, I. C., and Băicoianu, A. (2025). Multi-texture synthesis through signal responsive neural cellular automata. *Scientific Reports*. DOI `10.1038/s41598-025-23997-7`.
- Dacre, B., Bessone, N., Lo Preti, M., Cafiso, D., Moreno, R., Faíña, A., and Beccai, L. (2025). Neural Cellular Automata for Decentralized Sensing Using a Soft Inductive Sensor Array for Distributed Manipulator Systems. DOI `10.1109/iccar64901.2025.11073064`.
- Hamon, G., Etcheverry, M., Chan, B. W.-C., Moulin-Frier, C., and Oudeyer, P.-Y. (2025). Discovering sensorimotor agency in cellular automata using diversity search. *Science Advances*. DOI `10.1126/sciadv.adp0834`.
- Hernandez Ruiz, A., Vilalta, A., and Moreno-Noguer, F. (2021). Neural Cellular Automata Manifold. CVPR. DOI `10.1109/cvpr46437.2021.00989`.
- Horibe, K., Walker, K., and Risi, S. (2021). Regenerating Soft Robots through Neural Cellular Automata. Preprint, arXiv:2102.02579. DOI `10.48550/arxiv.2102.02579`.
- Kalkhof, J., González, C., and Mukhopadhyay, A. (2023). Med-NCA: Robust and Lightweight Segmentation with Neural Cellular Automata. Preprint, arXiv:2302.03473. DOI `10.48550/arxiv.2302.03473`.
- Lemke, N., et al. (2026). Applications of Neural Cellular Automata: State of the Art, Challenges and Opportunities. Preprint, arXiv:2609.24595. DOI `10.48550/arxiv.2609.24595`.
- Masumori, A., Sato, H., and Ikegami, T. (2026). Structured Fluctuations and the Information Dynamics of Self-Maintenance in Growing Neural Cellular Automata. *Entropy*. https://openalex.org/W7169473393
- Menta, A., Archetti, A., and Matteucci, M. (2024). Latent Neural Cellular Automata for Resource-Efficient Image Restoration. DOI `10.1162/isal_a_00772`.
- Mordvintsev, A., and Niklasson, E. (2021). μNCA: Texture Generation with Ultra-Compact Neural Cellular Automata. Preprint, arXiv:2111.13545. DOI `10.48550/arxiv.2111.13545`.
- Mordvintsev, A., Randazzo, E., and Fouts, C. (2022). Growing Isotropic Neural Cellular Automata. DOI `10.1162/isal_a_00552`.
- Mordvintsev, A., Randazzo, E., Niklasson, E., and Levin, M. (2020). Growing Neural Cellular Automata. *Distill*. DOI `10.23915/distill.00023`.
- Nadizar, G., Medvet, E., Nichele, S., and Pontes-Filho, S. (2022). Collective control of modular soft robots via embodied Spiking Neural Cellular Automata. Preprint, arXiv:2204.02099. DOI `10.48550/arxiv.2204.02099`.
- Najarro, E., Sudhakaran, S., Glanois, C., and Risi, S. (2022). HyperNCA: Growing Developmental Networks with Neural Cellular Automata. Preprint, arXiv:2204.11674. DOI `10.48550/arxiv.2204.11674`.
- Pajouheshgar, E., Xu, Y., Zhang, T., and Süsstrunk, S. (2023). DyNCA: Real-Time Dynamic Texture Synthesis Using Neural Cellular Automata. CVPR. DOI `10.1109/cvpr52729.2023.01987`.
- Pande, R., and Grattarola, D. (2023). Hierarchical Neural Cellular Automata. DOI `10.1162/isal_a_00601`.
- Pio-Lopez, L., Hartl, B., and Levin, M. (2026). BraiNCA: brain-inspired neural cellular automata and applications to morphogenesis and motor control. ALIFE 2026. DOI `10.1162/isal.a.1033`.
- Randazzo, E., Mordvintsev, A., and Fouts, C. (2023). Growing Steerable Neural Cellular Automata. DOI `10.1162/isal_a_00564`.
- Randazzo, E., Mordvintsev, A., Niklasson, E., Levin, M., and Greydanus, S. (2020). Self-classifying MNIST Digits. *Distill*. DOI `10.23915/distill.00027.002`.
- Richardson, A. D., Antal, T., Blythe, R. A., and Schumacher, L. J. (2024). Learning spatio-temporal patterns with Neural Cellular Automata. *PLoS Computational Biology*. DOI `10.1371/journal.pcbi.1011589`.
- Sandler, M., Zhmoginov, A., Luo, L., Mordvintsev, A., Randazzo, E., and Agüera y Arcas, B. (2020). Image segmentation via Cellular Automata. Preprint, arXiv:2008.04965. DOI `10.48550/arxiv.2008.04965`.
- Sato, H., and Masumori, A. (2024). Developmentally Morpho-Motor Coupled Systems Using Neural Cellular Automata. DOI `10.1162/isal_a_00710`.
- Stovold, J. (2023). Neural Cellular Automata Can Respond to Signals. DOI `10.1162/isal_a_00567`.
- Sudhakaran, S., Grbic, D., Li, S., Katona, Á., Najarro, E., Glanois, C., and Risi, S. (2021). Growing 3D Artefacts and Functional Machines with Neural Cellular Automata. DOI `10.1162/isal_a_00451`.
- Sudhakaran, S., Najarro, E., and Risi, S. (2022). Goal-Guided Neural Cellular Automata: Learning to Control Self-Organising Systems. Preprint, arXiv:2205.06806. DOI `10.48550/arxiv.2205.06806`.
- Tesfaldet, M., Nowrouzezahrai, D., and Pal, C. (2022). Attention-Based Neural Cellular Automata. NeurIPS. DOI `10.52202/068431-0593`.
- Towards Self-organized Control: Using Neural Cellular Automata to Robustly Control a Cart-pole Agent. (2021). *Innovations in Machine Intelligence*. DOI `10.54854/imi2021.01`. Author list absent from the gateway record.
- Walker, K., Palm, R. B., Moreno, R., Faíña, A., Støy, K., and Risi, S. (2022). Physical Neural Cellular Automata for 2D Shape Classification. Preprint, arXiv:2203.07548. DOI `10.48550/arxiv.2203.07548`.
- Xu, K. S., and Miikkulainen, R. (2025). Neural Cellular Automata for ARC-AGI. DOI `10.1162/isal.a.844`.
- Xu, Y., Zhang, T., and Süsstrunk, S. (2024). AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer. Preprint, arXiv:2406.08298. DOI `10.48550/arxiv.2406.08298`.
- Zhang, Y., Fontaine, M., Bhatt, V., Nikolaidis, S., and Li, J. (2023). Arbitrarily Scalable Environment Generators via Neural Cellular Automata. NeurIPS. DOI `10.52202/075280-2499`.
