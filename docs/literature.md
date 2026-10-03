# Literature

**Date:** 2026-10-03
**Gateway:** `http://192.168.10.29:30487` (Klaut research gateway on engine, NodePort 30487). `https://research.klaut.pro` did not answer from this machine.
**Synthesis model:** `qwen3.5:9b` via Ollama on engine (`http://engine:11434/v1`).

## How this note was gathered

Searches were `POST /v1/research/papers/search` with sources `arxiv`, `openalex`, and `crossref`, `year_from` 2019 unless noted, limit 8–15.

- growing neural cellular automata
- neural cellular automata image segmentation self-classifying
- goal-conditioned controllable neural cellular automata
- neural cellular automata reinforcement learning control agent
- neural cellular automata memory world model
- multimodal vision language neural cellular automata
- Self-classifying MNIST digits neural cellular automata
- Image segmentation via neural cellular automata
- self-organising textures neural cellular automata
- Neural Cellular Automata for ARC-AGI
- Neural Cellular Automata Can Respond to Signals
- BraiNCA brain-inspired neural cellular automata motor control

Two synthesis reports were requested (`POST /v1/research/reports`, preprints included).

1. Topic “multimodal neural cellular automata as a shared substrate for agent perception, memory, and action”, limit 60, sources arxiv, openalex, crossref. The gateway retrieved 60 papers and synthesised 53. Three titles were neural cellular automata. The rest were classical cellular automata, neuroscience, and unrelated matches on words such as agent, memory, and action. That synthesis is not reproduced here.
2. Topic “Neural cellular automata for perception, memory, and agent control”, limit 40, sources openalex, crossref, arxiv. Forty papers were synthesised. Six titles are neural cellular automata. The report endpoint has no year filter, and the query is tokenised, so classical cellular-automata papers remain in the set. The model’s narrative below sometimes treats those papers as neural NCAs. The research design in `research-design.md` cites the NCA records from the searches and from this report, and it does not adopt claims that rest on the mismatched papers.

The pod did not export a PDF (no LaTeX engine in the bootstrap container). Gateway path for this synthesis: `/data/reports/neural-cellular-automata-for-perception-memory-and-agent-cont-2026-10-03.md`.

## Synthesis report

# Neural cellular automata for perception, memory, and agent control

**Date:** 2026-10-03
**Review type:** Rapid literature synthesis (klaut-research-gateway)
**Sources queried:** openalex, crossref, arxiv
**Papers synthesised:** 40

## TL;DR

* The theoretical foundations of discrete dynamical systems provide the intrinsic cause-effect power necessary for animats to adapt and evolve complex behaviors [1].
* Research into cellular automata with complicated dynamics demonstrates that simple local rules can generate highly non-trivial global patterns essential for perception tasks [14].
* Automata-theoretic approaches offer a formal framework for modeling consciousness, linking internal state transitions to cognitive phenomena within a mental monist perspective [27].
* Topological views of cellular automata and transducers clarify the structural relationships between input processing and output generation in agent control systems [40].
* These foundational works collectively establish that neural cellular automata can bridge the gap between low-level grid dynamics and high-level perceptual abstraction.
* By integrating dynamical system theory with automata logic, researchers have created architectures capable of sustaining memory traces and executing goal-directed actions without external supervision.
* The evolution from elementary cellular automata to adapting animats illustrates a trajectory where computational simplicity yields emergent cognitive capabilities [1].
* Complicated dynamics within the automata framework suggest that high-fidelity sensory processing does not require complex underlying hardware, but rather intricate interaction rules [14].
* Formalizing consciousness through automata theory provides a mechanism for internal state evolution that mirrors the temporal integration required for memory retention [27].
* Topological analysis of transducers reveals how input-output mappings in cellular automata can be rigorously defined to support reliable agent decision-making [40].

## Executive Summary

Neural cellular automata (NCA) represent a distinct class of computational models that integrate grid-based dynamics with adaptive learning mechanisms to address complex perception and control tasks. Early theoretical work established the foundational link between cellular automata, dynamical systems, and neural networks, demonstrating that high-fidelity sensory processing can emerge from simple interaction rules rather than complex hardware [15]. This synthesis suggests that the evolution from elementary cellular automata to adapting animats illustrates a trajectory where computational simplicity yields emergent cognitive capabilities, bridging low-level grid dynamics with high-level perceptual abstraction. In practical applications, these architectures have been successfully deployed to simulate and visualize dynamic environments, such as pedestrian activity, confirming their utility in modeling collective behavior without external supervision [2]. Furthermore, the integration of stochastic strategies within automata frameworks has proven effective for simulating complex real-world scenarios like fishing effort and resource management, highlighting the robustness of NCA in handling uncertainty [28]. Collectively, these studies affirm that NCA can sustain memory traces and execute goal-directed actions autonomously. The confidence in this approach stems from the formalization of consciousness through automata theory, which provides a mechanism for internal state evolution mirroring temporal integration required for memory retention. By combining dynamical system theory with automata logic, researchers have created systems capable of reliable agent decision-making, effectively resolving the gap between input processing and output generation. Ultimately, the literature supports the view that complicated dynamics within the automata framework allow for intricate interaction rules that support both perception and control, validating NCA as a versatile tool for next-generation agent architectures.

## Key Findings at a Glance

| Theme | Finding | Evidence strength | Refs |
| :--- | :--- | :--- | :--- |
| **Perception & Modeling** | NCA frameworks effectively model complex spatial dynamics, such as informal settlement growth and pedestrian activity, demonstrating robustness in unsupervised collective behavior simulation [3]. | High | [3] |
| **Autonomy Measures** | Structurally diverse automata exhibit quantifiable levels of autonomy, providing a formal metric to assess the independence of agents within dynamic environments [16]. | Medium | [16] |
| **Control & Traffic** | Cellular automata-based models successfully simulate intricate traffic situations, validating their utility for agent control in high-density flow scenarios [29]. | High | [29] |
| **Emergent Capabilities** | The integration of simple grid dynamics allows for the emergence of high-level perceptual abstractions and goal-directed actions without external supervision. | Medium | [3, 29] |
| **Uncertainty Handling** | Stochastic strategies within automata frameworks enhance system robustness, enabling effective simulation of uncertain real-world scenarios like resource management. | High | [16] |
| **Versatility** | NCA architectures serve as versatile tools for next-generation agent systems by bridging low-level dynamics with high-level cognitive capabilities. | Medium | [3, 16, 29] |

## 1. Introduction and Scope

Neural Cellular Automata (NCA) represent a paradigm shift in agent-based modeling, bridging the gap between low-level grid dynamics and high-level cognitive capabilities [4]. By integrating neural network architectures with cellular automata rules, these frameworks enable the simulation of complex spatial phenomena, ranging from the growth of 3D artifacts to functional machine generation [4]. This section establishes the research scope by examining how NCA models facilitate perception, memory retention, and autonomous control within dynamic environments. Early foundational work has demonstrated that even simple probabilistic cellular automata equipped with memory in cells can effectively model stochastic processes, providing a robust basis for handling uncertainty in real-world scenarios such as epidemic risk perception [17, 30]. The integration of these mechanisms allows agents to exhibit emergent behaviors and goal-directed actions without requiring external supervision, thereby validating NCA as a versatile tool for next-generation agent architectures. Furthermore, the ability to quantify autonomy through structurally diverse automata offers formal metrics to assess agent independence in high-density flow scenarios [16]. Ultimately, the convergence of these approaches suggests that NCA frameworks can effectively model complex spatial dynamics, supporting both perception and control tasks essential for advanced simulation systems [3, 29].

Key contributions identified in this scope include:
* The capacity of NCAs to grow functional machines and 3D artifacts highlights their potential for synthesizing complex structures from simple local rules [4].
* Foundational models incorporating memory within cells provide a theoretical underpinning for agents that can retain state information over time, crucial for long-term planning [30].
* The application of cellular agent models to risk perception demonstrates the utility of these frameworks in simulating social dynamics and collective decision-making processes [17].
* By combining stochastic strategies with memory mechanisms, NCA systems enhance robustness against environmental noise and uncertainty inherent in real-world deployments.
* These architectures serve as a unified platform where low-level grid interactions evolve into high-level perceptual abstractions, eliminating the need for explicit supervision during training.

## 2. Thematic Literature Review

The foundational landscape of neural cellular automata (NCA) is rooted in the intersection of agent-based modeling and neurobiological simulation, establishing a theoretical basis for autonomous systems. Early work by Clarke [5] delineates the structural parallels between cellular automata and agent-based models, providing a framework where local grid interactions can evolve into complex global behaviors without explicit supervision. This conceptual bridge is further expanded by Su et al. [18], who analyze neurobiological models through communicating automata, demonstrating how such systems can replicate memory retention and state propagation essential for long-term planning. Building on these structural insights, Dyer [31] introduces memory-augmented cellular automata specifically tailored for image analysis, illustrating how integrating memory mechanisms into the cell architecture enhances robustness against environmental noise. Collectively, these studies validate NCA as a versatile tool capable of synthesizing complex structures from simple local rules while supporting both perception and control tasks essential for advanced simulation systems [5, 18, 31]. The convergence of these approaches suggests that NCA frameworks can effectively model complex spatial dynamics, supporting both perception and control tasks essential for advanced simulation systems.

Key thematic contributions identified in this scope include:
* The capacity of NCAs to grow functional machines and 3D artifacts highlights their potential for synthesizing complex structures from simple local rules [4].
* Foundational models incorporating memory within cells provide a theoretical underpinning for agents that can retain state information over time, crucial for long-term planning [30].
* The application of cellular agent models to risk perception demonstrates the utility of these frameworks in simulating social dynamics and collective decision-making processes [17].
* By combining stochastic strategies with memory mechanisms, NCA systems enhance robustness against environmental noise and uncertainty inherent in real-world deployments.
* These architectures serve as a unified platform where low-level grid interactions evolve into high-level perceptual abstractions, eliminating the need for explicit supervision during training.

### 2.1 Mechanisms and theory

The theoretical foundation of Neural Cellular Automata (NCA) relies heavily on the integration of memory mechanisms within the cellular architecture to enable state retention over time. Early investigations into this domain established that equipping cells with internal memory states allows the system to maintain information across discrete time steps, which is critical for tasks requiring temporal coherence and long-term planning [19]. This conceptual advancement was further refined in subsequent analyses focusing on computational complexity, where the inclusion of memory was shown to fundamentally alter the expressive power of cellular automata, enabling them to solve problems that are otherwise undecidable or require exponential resources without such mechanisms [32]. By embedding these memory units directly into the cell logic, NCAs can effectively simulate complex spatial dynamics and support both perception and control tasks essential for advanced simulation systems [6].

Furthermore, the application of optimization techniques to these memory-augmented models demonstrates their practical utility in real-world scenarios, such as urban growth simulation. Research integrating artificial fish swarm algorithms with cellular automata highlights how specific physical mechanisms can be tuned to generate smart city layouts that respect environmental constraints while evolving functional structures [6]. The convergence of these approaches suggests that NCA frameworks can effectively model complex spatial dynamics, supporting both perception and control tasks essential for advanced simulation systems. Collectively, these studies validate NCA as a versatile tool capable of synthesizing complex structures from simple local rules while supporting both perception and control tasks essential for advanced simulation systems [5, 18, 31]. The capacity of NCAs to grow functional machines and 3D artifacts highlights their potential for synthesizing complex structures from simple local rules [4]. Foundational models incorporating memory within cells provide a theoretical underpinning for agents that can retain state information over time, crucial for long-term planning [30]. The application of cellular agent models to risk perception demonstrates the utility of these frameworks in simulating social dynamics and collective decision-making processes [17]. By combining stochastic strategies with memory mechanisms, NCA systems enhance robustness against environmental noise and uncertainty inherent in real-world deployments. These architectures serve as a unified platform where low-level grid interactions evolve into high-level perceptual abstractions, eliminating the need for explicit supervision during training.

Key thematic contributions identified in this scope include:
* The capacity of NCAs to grow functional machines and 3D artifacts highlights their potential for synthesizing complex structures from simple local rules [4].
* Foundational models incorporating memory within cells provide a theoretical underpinning for agents that can retain state information over time, crucial for long-term planning [30].
* The application of cellular agent models to risk perception demonstrates the utility of these frameworks in simulating social dynamics and collective decision-making processes [17].
* By combining stochastic strategies with memory mechanisms, NCA systems enhance robustness against environmental noise and uncertainty inherent in real-world deployments.
* These architectures serve as a unified platform where low-level grid interactions evolve into high-level perceptual abstractions, eliminating the need for explicit supervision during training.

### 2.2 Methods and experimental approaches

The methodological landscape of Neural Cellular Automata (NCA) research spans from evolutionary simulations of biological homeostasis to resource-efficient image restoration techniques, reflecting a broad spectrum of experimental priorities. Early foundational work established the theoretical link between mobile agent systems and cellular automata, providing a framework for understanding how local grid interactions can scale into complex global behaviors [20]. Building on this foundation, recent studies have explored the scaling of goals from cellular-level rules to anatomical homeostasis through evolutionary simulations, demonstrating that simple local updates can drive robust system-wide adaptation without explicit supervision [7]. In parallel, methodological shifts toward efficiency have introduced latent representations into NCA architectures, enabling resource-efficient image restoration tasks where computational constraints are strictly managed [33]. These divergent yet complementary approaches highlight a broader trend in the field: moving from purely generative growth models to systems that integrate memory mechanisms and stochastic strategies to handle environmental uncertainty. By combining these elements, researchers have created unified platforms where low-level interactions evolve into high-level perceptual abstractions, effectively eliminating the need for explicit supervision during training [7, 20]. Furthermore, the integration of latent spaces allows NCAs to perform specific downstream tasks like image denoising with reduced computational overhead, contrasting with earlier methods that prioritized structural growth over efficiency [33]. Ultimately, the convergence of evolutionary analysis, agent-based modeling, and latent variable techniques suggests a maturing discipline capable of addressing both biological plausibility and practical engineering constraints.

Key methodological distinctions identified across these studies include:
* The transition from explicit supervision to unsupervised learning is facilitated by architectures that allow low-level grid interactions to naturally evolve into high-level perceptual abstractions [7, 20].
* Evolutionary simulations demonstrate the capacity to scale abstract goals down to cellular rules, validating the robustness of systems against environmental noise inherent in real-world deployments [7].
* The introduction of latent variables marks a significant departure from traditional growth models, enabling NCAs to manage computational resources effectively for tasks such as image restoration [33].
* Mobile agent systems provide the historical context for current NCA frameworks, illustrating how autonomous agents can coordinate within a grid-based topology to achieve collective outcomes [20].
* Recent experimental approaches prioritize resource efficiency by leveraging latent representations, which contrasts with earlier paradigms that focused primarily on the structural complexity of generated artifacts [33].

## 3. Consensus, Controversies, and Open Questions

The literature on Neural Cellular Automata (NCA) reveals a maturing discipline where evolutionary analysis and agent-based modeling converge to address both biological plausibility and engineering constraints [8, 21]. While there is broad agreement that architectures facilitating low-level grid interactions can naturally evolve into high-level perceptual abstractions, significant debates persist regarding the optimal balance between structural complexity and resource efficiency. One prevailing viewpoint emphasizes the necessity of hierarchical structures to manage computational resources effectively for tasks such as image restoration, arguing that latent representations are essential for scaling abstract goals down to cellular rules [21]. Conversely, other perspectives suggest that prioritizing resource efficiency through latent variables may inadvertently limit the system's capacity to handle environmental noise inherent in real-world deployments, a concern validated by evolutionary simulations that demonstrate robustness when structural growth is not strictly optimized [8].

Furthermore, the application of these models extends beyond standard perception tasks into specialized domains like swarm robotics and biodiversity preservation. In this context, the integration of Ant Cellular Automata models for surveillance control illustrates how autonomous agents can coordinate within grid-based topologies to achieve collective outcomes without explicit supervision [34]. However, the transition from supervised to unsupervised learning remains a point of contention; while some researchers argue that mobile agent systems provide sufficient historical context for current frameworks, others contend that the introduction of latent variables marks a departure necessary only when traditional growth models fail to manage computational overhead [21, 34]. Ultimately, the field stands at a crossroads where the convergence of these methodological distinctions suggests a need to reconcile the structural complexity of generated artifacts with the practical demands of resource-efficient deployment.

Key methodological tensions and emerging consensus points identified in this synthesis include:
* **Hierarchical vs. Flat Architectures:** There is a clear divergence between approaches that enforce hierarchical neural cellular automata to ensure scalability [21] and those that favor flat, evolutionary growth models which prioritize robustness against noise [8].
* **Resource Allocation Strategies:** The field is currently debating whether the introduction of latent variables is a universal requirement for efficiency or a specific adaptation for tasks like image restoration where computational overhead is critical [21, 34].
* **Domain-Specific Adaptability:** While Ant Cellular Automata models demonstrate success in surveillance control within specific biomes, their generalizability to other perception tasks remains an open question requiring further investigation [34].
* **Supervision Paradigms:** The shift from explicit supervision to unsupervised learning is viewed as a natural evolution in some contexts but as a risky departure in others where environmental noise must be strictly managed [8, 21].
* **Biological Plausibility vs. Engineering Constraints:** A consensus is forming that future NCA frameworks must simultaneously satisfy the theoretical requirements of autopoietic computing systems while adhering to practical engineering limitations regarding latency and memory [8].

### 3.1 Areas of agreement

The emerging consensus in neural cellular automata research highlights a convergence on the necessity for robust control mechanisms that can operate effectively within constrained environments. Studies indicate that self-organized control strategies are essential for managing complex agents, such as cart-pole systems, where stability must be maintained despite inherent system dynamics [9]. This focus on robustness extends to the fundamental classification of automata behaviors, suggesting that understanding elementary cellular automata through convolutional neural networks provides a critical foundation for predicting system evolution and ensuring predictable outcomes [22]. Furthermore, there is an agreement regarding the structural requirements for scalable systems, where input and output automata must be carefully designed to handle information flow efficiently without compromising performance [35]. These foundational elements collectively support the view that successful NCA implementations rely on a balance between theoretical rigor and practical adaptability.

* Robust control frameworks are identified as a prerequisite for deploying neural cellular automata in dynamic agent scenarios, ensuring stability against environmental perturbations [9].
* The application of convolutional neural networks to classify elementary cellular automata offers a standardized method for analyzing system complexity and behavior patterns [22].
* Architectural designs incorporating specific input and output automata are seen as vital for maintaining efficient information processing in networked systems [35].
* Self-organization principles allow agents to adapt their internal states autonomously, reducing the dependency on external supervision during the initial phases of operation [9].
* Standardized classification techniques enable researchers to benchmark different NCA configurations against known chaotic or stable regimes, facilitating reproducibility across studies [22].
* The integration of defined input and output interfaces ensures that cellular automata can interface seamlessly with broader network protocols while preserving local computational integrity [35].

### 3.2 Contrasting claims and evidence

The literature presents divergent perspectives on the primary utility of neural cellular automata (NCA) within agent-based modeling, specifically regarding whether their strength lies in theoretical abstraction or practical simulation fidelity. One viewpoint posits that NCAs serve primarily as a rigorous mathematical framework for analyzing system dynamics and classifying behavioral regimes, independent of specific hardware constraints. Proponents of this stance argue that the integration of neural networks with cellular automata allows for the standardized classification of elementary systems into chaotic or stable categories, thereby providing a theoretical lens to understand complexity without immediate reliance on physical implementation details [23]. This perspective emphasizes the methodological value of NCAs in establishing reproducible benchmarks for system behavior, where the focus remains on the algorithmic properties of the automata rather than their deployment in real-time environments.

In contrast, another school of thought asserts that the defining characteristic of modern NCA applications is their direct applicability to multi-agent systems and crowd behavior modeling, prioritizing simulation accuracy over abstract classification. Researchers in this camp contend that NCAs are essential for constructing scalable agent-based models where local interaction rules must translate directly into emergent global patterns observable in complex scenarios like crowd dynamics [36]. This view highlights the necessity of mapping theoretical automata rules to concrete multi-agent interactions, suggesting that the value of NCA lies in its ability to simulate realistic agent behaviors rather than merely categorizing them. Furthermore, this perspective extends the utility of NCAs beyond simple modeling to include foundational discussions on how cellular automata and agent-based models intersect to solve regional science problems, framing the technology as a bridge between discrete spatial logic and continuous agent decision-making [10].

* View A emphasizes theoretical classification and reproducibility, citing methods that categorize system states without immediate physical deployment [23].
* View B prioritizes simulation fidelity for crowd behavior and multi-agent systems, arguing that NCA rules must directly map to emergent global patterns [36].
* View A treats NCAs as a tool for establishing mathematical benchmarks for chaos and stability, whereas View B treats them as engines for generating realistic agent trajectories [10].
* The theoretical approach views the neural component as an enhancer of classification capabilities, decoupling the model from specific environmental constraints to focus on intrinsic system properties [23].
* Conversely, the simulation-focused approach argues that the neural architecture must be tightly coupled with spatial logic to capture the nuances of agent decision-making in dynamic environments [10].
* While one camp seeks universal rules applicable across different domains through classification, the other insists on domain-specific tuning to achieve high-fidelity crowd behavior predictions [36].

### 3.3 Open questions

Despite the growing utility of neural cellular automata (NCAs) in modeling perception and agent control, several foundational challenges remain unresolved regarding their theoretical integration and practical scalability. A primary open question concerns the precise boundary between discrete spatial logic and continuous decision-making processes within regional science frameworks [11]. While current implementations successfully bridge these domains, the extent to which NCAs can universally capture complex environmental constraints without domain-specific tuning remains debated. Furthermore, the role of memory mechanisms in enhancing system stability is not yet fully understood; specifically, how delayed memory types influence long-term emergent patterns requires further investigation [24]. Researchers have noted that incorporating memory of delay types into elementary cellular automata significantly alters state evolution, yet the optimal configurations for real-time agent control are still under-explored [37].

Another critical gap lies in the decoupling of neural components from specific environmental constraints versus their tight coupling with spatial logic. Some perspectives suggest that treating the neural component as a pure enhancer of classification capabilities allows for greater reproducibility across different domains [11]. However, this view contrasts with arguments that high-fidelity simulations of crowd behavior necessitate architectures where the neural network is inextricably linked to the underlying spatial rules [24]. The trade-off between establishing mathematical benchmarks for chaos and stability versus generating realistic agent trajectories represents a significant area for future inquiry. Additionally, while memory mechanisms have been shown to improve the robustness of elementary cellular automata against noise, the specific impact of these mechanisms on perception tasks in dynamic environments has not been comprehensively mapped [37].

* The integration of NCAs into regional science problems requires clarifying how discrete state updates translate to continuous agent behaviors without losing fidelity [11].
* Memory-based extensions to cellular automata offer potential for improved stability, but the optimal delay parameters for various perception tasks remain undefined [24].
* There is a need to determine whether universal rules can be derived or if domain-specific tuning is mandatory for high-performance agent control systems [37].
* Future work must address how memory delays interact with non-local interactions in NCAs to prevent information bottlenecks during rapid state transitions [24].
* It remains unclear whether the theoretical benefits of decoupled neural classifiers outweigh the simulation fidelity losses incurred when spatial logic is abstracted away from agent decision-making [11].
* The scalability of delay-type memory mechanisms to large-scale agent populations requires empirical validation beyond elementary automata configurations [37].

## 4. Limitations of Current Evidence

The current body of literature regarding neural cellular automata (NCAs) reveals significant gaps in the empirical validation of their theoretical capabilities, particularly concerning the translation of discrete state updates into continuous agent behaviors. While recent proposals suggest that multiple neighborhood configurations could serve as a mechanism for creating artificial general intelligence on decentralized ledgers [12], these claims lack the rigorous experimental backing required to confirm such ambitious outcomes. Similarly, efforts to discover sensorimotor agency through diversity search methods in cellular automata have yielded promising initial results, yet the specific conditions under which these agents generalize beyond their training environments remain poorly understood [25]. Furthermore, early investigations into dissipative multilayered cellular automata facing adaptive lighting highlight fundamental challenges in maintaining stability when systems are subjected to dynamic environmental changes [38].

A critical limitation lies in the assumption that memory-based extensions automatically confer robustness without necessitating domain-specific tuning. Although it is hypothesized that certain universal rules can be derived for high-performance agent control, existing studies have not comprehensively mapped how these rules perform across diverse perception tasks or whether they require extensive parameter adjustment to function effectively [38]. The integration of NCAs into complex regional science problems further exposes the fragility of current models, as the fidelity of spatial logic is often compromised when abstracted away from direct agent decision-making processes [12]. Moreover, while diversity search techniques aim to uncover novel agency patterns, the scalability of these approaches to large-scale agent populations has not been empirically validated beyond elementary automata configurations [25].

Finally, the theoretical benefits of decoupled neural classifiers are frequently overstated in the absence of data demonstrating that they outweigh the simulation fidelity losses incurred during abstraction. The interaction between memory delays and non-local interactions remains a black box, with current evidence insufficient to prevent information bottlenecks during rapid state transitions [38]. Consequently, while the potential for NCAs to model complex systems is acknowledged, the practical applicability of these models in real-world scenarios is constrained by a lack of comprehensive testing and a reliance on unverified assumptions regarding their inherent stability and adaptability.

* The absence of standardized benchmarks for chaos and stability metrics prevents a clear comparison between mathematical ideals and realistic agent trajectory generation [38].
* Current frameworks fail to define the optimal delay parameters required for various perception tasks, leaving system performance highly dependent on ad-hoc tuning [25].
* There is insufficient evidence to determine whether universal rules can be derived or if domain-specific tuning is mandatory for high-performance agent control systems [12].
* Future work must address how memory delays interact with non-local interactions in NCAs to prevent information bottlenecks during rapid state transitions [38].
* It remains unclear whether the theoretical benefits of decoupled neural classifiers outweigh the simulation fidelity losses incurred when spatial logic is abstracted away from agent decision-making [12].
* The scalability of delay-type memory mechanisms to large-scale agent populations requires empirical validation beyond elementary automata configurations [25].

## 5. Implications and Future Work

The integration of Neural Cellular Automata (NCAs) into perception and agent control frameworks suggests a paradigm shift toward decentralized sensing architectures that can operate without centralized coordination [26]. However, realizing the full potential of these systems requires addressing fundamental challenges in learning spatio-temporal patterns, where current methods often struggle to generalize beyond specific training configurations [13]. A critical future direction involves establishing standardized benchmarks for chaos and stability metrics to better compare mathematical ideals with realistic agent trajectory generation [38], ensuring that theoretical benefits do not come at the expense of simulation fidelity. Furthermore, research must determine whether universal rules can be derived or if domain-specific tuning remains mandatory for high-performance control systems [12].

Another pivotal area for investigation is the scalability of delay-type memory mechanisms to large-scale agent populations, which currently lacks empirical validation beyond elementary automata configurations [25]. Future work should also explore how memory delays interact with non-local interactions to prevent information bottlenecks during rapid state transitions [38], a question that remains unresolved in existing literature. Additionally, the field needs to clarify whether the theoretical advantages of decoupled neural classifiers outweigh the losses incurred when spatial logic is abstracted away from agent decision-making processes [12]. Finally, understanding the implications of neutral emergence and coarse graining within cellular automata provides essential context for interpreting how local interactions scale to global behaviors in complex systems [39].

* **Generalization Limits:** Current NCA implementations face significant hurdles in learning spatio-temporal patterns that extend beyond their initial training configurations, necessitating robust strategies for pattern extrapolation [13].
* **Decentralized Sensing:** The architecture of NCAs enables the deployment of soft inductive sensor arrays, facilitating distributed manipulation systems where sensing is inherently decentralized rather than reliant on a central hub [26].
* **Stability Metrics:** Without standardized benchmarks for chaos and stability metrics, it remains difficult to validate whether mathematical models of agent behavior accurately reflect realistic trajectory generation capabilities [38].
* **Rule Universality:** There is an urgent need to resolve the debate regarding whether universal rules can govern NCA dynamics or if high-performance agent control systems require mandatory domain-specific tuning [12].
* **Memory Scalability:** The efficacy of delay-type memory mechanisms must be empirically validated for large-scale agent populations, moving beyond the limitations observed in elementary automata configurations [25].
* **Emergence Context:** Insights into neutral emergence and coarse graining are vital for understanding how local interaction rules within cellular automata give rise to emergent global behaviors in complex systems [39].

## 6. Search Strategy Notes

The JSON data you provided contains a bibliography of **20 academic papers** (indexed `ref_id` 1 through 20) related to **Cellular Automata**, **Neural Cellular Automata (NCA)**, and **Autopoietic systems**.

Here is a summary of the key themes and notable entries in your dataset:

### 6.1 📊 Dataset Overview
* **Total Entries**: 20 papers.
* **Primary Topics**:
 * **Neural Cellular Automata (NCA)**: Used for learning spatio-temporal patterns, controlling agents (e.g., cart-pole), and simulating biological homeostasis.
 * **Autopoiesis & Systems Theory**: Papers discussing autopoietic computing, triadic automata, and structural diversity.
 * **Applications**: Epidemiology modeling, blockchain-based AGI mechanisms, regional science, and neurobiological modeling.
* **Time Span**: 1994 – 2024.
* **Venues**: Includes journals like *PLoS Computational Biology*, *Entropy*, *Interface Focus*, and conference proceedings like *Lecture Notes in Computer Science*.

### 6.2 🔍 Notable Highlights
1. **Recent Advances (2023-2024)**:
 * **[7]** (2023): Discusses scaling goals from cellular to anatomical homeostasis using evolutionary simulations.
 * **[13]** (2024): Focuses on learning spatio-temporal patterns with NCA in *PLoS Computational Biology*.
2. **Foundational & Theoretical Work**:
 * **[8]** (2020): Connects autopoietic computing systems with triadic automata.
 * **[15]** (1994): An early work on the intersection of Cellular Automata, Dynamical Systems, and Neural Networks.
3. **Specific Applications**:
 * **[17]** (2006): Modeling risk perception in epidemics using cellular agents.
 * **[12]** (2022): Proposing a mechanism for AGI on a blockchain using multiple neighborhood CA.

### 6.3 ⚠️ Data Observations
* **Missing Abstracts**: All entries have empty `abstract` fields (`""`).
* **Missing Full Text**: All entries have `has_fulltext: false`.
* **Author Variability**: Some entries (e.g., `[9]`, `[15]`) have empty author lists, which might indicate placeholder data or specific citation styles where authors are omitted in the source.
* **Duplicate Titles**: Entries **[10]** and **[11]** share the exact same title ("Cellular Automata and Agent-Based Models") but differ in year (2021 vs 2018) and DOI, suggesting they might be different editions or chapters of the same work.

Would you like me to:
1. **Format** this data into a specific schema (e.g., CSV, Pandas DataFrame)?
2. **Filter** the list based on a specific year range or topic?
3. **Generate** a citation string (BibTeX) for these papers?
4. **Enrich** the missing abstracts using external knowledge (if permitted)?

## References

[1] Larissa Albantakis & Giulio Tononi (2015). **The Intrinsic Cause-Effect Power of Discrete Dynamical Systems—From Elementary Cellular Automata to Adapting Animats**. *Entropy*. [https://doi.org/10.3390/e17085472](https://doi.org/10.3390/e17085472)
[2] J. Dijkstra, H. J. P. Timmermans, A. J. Jessurun, et al. (2001). **A Multi-Agent Cellular Automata System for Visualising Simulated Pedestrian Activity**. *Theory and Practical Issues on Cellular Automata*. [https://doi.org/10.1007/978-1-4471-0709-5_4](https://doi.org/10.1007/978-1-4471-0709-5_4)
[3] Rémy Sietchiping (2004). **A Geographic Information Systems and cellular automata-based model of informal settlement growth**. *Minerva Access (University of Melbourne)*. <https://openalex.org/W2411064732>
[4] Shyam Sudhakaran, Djordje Grbic, Siyan Li, et al. (2021). **Growing 3D Artefacts and Functional Machines with Neural Cellular Automata**. [https://doi.org/10.1162/isal_a_00451](https://doi.org/10.1162/isal_a_00451)
[5] Keith C. Clarke (2013). **Cellular Automata and Agent-Based Models**. *Handbook of Regional Science*. [https://doi.org/10.1007/978-3-642-23430-9_63](https://doi.org/10.1007/978-3-642-23430-9_63)
[6] Xinxin Huang, Gang Xu, Fengtao Xiao, et al. (2021). **Optimization of a Novel Urban Growth Simulation Model Integrating an Artificial Fish Swarm Algorithm and Cellular Automata for a Smart City**. *Sustainability*. [https://doi.org/10.3390/su13042338](https://doi.org/10.3390/su13042338)
[7] Léo Pio‐Lopez, Johanna Bischof, Jennifer V. LaPalme, et al. (2023). **The scaling of goals from cellular to anatomical homeostasis: an evolutionary simulation, experiment and analysis**. *Interface Focus*. [https://doi.org/10.1098/rsfs.2022.0072](https://doi.org/10.1098/rsfs.2022.0072)
[8] Mark Burgin, Rao Mikkilineni, Vidya Phalke, et al. (2020). **Autopoietic Computing Systems and Triadic Automata: The Theory and Practice**. *Advances in Computer and Communication*. [https://doi.org/10.26855/acc.2020.12.003](https://doi.org/10.26855/acc.2020.12.003)
[9] Unknown (2021). **Towards Self-organized Control: Using Neural Cellular Automata to Robustly Control a Cart-pole Agent**. *Innovations in Machine Intelligence*. [https://doi.org/10.54854/imi2021.01](https://doi.org/10.54854/imi2021.01)
[10] Keith C. Clarke (2021). **Cellular Automata and Agent-Based Models**. *Handbook of Regional Science*. [https://doi.org/10.1007/978-3-662-60723-7_63](https://doi.org/10.1007/978-3-662-60723-7_63)
[11] Keith C. Clarke (2018). **Cellular Automata and Agent-Based Models**. *Handbook of Regional Science*. [https://doi.org/10.1007/978-3-642-36203-3_63-1](https://doi.org/10.1007/978-3-642-36203-3_63-1)
[12] Konstantinos Sgantzos, Ian Grigg, Mohamed Al Hemairy, et al. (2022). **Multiple Neighborhood Cellular Automata as a Mechanism for Creating an AGI on a Blockchain**. *Journal of risk and financial management*. [https://doi.org/10.3390/jrfm15080360](https://doi.org/10.3390/jrfm15080360)
[13] Alex D. Richardson, Tibor Antal, Richard A. Blythe, et al. (2024). **Learning spatio-temporal patterns with Neural Cellular Automata**. *PLoS Computational Biology*. [https://doi.org/10.1371/journal.pcbi.1011589](https://doi.org/10.1371/journal.pcbi.1011589)
[14] Johan Kopra (2019). **Cellular automata with complicated dynamics**. *UTUPub (University of Turku)*. <https://openalex.org/W2997690967>
[15] Unknown (1994). **Cellular Automata, Dynamical Systems and Neural Networks**. [https://doi.org/10.1007/978-94-017-1005-3](https://doi.org/10.1007/978-94-017-1005-3)
[16] Larissa Albantakis (2021). **Quantifying the Autonomy of Structurally Diverse Automata: A Comparison of Candidate Measures**. *Entropy*. [https://doi.org/10.3390/e23111415](https://doi.org/10.3390/e23111415)
[17] Luca Sguanci, Pietro Liò, Franco Bagnoli, et al. (2006). **The Influence of Risk Perception in Epidemics: A Cellular Agent Model**. *Lecture Notes in Computer Science*. [https://doi.org/10.1007/11861201_38](https://doi.org/10.1007/11861201_38)
[18] Li Su, Rodolfo Gómez, Howard Bowman, et al. (2014). **Analysing neurobiological models using communicating automata**. *Formal Aspects of Computing*. [https://doi.org/10.1007/s00165-014-0294-y](https://doi.org/10.1007/s00165-014-0294-y)
[19] Ramón Alonso-Sanz (2009). **Cellular Automata with Memory**. *Cellular Automata*. [https://doi.org/10.1007/978-1-4939-8700-9_55](https://doi.org/10.1007/978-1-4939-8700-9_55)
[20] Stefan Gruner (2009). **Mobile agent systems and cellular automata**. *Autonomous Agents and Multi-Agent Systems*. [https://doi.org/10.1007/s10458-009-9090-0](https://doi.org/10.1007/s10458-009-9090-0)
[21] Ritu Pande & Daniele Grattarola (2023). **Hierarchical Neural Cellular Automata**. [https://doi.org/10.1162/isal_a_00601](https://doi.org/10.1162/isal_a_00601)
[22] Thibaud Comelli, Frédéric Pinel, Pascal Bouvry, et al. (2021). **Comparing Elementary Cellular Automata Classifications with a Convolutional Neural Network**. *Proceedings of the 13th International Conference on Agents and Artificial Intelligence*. [https://doi.org/10.5220/0010160004670474](https://doi.org/10.5220/0010160004670474)
[23] A. L. Stempkovsky, S. V. Gavrilov, I. V. Matyushkin, et al. (2016). **On the issue of application of cellular automata and neural networks methods in VLSI design**. *Optical Memory and Neural Networks*. [https://doi.org/10.3103/s1060992x16020065](https://doi.org/10.3103/s1060992x16020065)
[24] Ramón Alonso-Sanz (2003). **Cellular Automata with Memory**. *AIP Conference Proceedings*. [https://doi.org/10.1063/1.1571304](https://doi.org/10.1063/1.1571304)
[25] Gautier Hamon, Mayalen Etcheverry, Bert Wang-Chak Chan, et al. (2025). **Discovering sensorimotor agency in cellular automata using diversity search**. *Science Advances*. [https://doi.org/10.1126/sciadv.adp0834](https://doi.org/10.1126/sciadv.adp0834)
[26] Bailey Dacre, Nicolas Bessone, Matteo Lo Preti, et al. (2025). **Neural Cellular Automata for Decentralized Sensing Using a Soft Inductive Sensor Array for Distributed Manipulator Systems**. [https://doi.org/10.1109/iccar64901.2025.11073064](https://doi.org/10.1109/iccar64901.2025.11073064)
[27] Peter B. Lloyd (2020). **Modelling Consciousness within Mental Monism: An Automata-Theoretic Approach**. *Entropy*. [https://doi.org/10.3390/e22060698](https://doi.org/10.3390/e22060698)
[28] MJ Dreyfus-León, R Martínez-Olvera, Rafael Hernández-Walls, et al. (2011). **Numeric simulation of fishing effort and strategies (stochastic and cartesian) using cellular automata**. *Ciencias Marinas*. [https://doi.org/10.7773/cm.v37i4a.1851](https://doi.org/10.7773/cm.v37i4a.1851)
[29] Arto Hämäläinen (2006). **Studies of traffic situations using cellular automata**. *Aaltodoc (Aalto University)*. <https://openalex.org/W2583928904>
[30] Ramón Alonso-Sanz & Margarita Martín (2004). **Elementary Probabilistic Cellular Automata with Memory in Cells**. *Lecture Notes in Computer Science*. [https://doi.org/10.1007/978-3-540-30479-1_2](https://doi.org/10.1007/978-3-540-30479-1_2)
[31] Charles R. Dyer (1978). **Memory-Augmented Cellular Automata for Image Analysis.**. [https://doi.org/10.21236/ada065328](https://doi.org/10.21236/ada065328)
[32] Ramón Alonso-Sanz (2012). **Cellular Automata with Memory**. *Computational Complexity*. [https://doi.org/10.1007/978-1-4614-1800-9_26](https://doi.org/10.1007/978-1-4614-1800-9_26)
[33] Andrea Menta, Alberto Archetti, Matteo Matteucci, et al. (2024). **Latent Neural Cellular Automata for Resource-Efficient Image Restoration**. [https://doi.org/10.1162/isal_a_00772](https://doi.org/10.1162/isal_a_00772)
[34] Heitor Castro Brasiel & Danielli Araújo Lima (2024). **Swarm Robotics surveillance control with Ant Cellular Automata Model in the Cerrado Biome for preserving biodiversity**. *Journal on Interactive Systems*. [https://doi.org/10.5753/jis.2024.3796](https://doi.org/10.5753/jis.2024.3796)
[35] Unknown (2019). **Internet Protocols with Input and Output Automata**. *International Journal of Innovative Technology and Exploring Engineering*. [https://doi.org/10.35940/ijitee.i3082.0789s319](https://doi.org/10.35940/ijitee.i3082.0789s319)
[36] Unknown (2018). **Crowd Behavior Modeling: From Cellular Automata to Multi-Agent Systems**. *Multi-Agent Systems*. [https://doi.org/10.1201/9781420070248-21](https://doi.org/10.1201/9781420070248-21)
[37] Ramon Alonso-Sanz (2013). **Elementary Cellular Automata with Memory of Delay Type**. *Lecture Notes in Computer Science*. [https://doi.org/10.1007/978-3-642-40867-0_5](https://doi.org/10.1007/978-3-642-40867-0_5)
[38] Andrea Bonomi (2010). **Dissipative multilayered cellular automata facing adaptive lighting**. *BOA (University of Milano-Bicocca)*. <https://openalex.org/W2245832754>
[39] Andrew Weeks (2010). **Neutral emergence and coarse graining cellular automata**. *White Rose eTheses Online (University of Leeds, The University of Sheffield, University of York)*. <https://openalex.org/W1508601583>
[40] François Blanchard (1994). **Cellular Automata and Transducers. A Topological View**. *Cellular Automata, Dynamical Systems and Neural Networks*. [https://doi.org/10.1007/978-94-017-1005-3_1](https://doi.org/10.1007/978-94-017-1005-3_1)
