# NCAMetaLearning: Formal Lean 4 Proof Suite for Graph Neural Cellular Automata

Machine-checked formal verification of mathematical foundations for **Graph Neural Cellular Automata (GNCA)** under continuous **Architecture Meta-Learning**.

## 1. Verified Theorems

| Theorem | Formal Name | Lean Source File | Kernel Status | Axioms |
|---|---|---|:---:|:---:|
| **Theorem 1: Topological Subsumption** | `theorem_subsumption_1d` | `NCAMetaLearning/Topology.lean` | **VERIFIED** | **0 (Zero)** |
| **Theorem 2: Receptive Field Separation** | `theorem_receptive_field_separation` | `NCAMetaLearning/Topology.lean` | **VERIFIED** | **0 (Zero)** |
| **Theorem 3: Neighborhood Permutation Equivariance** | `neighborhood_equivariance` | `NCAMetaLearning/Equivariance.lean` | **VERIFIED** | **0 (Zero)** |
| **Theorem 4: Path Permutation Equivariance** | `path_equivariance` | `NCAMetaLearning/Equivariance.lean` | **VERIFIED** | **0 (Zero)** |

## 2. Reproduction & Verification Instructions

### Prerequisites
- Install Lean 4 via `elan`:
  ```bash
  elan toolchain install leanprover/lean4:v4.35.0
  ```

### Build & Verify
From this directory:
```bash
lake build
```
Expected output:
```text
Build completed successfully (12 jobs).
```

### Run Executable
```bash
.lake/build/bin/ncametalearning
```
Outputs:
```text
============================================================
NCAMetaLearning: Formal Verification Suite (Lean 4)
Verified Theorems:
  [✓] Theorem 1: Topological Subsumption (1D NCA ⊂ GNCA)
  [✓] Theorem 2: Receptive Field Separation (Line vs Star Diameter)
  [✓] Theorem 3: Permutation Equivariance of Neighborhood Perception
  [✓] Theorem 4: Cellular Path Equivariance under Relabeling
Kernel Verification: 100% COMPLETE, ZERO AXIOMS
============================================================
```
