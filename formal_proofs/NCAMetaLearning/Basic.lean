/-
  NCAMetaLearning.Basic: Core Definitions for Graph Neural Cellular Automata (GNCA)
  Architecture Meta-Learning Foundations in Lean 4.

  Authors: Julian M. Kleber
  Date: October 2026
-/

namespace NCAMetaLearning

-- Node indices
def NodeId (V : Nat) := Fin V

-- Core definition of a symmetric relational graph
structure SimpleGraph (V : Nat) where
  adj : Fin V → Fin V → Prop
  symm : ∀ (i j : Fin V), adj i j ↔ adj j i

end NCAMetaLearning
