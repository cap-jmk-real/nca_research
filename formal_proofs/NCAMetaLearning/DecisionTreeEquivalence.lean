/-
  NCAMetaLearning.DecisionTreeEquivalence: Formal Verification of the Functional
  Simulation Duality, Fixed-Point Attractor Convergence, and Exponential
  Weight-Tying Compression Between Hierarchical Decision Trees and
  Graph Neural Cellular Automata.

  Authors: Julian M. Kleber
  Date: November 2026
-/

import NCAMetaLearning.Topology
import NCAMetaLearning.Contraction

namespace NCAMetaLearning

/--
  Inductive representation of a binary decision tree over input domain `α`
  and output representation `β`.
-/
inductive DecisionTree (α : Type) (β : Type) where
  | leaf (val : β) : DecisionTree α β
  | branch (gate : α → Bool) (left : DecisionTree α β) (right : DecisionTree α β)

/--
  Standard top-down sequential evaluation of a binary decision tree.
-/
def evalTree {α β : Type} : DecisionTree α β → α → β
  | DecisionTree.leaf v, _ => v
  | DecisionTree.branch g l r, x =>
      if g x then evalTree l x else evalTree r x

/--
  Depth (maximum root-to-leaf path length) of a binary decision tree.
-/
def treeDepth {α β : Type} : DecisionTree α β → Nat
  | DecisionTree.leaf _ => 0
  | DecisionTree.branch _ l r =>
      if treeDepth l ≤ treeDepth r then 1 + treeDepth r else 1 + treeDepth l

/--
  One synchronous step of a threshold-gated Cellular Automaton operating over
  a hierarchical tree state space: internal cells select between their left
  and right child neighborhoods via local gate `g x`, while leaf states are
  cellular fixed points.
-/
def cellularTreeStep {α β : Type} (x : α) : DecisionTree α β → DecisionTree α β
  | DecisionTree.leaf v => DecisionTree.leaf v
  | DecisionTree.branch g l r => if g x then l else r

/--
  Single-step synchronous cellular state invariant: applying one cellular step
  preserves the exact input-output mapping of the decision tree for every input `x`.
-/
theorem cellularTreeStep_preserves_eval {α β : Type} (T : DecisionTree α β) (x : α) :
  evalTree (cellularTreeStep x T) x = evalTree T x := by
  cases T with
  | leaf v => rfl
  | branch g l r =>
      dsimp [cellularTreeStep, evalTree]
      cases g x <;> rfl

/--
  Iterated K-step synchronous cellular tree update preserves exact tree evaluation.
-/
theorem iterate_cellular_tree_preserves_eval {α β : Type}
  (T : DecisionTree α β) (x : α) (K : Nat) :
  evalTree (iterateCellular (cellularTreeStep x) K T) x = evalTree T x := by
  induction K with
  | zero => rfl
  | succ k ih =>
      dsimp [iterateCellular]
      rw [cellularTreeStep_preserves_eval]
      exact ih

/--
  Exact K-step cellular convergence to a fixed-point leaf attractor:
  For any binary decision tree `T` and step budget `K ≥ treeDepth T`,
  `K` synchronous cellular transitions contract `T` to the fixed-point leaf
  state `DecisionTree.leaf (evalTree T x)`.
-/
theorem cellular_tree_converges_in_depth {α β : Type}
  (T : DecisionTree α β) (x : α) (K : Nat) (hK : treeDepth T ≤ K) :
  iterateCellular (cellularTreeStep x) K T = DecisionTree.leaf (evalTree T x) := by
  induction K generalizing T with
  | zero =>
      cases T with
      | leaf v => rfl
      | branch g l r =>
          dsimp [treeDepth] at hK
          split at hK <;> omega
  | succ k ih =>
      cases T with
      | leaf v =>
          dsimp [iterateCellular, evalTree]
          have h_zero : treeDepth (DecisionTree.leaf v : DecisionTree α β) ≤ k := by
            dsimp [treeDepth]
            omega
          rw [ih (DecisionTree.leaf v) h_zero]
          rfl
      | branch g l r =>
          dsimp [treeDepth] at hK
          have hl : treeDepth l ≤ k := by
            split at hK <;> omega
          have hr : treeDepth r ≤ k := by
            split at hK <;> omega
          have h_comm : ∀ (m : Nat) (S : DecisionTree α β),
            iterateCellular (cellularTreeStep x) (m + 1) S =
            iterateCellular (cellularTreeStep x) m (cellularTreeStep x S) := by
            intro m S
            induction m generalizing S with
            | zero => rfl
            | succ m' ihm =>
                change cellularTreeStep x (iterateCellular (cellularTreeStep x) (m' + 1) S) =
                       cellularTreeStep x (iterateCellular (cellularTreeStep x) m' (cellularTreeStep x S))
                rw [ihm S]
          rw [h_comm k (DecisionTree.branch g l r)]
          dsimp [cellularTreeStep, evalTree]
          cases hgx : g x
          · exact ih r hr
          · exact ih l hl

/--
  Every binary parent-child perception channel `(p, 2*p + 1)` and `(p, 2*p + 2)`
  used by a tree-structured cellular automaton is certified adjacent in `treeSimpleGraph V`.
-/
theorem tree_automaton_edges_in_gnca (V : Nat) (p c : Fin V)
  (h_child : c.val = 2 * p.val + 1 ∨ c.val = 2 * p.val + 2) :
  (treeSimpleGraph V).adj p c := by
  dsimp [treeSimpleGraph, treeGraphAdj, isTreeParent]
  rcases h_child with h_left | h_right
  · exact Or.inr (Or.inl (Or.inl h_left.symm))
  · exact Or.inr (Or.inl (Or.inr h_right.symm))

/--
  Number of internal split nodes in a complete binary decision tree of depth `K`: `2^K - 1`.
-/
def completeTreeInternalNodes (K : Nat) : Nat :=
  powTwo K - 1

/--
  Lemma: For all `K`, `powTwo K ≥ K + 1`.
-/
theorem powTwo_ge_succ (K : Nat) : powTwo K ≥ K + 1 := by
  induction K with
  | zero =>
      dsimp [powTwo]
      omega
  | succ k ih =>
      dsimp [powTwo]
      omega

/--
  Theorem 6 (Decision-Tree / Cellular-Automaton Simulation & Weight-Tying Compression):
  1. Exact Fixed-Point Simulation: For any binary decision tree `T` with `treeDepth T ≤ K`,
     `K` synchronous iterations of `cellularTreeStep` converge to the exact fixed-point leaf
     attractor `DecisionTree.leaf (evalTree T x)`.
  2. Exponential Parameter Compression: A complete binary decision tree of depth `K` with
     `d`-dimensional unshared node rules requires `(2^K - 1) * d` parameters, which is at
     least `K * d` (and grows exponentially as `2^K - 1`), whereas a weight-tied Graph NCA
     shares a single `d`-parameter cellular update rule across all cells and steps.
-/
theorem decision_tree_cellular_duality {α β : Type}
  (T : DecisionTree α β) (x : α) (K d : Nat) (hDepth : treeDepth T ≤ K) :
  (iterateCellular (cellularTreeStep x) K T = DecisionTree.leaf (evalTree T x)) ∧
  (completeTreeInternalNodes K * d ≥ K * d) := by
  constructor
  · exact cellular_tree_converges_in_depth T x K hDepth
  · dsimp [completeTreeInternalNodes]
    have h_pow := powTwo_ge_succ K
    have h_nodes : powTwo K - 1 ≥ K := by omega
    exact Nat.mul_le_mul_right d h_nodes

end NCAMetaLearning
