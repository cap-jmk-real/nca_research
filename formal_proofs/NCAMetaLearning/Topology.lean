/-
  NCAMetaLearning.Topology: Formal Verification of Graph Neural Cellular Automata
  Architecture Meta-Learning and Receptive Field Diameter Bounds.

  Authors: Julian M. Kleber
  Date: October 2026
-/

import NCAMetaLearning.Basic

namespace NCAMetaLearning

/--
  1D Linear lattice adjacency: nodes are connected if they are adjacent or equal.
-/
def lineGraphAdj (V : Nat) : Fin V → Fin V → Prop :=
  fun i j => (i.val + 1 = j.val) ∨ (j.val + 1 = i.val) ∨ (i.val = j.val)

theorem lineGraphAdj_symm (V : Nat) :
  ∀ (i j : Fin V), lineGraphAdj V i j ↔ lineGraphAdj V j i := by
  intro i j
  dsimp [lineGraphAdj]
  constructor
  · intro h
    rcases h with h1 | h2 | h3
    · exact Or.inr (Or.inl h1)
    · exact Or.inl h2
    · exact Or.inr (Or.inr h3.symm)
  · intro h
    rcases h with h1 | h2 | h3
    · exact Or.inr (Or.inl h1)
    · exact Or.inl h2
    · exact Or.inr (Or.inr h3.symm)

def lineSimpleGraph (V : Nat) : SimpleGraph V :=
  { adj := lineGraphAdj V,
    symm := lineGraphAdj_symm V }

/--
  Star Graph topology (Goal-Broadcasting): node 0 (Goal Anchor) connects to all nodes.
-/
def starGraphAdj (V : Nat) : Fin V → Fin V → Prop :=
  fun i j => (i.val = 0) ∨ (j.val = 0) ∨ (i.val = j.val)

theorem starGraphAdj_symm (V : Nat) :
  ∀ (i j : Fin V), starGraphAdj V i j ↔ starGraphAdj V j i := by
  intro i j
  dsimp [starGraphAdj]
  constructor
  · intro h
    rcases h with h1 | h2 | h3
    · exact Or.inr (Or.inl h1)
    · exact Or.inl h2
    · exact Or.inr (Or.inr h3.symm)
  · intro h
    rcases h with h1 | h2 | h3
    · exact Or.inr (Or.inl h1)
    · exact Or.inl h2
    · exact Or.inr (Or.inr h3.symm)

def starSimpleGraph (V : Nat) : SimpleGraph V :=
  { adj := starGraphAdj V,
    symm := starGraphAdj_symm V }

/--
  Theorem 1 (Topological Subsumption):
  Every 1D Cellular Automaton lattice topology is an element of the Graph NCA topology space.
-/
theorem theorem_subsumption_1d (V : Nat) :
  ∃ (G : SimpleGraph V), G.adj = lineGraphAdj V :=
  ⟨lineSimpleGraph V, rfl⟩

/--
  Path distance representation:
  A path of length k between node u and node v in graph G.
-/
inductive Path (V : Nat) (G : SimpleGraph V) : Nat → Fin V → Fin V → Prop where
  | refl (u : Fin V) : Path V G 0 u u
  | step {k : Nat} {u v w : Fin V} :
      Path V G k u v → G.adj v w → Path V G (k + 1) u w

/--
  Star graph has maximum diameter 2 between any two nodes via node 0 (for V > 0).
-/
theorem star_graph_diameter_two (V : Nat) (hV : V > 0) (u v : Fin V) :
  Path V (starSimpleGraph V) 2 u v := by
  let zero_fin : Fin V := ⟨0, hV⟩
  have p0 : Path V (starSimpleGraph V) 0 u u := Path.refl u
  have h_u_zero : (starSimpleGraph V).adj u zero_fin := by
    dsimp [starSimpleGraph, starGraphAdj]
    exact Or.inr (Or.inl rfl)
  have p1 : Path V (starSimpleGraph V) 1 u zero_fin := Path.step p0 h_u_zero
  have h_zero_v : (starSimpleGraph V).adj zero_fin v := by
    dsimp [starSimpleGraph, starGraphAdj]
    exact Or.inl rfl
  exact Path.step p1 h_zero_v

/--
  Lemma: In a 1D line graph, any path of length k from node u can reach at most
  indices in range [u - k, u + k].
-/
theorem line_graph_distance_bound (V : Nat) {k : Nat} {u w : Fin V} :
  Path V (lineSimpleGraph V) k u w →
  w.val ≤ u.val + k ∧ u.val ≤ w.val + k := by
  intro p
  induction p with
  | refl x =>
      omega
  | @step k' u' v' w' p_prev h_adj ih =>
      dsimp [lineSimpleGraph, lineGraphAdj] at h_adj
      rcases h_adj with h1 | h2 | h3
      · omega
      · omega
      · omega

/--
  Theorem 2 (Receptive Field Separation):
  For any 1D NCA with V nodes, reaching node (V-1) from node 0 requires at least (V-1) steps.
  In contrast, the meta-learned goal-broadcast star graph reaches all nodes in at most 2 steps.
-/
theorem theorem_receptive_field_separation (V : Nat) (hV : V ≥ 3) :
  (∀ k : Nat, Path V (lineSimpleGraph V) k ⟨0, by omega⟩ ⟨V - 1, by omega⟩ → k ≥ V - 1) ∧
  (∀ (u v : Fin V), Path V (starSimpleGraph V) 2 u v) := by
  constructor
  · intro k p
    have h_bound := line_graph_distance_bound V p
    have h_u_zero : (⟨0, by omega⟩ : Fin V).val = 0 := rfl
    have h_w_last : (⟨V - 1, by omega⟩ : Fin V).val = V - 1 := rfl
    rw [h_u_zero, h_w_last] at h_bound
    omega
  · intro u v
    exact star_graph_diameter_two V (by omega) u v

/--
  Hierarchical Tree DAG adjacency:
  Node p is parent of c if c = 2*p + 1 or c = 2*p + 2.
  Adjacency includes self-loops and undirected tree edges between parent and child.
-/
def isTreeParent (p c : Nat) : Prop :=
  (2 * p + 1 = c) ∨ (2 * p + 2 = c)

def treeGraphAdj (V : Nat) : Fin V → Fin V → Prop :=
  fun i j => (i.val = j.val) ∨ isTreeParent i.val j.val ∨ isTreeParent j.val i.val

theorem treeGraphAdj_symm (V : Nat) :
  ∀ (i j : Fin V), treeGraphAdj V i j ↔ treeGraphAdj V j i := by
  intro i j
  dsimp [treeGraphAdj]
  constructor
  · intro h
    rcases h with h1 | h2 | h3
    · exact Or.inl h1.symm
    · exact Or.inr (Or.inr h2)
    · exact Or.inr (Or.inl h3)
  · intro h
    rcases h with h1 | h2 | h3
    · exact Or.inl h1.symm
    · exact Or.inr (Or.inr h2)
    · exact Or.inr (Or.inl h3)

def treeSimpleGraph (V : Nat) : SimpleGraph V :=
  { adj := treeGraphAdj V,
    symm := treeGraphAdj_symm V }

/--
  Theorem 5 (Hierarchical Tree Subsumption):
  Hierarchical tree decision DAGs are valid topologies within the Graph NCA space.
-/
theorem theorem_tree_subsumption (V : Nat) :
  ∃ (G : SimpleGraph V), G.adj = treeGraphAdj V :=
  ⟨treeSimpleGraph V, rfl⟩

end NCAMetaLearning
