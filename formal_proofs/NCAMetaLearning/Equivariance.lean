/-
  NCAMetaLearning.Equivariance: Formal Verification of Permutation Equivariance
  for Graph Neural Cellular Automata (GNCA) under Architecture Meta-Learning.

  Authors: Julian M. Kleber
  Date: October 2026
-/

import NCAMetaLearning.Topology

namespace NCAMetaLearning

/--
  A bijection (permutation) on Fin V.
-/
structure Permutation (V : Nat) where
  toFun : Fin V → Fin V
  invFun : Fin V → Fin V
  left_inv : ∀ x, invFun (toFun x) = x
  right_inv : ∀ x, toFun (invFun x) = x

/--
  Permuted graph adjacency under permutation σ.
-/
def permuteAdj (V : Nat) (adj : Fin V → Fin V → Prop) (σ : Permutation V) :
  Fin V → Fin V → Prop :=
  fun i j => adj (σ.invFun i) (σ.invFun j)

theorem permuteAdj_symm (V : Nat) (G : SimpleGraph V) (σ : Permutation V) :
  ∀ i j, permuteAdj V G.adj σ i j ↔ permuteAdj V G.adj σ j i := by
  intro i j
  dsimp [permuteAdj]
  exact G.symm (σ.invFun i) (σ.invFun j)

/--
  Permuted SimpleGraph.
-/
def permuteGraph (V : Nat) (G : SimpleGraph V) (σ : Permutation V) : SimpleGraph V :=
  { adj := permuteAdj V G.adj σ,
    symm := permuteAdj_symm V G σ }

/--
  Neighborhood indicator predicate: node j is in neighborhood of i in graph G.
-/
def inNeighborhood (V : Nat) (G : SimpleGraph V) (i j : Fin V) : Prop :=
  G.adj i j

/--
  Theorem 4 (Neighborhood Permutation Equivariance):
  Node j is in the neighborhood of node i in G iff
  σ(j) is in the neighborhood of σ(i) in the permuted graph (permuteGraph G σ).
-/
theorem neighborhood_equivariance (V : Nat) (G : SimpleGraph V) (σ : Permutation V)
  (i j : Fin V) :
  inNeighborhood V G i j ↔ inNeighborhood V (permuteGraph V G σ) (σ.toFun i) (σ.toFun j) := by
  dsimp [inNeighborhood, permuteGraph, permuteAdj]
  rw [σ.left_inv i, σ.left_inv j]

/--
  Path Equivariance:
  A path of length k exists between u and v in G iff
  a path of length k exists between σ(u) and σ(v) in permuteGraph G σ.
-/
theorem path_equivariance (V : Nat) (G : SimpleGraph V) (σ : Permutation V)
  (k : Nat) (u v : Fin V) :
  Path V G k u v → Path V (permuteGraph V G σ) k (σ.toFun u) (σ.toFun v) := by
  intro p
  induction p with
  | refl x =>
      exact Path.refl (σ.toFun x)
  | @step k' u' v' w' p_prev h_adj ih =>
      have h_adj_perm : (permuteGraph V G σ).adj (σ.toFun v') (σ.toFun w') := by
        dsimp [permuteGraph, permuteAdj]
        rw [σ.left_inv v', σ.left_inv w']
        exact h_adj
      exact Path.step ih h_adj_perm

end NCAMetaLearning
