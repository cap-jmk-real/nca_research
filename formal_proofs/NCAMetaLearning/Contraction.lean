/-
  NCAMetaLearning.Contraction: Formal Verification of Contraction Mapping,
  Fixed-Point Attractor Uniqueness, and Exponential Convergence
  for Meta-Learned Neural Cellular Automata.

  Authors: Julian M. Kleber
  Date: October 2026
-/

import NCAMetaLearning.Basic

namespace NCAMetaLearning

/--
  Discrete metric space for cellular state topologies.
-/
structure DiscreteMetricSpace (α : Type) where
  dist : α → α → Nat
  dist_self : ∀ x, dist x x = 0
  dist_eq_zero : ∀ x y, dist x y = 0 → x = y
  dist_triangle : ∀ x y z, dist x z ≤ dist x y + dist y z

/--
  Strict contraction operator: maps states such that distinct states strictly decrease distance.
-/
structure ContractionOp {α : Type} (M : DiscreteMetricSpace α) where
  op : α → α
  contract : ∀ x y, x ≠ y → M.dist (op x) (op y) < M.dist x y

/--
  Fixed point definition.
-/
def isFixedPoint {α : Type} (op : α → α) (x : α) : Prop :=
  op x = x

/--
  Theorem 4a (Uniqueness of Cellular Fixed-Point Attractor):
  Under a meta-learned contraction operator, there exists at most one stable fixed point.
-/
theorem fixed_point_unique {α : Type} (M : DiscreteMetricSpace α) (C : ContractionOp M)
  (x y : α) (hx : isFixedPoint C.op x) (hy : isFixedPoint C.op y) : x = y := by
  by_cases h_eq : x = y
  · exact h_eq
  · have h_contra := C.contract x y h_eq
    dsimp [isFixedPoint] at hx hy
    rw [hx, hy] at h_contra
    omega

/--
  Dyadic Contraction operator with Lipschitz rate ≤ 1/2:
  2 * dist(T(x), T(y)) ≤ dist(x, y).
-/
structure DyadicContraction {α : Type} (M : DiscreteMetricSpace α) where
  op : α → α
  step_decay : ∀ x y, 2 * M.dist (op x) (op y) ≤ M.dist x y

/--
  Iterated application of cellular update operator: T^n(x).
-/
def iterateCellular {α : Type} (op : α → α) : Nat → α → α
  | 0, x => x
  | n + 1, x => op (iterateCellular op n x)

/--
  Powers of 2: 2^n.
-/
def powTwo : Nat → Nat
  | 0 => 1
  | n + 1 => 2 * powTwo n

/--
  Theorem 4b (Geometric Perturbation Decay / Exponential Convergence):
  Any two trajectories initialized with perturbation δ decay exponentially:
  2^n * dist(T^n(x), T^n(y)) ≤ dist(x, y).
-/
theorem trajectory_decay {α : Type} (M : DiscreteMetricSpace α) (C : DyadicContraction M)
  (n : Nat) (x y : α) :
  (powTwo n) * M.dist (iterateCellular C.op n x) (iterateCellular C.op n y) ≤ M.dist x y := by
  induction n with
  | zero =>
      dsimp [iterateCellular, powTwo]
      omega
  | succ m ih =>
      dsimp [iterateCellular, powTwo]
      have h_step := C.step_decay (iterateCellular C.op m x) (iterateCellular C.op m y)
      have h1 : (powTwo m) * (2 * M.dist (C.op (iterateCellular C.op m x)) (C.op (iterateCellular C.op m y))) ≤
                (powTwo m) * M.dist (iterateCellular C.op m x) (iterateCellular C.op m y) :=
        Nat.mul_le_mul_left (powTwo m) h_step
      have h_assoc := Nat.mul_assoc 2 (powTwo m) (M.dist (C.op (iterateCellular C.op m x)) (C.op (iterateCellular C.op m y)))
      have h_comm := Nat.mul_left_comm 2 (powTwo m) (M.dist (C.op (iterateCellular C.op m x)) (C.op (iterateCellular C.op m y)))
      rw [h_assoc, h_comm]
      exact Nat.le_trans h1 ih

end NCAMetaLearning
