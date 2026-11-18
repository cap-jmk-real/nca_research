#!/usr/bin/env bash
# Build the Lean development and print the axioms of every top-level theorem.
# Output: stage0/stage0_results/review_1110/lean_axioms.txt (read by
# scripts/build_review_tables_1110.py for \numLeanThms, \numLeanAxFree, \numLeanSorry).
# Usage (repo root): bash scripts/lean_print_axioms.sh [output-file]
set -euo pipefail
cd "$(dirname "$0")/../formal_proofs"
OUT="${1:-../stage0/stage0_results/review_1110/lean_axioms.txt}"
lake build
names=$(grep -hoE '^(theorem|lemma) [A-Za-z0-9_.'"'"']+' NCAMetaLearning/*.lean | awk '{print $2}')
tmp=$(mktemp --suffix=.lean -p .)
trap 'rm -f "$tmp"' EXIT
{
  echo "import NCAMetaLearning"
  for n in $names; do echo "#print axioms NCAMetaLearning.$n"; done
} > "$tmp"
lake env lean "$tmp" > "$OUT"
lake env lean --version >> "$OUT"
echo "theorems: $(grep -c 'axioms' "$OUT"), sorryAx: $(grep -c 'sorryAx' "$OUT" || true)"
