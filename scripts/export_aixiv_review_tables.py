#!/usr/bin/env python3
"""Build reconciliation tables for aiXiv review response from on-disk result JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = ROOT / "stage0" / "stage0_results" / "reviewer_suite_summary.json"
MANIFEST = ROOT / "stage0" / "stage0_results" / "bfcl_official_manifest.json"
OUT = ROOT / "stage0" / "stage0_results" / "aixiv_review_tables.json"


def main() -> int:
    root = json.loads(REVIEWER.read_text())
    hermes = root["hermes_factorial_and_contractive"]
    esc = hermes["escalation_calibration_3seed"]
    unconstrained = [s["unconstrained_step_ratio_mean"] for s in esc]
    damped = hermes["contractive_audit_3seed"]
    rho_damped = [s["perturbation_contraction_ratio_mean"] for s in damped]

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())

    tables = {
        "hermes_headline_reconciliation": [
            {
                "config": "Primary pre-specified (3-seed LS, K=8, SVD128, 3k steps/seed)",
                "split": "curated test (3530)",
                "top1": "68.92 ± 0.47%",
                "source": "reviewer_suite_summary factorial 3000-step checkpoint",
                "primary": True,
            },
            {
                "config": "Statistical parity K=4 (3-seed LS)",
                "split": "curated test",
                "top1": "68.90 ± 0.21%",
                "source": "paper Table tab:ablations",
                "primary": False,
            },
            {
                "config": "200k grokking (single schedule, LS)",
                "split": "curated test",
                "top1": "68.07%",
                "source": "paper Table tab:primary",
                "primary": False,
            },
            {
                "config": "10k LS deep tree snapshot",
                "split": "curated test",
                "top1": "67.37%",
                "source": "paper Table tab:primary",
                "primary": False,
            },
            {
                "config": "10k CW star (rare-tool focus)",
                "split": "curated test",
                "top1": "66.19%",
                "source": "paper Table tab:primary",
                "primary": False,
            },
            {
                "config": "Certified damped Eq. (14) only",
                "split": "curated test",
                "top1": "61.99 ± 0.14%",
                "source": "contractive_audit_3seed",
                "primary": False,
            },
        ],
        "compression_reference_baseline": {
            "reference_model": "Random Forest (Hermes tuned, 100 estimators, unpruned)",
            "n_params": 18_200_000,
            "gnca_headline_params": 59_182,
            "ratio_307x_note": "59.2k vs 18.2M in abstract; per-benchmark RF node counts differ (Table baseline-unified).",
        },
        "per_benchmark_rf_params_from_reviewer_suite": {
            bench: data["models"]["RandomForest"]["n_params"]
            for bench, data in root["benchmarks"].items()
            if "RandomForest" in data.get("models", {})
        },
        "deployed_residual_eq7_lipschitz_audit": {
            "equation": "residual-update Eq. (7)",
            "metric": "mean hidden step ratio ||h^{t+1}-h^t||/||h^t|| on escalation audit",
            "mean_across_seeds": round(sum(unconstrained) / len(unconstrained), 4),
            "per_seed": unconstrained,
            "dyadic_contractive": False,
            "note": "Values >1 indicate non-contraction; Theorems 4/7 apply only to damped Eq. (14).",
        },
        "damped_eq14_rho": {
            "mean_across_seeds": round(sum(rho_damped) / len(rho_damped), 4),
            "per_seed": rho_damped,
        },
        "bfcl_official_manifest_summary": {
            "n_scoring_entries": manifest.get("n_scoring_entries"),
            "bfcl_eval_version": manifest.get("bfcl_eval_version"),
        },
    }
    OUT.write_text(json.dumps(tables, indent=2) + "\n")
    print(json.dumps(tables, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
