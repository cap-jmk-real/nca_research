#!/usr/bin/env python3
"""One cycle of the NCA auto-research loop.

1. Load benchmark gates and current scoreboard (NCA metrics, local LLM baselines).
2. Mark failing gates and map the top failure to 2–3 short Klaut queries.
3. Optionally call Klaut search (hypothesis intake) and write a cycle artifact.
4. Emit prioritized hypothesis ids from docs/hypotheses-backlog.md.

Run from repo root on a machine that can reach engine (192.168.10.29) for live
Klaut calls. Offline mode uses SSD cache only.

Example:
  python scripts/auto_research.py --klaut-search

``--klaut-search`` writes the scoreboard and the SSD intake note. It does not
start or stop training. A full Klaut report is not requested from this path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
GATES_PATH = ROOT / "config" / "benchmark_gates.json"
BACKLOG_PATH = ROOT / "docs" / "hypotheses-backlog.md"
OUT_DIR = Path("/Volumes/Extreme SSD/nca_research/results/auto_research")

# Known artifact locations (external SSD)
METRICS_MIX13 = Path("/Volumes/Extreme SSD/nca_research/results/text_nca_1d_mix13_metrics.json")
METRICS_FINEENVS = Path("/Volumes/Extreme SSD/nca_research/results/text_nca_1d_fineenvs_metrics.json")
BENCHMARKS_RUN = Path("/Volumes/Extreme SSD/nca_research/results/benchmarks_run.json")
BENCHMARKS_1D = Path("/Volumes/Extreme SSD/nca_research/results/benchmarks_1d.json")
LOCAL_QWEN = Path("/Volumes/Extreme SSD/nca_research/results/local_next_tool_qwen25_1_5b.json")
CKPT_BEST = Path("/Volumes/Extreme SSD/nca_research/results/checkpoints/text_nca_1d_mix13_best.pt")


def _load_json(path: Path) -> dict | None:
    if path.is_file():
        return json.loads(path.read_text())
    return None


def _nca_scores() -> dict:
    for path in (METRICS_FINEENVS, METRICS_MIX13, BENCHMARKS_1D):
        data = _load_json(path)
        if not data:
            continue
        best_val = (data.get("best_val_top1") or data.get("final_val_top1") or {}).get("nca")
        best_local = (data.get("best_local_test_top1") or data.get("final_local_test_top1") or {}).get("nca")
        flat_val = (data.get("best_val_top1") or data.get("final_val_top1") or {}).get("flat")
        flat_local = (data.get("best_local_test_top1") or data.get("final_local_test_top1") or {}).get("flat")
        term_local = (data.get("best_local_test_top1") or data.get("final_local_test_top1") or {}).get(
            "always_terminal"
        )
        return {
            "source": str(path),
            "curated_val_top1_nca": best_val,
            "local_hermes_top1_nca": best_local,
            "curated_val_top1_flat": flat_val,
            "local_hermes_top1_flat": flat_local,
            "local_always_terminal": term_local,
        }
    return {}


def _benchmark_scores() -> dict:
    run = _load_json(BENCHMARKS_RUN)
    if not run:
        return {}
    bfcl = run.get("BFCL") or {}
    nest = run.get("NESTFUL") or {}
    return {
        "source": str(BENCHMARKS_RUN),
        "bfcl_ast_accuracy": bfcl.get("ast_accuracy"),
        "bfcl_irrelevance_accuracy": bfcl.get("irrelevance_accuracy"),
        "nestful_full_sequence_accuracy": nest.get("full_sequence_accuracy"),
        "tau2": run.get("tau2"),
        "appworld": run.get("AppWorld"),
    }


def _baseline_scores(gates_cfg: dict) -> dict:
    out: dict = {}
    qwen = _load_json(LOCAL_QWEN)
    if qwen:
        out["local_qwen2_5_1_5b_instruct"] = qwen.get("top1")
    nca = _nca_scores()
    if nca.get("local_always_terminal") is not None:
        out["local_always_terminal"] = nca["local_always_terminal"]
    return out


def _parse_backlog() -> list[dict]:
    if not BACKLOG_PATH.is_file():
        return []
    text = BACKLOG_PATH.read_text()
    rows: list[dict] = []
    for match in re.finditer(
        r"^### (H\d+[a-z]?)\s+—\s+(.+)$\n(?:.*\n)*?^- \*\*Priority:\*\* (\d+)",
        text,
        re.MULTILINE,
    ):
        rows.append({"id": match.group(1), "title": match.group(2).strip(), "priority": int(match.group(3))})
    rows.sort(key=lambda r: r["priority"])
    return rows


# Two or three short queries per gate. The retired single string
# "next tool prediction imitation learning agent traces" is not used:
# it kept returning the same off-domain papers with no code.
GAP_TO_QUERY = {
    # Anchor every query to neural cellular automata. Unanchored "top-1" or
    # "receptive field" searches come back as unrelated papers.
    "curated_val_top1_nca": [
        "text neural cellular automata",
        "next-tool accuracy neural cellular automata",
        "receptive field halting neural cellular automata",
    ],
    "local_hermes_top1_nca": [
        "local Hermes next-tool accuracy",
        "session tool-routing accuracy",
    ],
    "bfcl_ast_accuracy": [
        "BFCL AST function-call accuracy",
        "tool-call argument prediction",
    ],
    "bfcl_irrelevance_accuracy": [
        "BFCL irrelevance abstention",
        "tool-call abstain detection",
    ],
    "nestful_full_sequence_accuracy": [
        "NESTFUL nested tool sequence",
        "full-sequence tool-call accuracy",
    ],
    "tau2_airline_pass1": [
        "tau2 airline tool agent",
    ],
    "tau2_retail_pass1": [
        "tau2 retail tool agent",
    ],
    "appworld_dev_tgc": [
        "AppWorld API agent benchmark",
    ],
}


# GitHub repository search for the same gate. These look for code, not papers.
# Two queries stay under the unauthenticated search rate limit.
GITHUB_QUERIES = {
    "curated_val_top1_nca": [
        "neural cellular automata",
        "text neural cellular automata",
    ],
    "local_hermes_top1_nca": [
        "hermes function calling",
        "tool calling agent",
    ],
    "bfcl_ast_accuracy": [
        "berkeley function calling leaderboard",
        "function calling ast",
    ],
    "bfcl_irrelevance_accuracy": [
        "function calling irrelevance",
        "tool call abstention",
    ],
    "nestful_full_sequence_accuracy": [
        "NESTFUL nested function calling",
        "nested tool calling",
    ],
    "tau2_airline_pass1": [
        "tau2-bench airline",
    ],
    "tau2_retail_pass1": [
        "tau2-bench retail",
    ],
    "appworld_dev_tgc": [
        "appworld stonybrook",
    ],
}


# Adjacent fields. These are not the failing-gate phrase. They look for a
# mechanism we can port onto the text trace (attention perception, hierarchy).
ADJACENT_QUERIES = [
    "attention neural cellular automata",
    "TextNCA hierarchical local attention",
]


def adjacent_queries() -> list[str]:
    return list(ADJACENT_QUERIES)


def github_queries_for(gate: str) -> list[str]:
    """Short GitHub repository queries for one gate. At most two."""
    raw = GITHUB_QUERIES.get(gate, [])
    return [query.strip() for query in raw if isinstance(query, str) and query.strip()][:2]


def queries_for(gate: str) -> list[str]:
    """Short Klaut queries for one gate. At most three, each under 80 characters."""
    raw = GAP_TO_QUERY.get(gate, [])
    if isinstance(raw, str):
        raw = [raw]
    queries = [query.strip() for query in raw if isinstance(query, str) and query.strip()][:3]
    if not queries:
        queries = [
            "text neural cellular automata",
            "next-tool top-1 accuracy",
        ]
    for query in queries:
        if len(query) > 80:
            raise ValueError(f"Klaut query too long ({len(query)}): {query}")
        if "imitation learning agent traces" in query:
            raise ValueError(f"retired imitation-learning query: {query}")
    return queries


GAP_TO_HYPOTHESES = {
    "local_hermes_top1_nca": ["H2b", "H16", "H17", "H5"],
    "curated_val_top1_nca": ["H6", "H20", "H_long_range_prod"],
    "bfcl_ast_accuracy": ["H11", "H12", "H21"],
    "bfcl_irrelevance_accuracy": ["H11", "H22"],
    "nestful_full_sequence_accuracy": ["H23", "H24"],
    "tau2_airline_pass1": ["H25", "H13"],
    "tau2_retail_pass1": ["H25", "H13"],
    "appworld_dev_tgc": ["H26", "H27"],
}


def _evaluate_gates(gates_cfg: dict, scores: dict, baselines: dict) -> list[dict]:
    results = []
    gate_defs = gates_cfg.get("gates") or {}
    for name, spec in gate_defs.items():
        if spec.get("blocker_until"):
            results.append(
                {
                    "gate": name,
                    "status": "blocked",
                    "reason": spec["blocker_until"],
                }
            )
            continue
        value = scores.get(name)
        passed = value is not None and value == value
        detail = {"gate": name, "value": value, "min": spec.get("min")}
        if spec.get("min") is not None and value is not None:
            passed = passed and value >= float(spec["min"])
        for beat in spec.get("beats") or []:
            base = baselines.get(beat)
            if base is not None and value is not None:
                if not (value > base):
                    passed = False
                detail[f"beat_{beat}"] = base
        detail["status"] = "pass" if passed else "fail"
        results.append(detail)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--klaut-search",
        action="store_true",
        help="Search Klaut and GitHub for the top failing gate",
    )
    parser.add_argument(
        "--klaut-report",
        action="store_true",
        help="Run a Klaut report only when intake found a new architecture family with code",
    )
    parser.add_argument("--topic", default="", help="Override report topic")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gates_cfg = json.loads(GATES_PATH.read_text())
    nca = _nca_scores()
    bench = _benchmark_scores()
    scores = {**nca, **bench}
    baselines = _baseline_scores(gates_cfg)
    baselines["flat"] = nca.get("curated_val_top1_flat")
    gate_results = _evaluate_gates(gates_cfg, scores, baselines)

    failing = [g for g in gate_results if g.get("status") == "fail"]
    blocked = [g for g in gate_results if g.get("status") == "blocked"]
    passing = [g for g in gate_results if g.get("status") == "pass"]

    top_fail = failing[0]["gate"] if failing else None
    suggested_h = []
    if top_fail:
        suggested_h = GAP_TO_HYPOTHESES.get(top_fail, [])

    import hypothesis_intake

    harness = hypothesis_intake.harness_status(gate_results)
    klaut_payload = None
    if args.klaut_search and top_fail and not args.dry_run:
        klaut_payload = hypothesis_intake.run_intake(
            gate=top_fail,
            queries=queries_for(top_fail),
            gate_results=gate_results,
            github_queries=github_queries_for(top_fail),
            adjacent_queries=adjacent_queries(),
        )
        klaut_payload["type"] = "search"
    if args.klaut_report and not args.dry_run:
        allowed = bool(klaut_payload and klaut_payload.get("report_recommended"))
        if not allowed:
            skipped = {
                "type": "report",
                "skipped": "no new architecture family with code",
                "report_ran": False,
            }
            if klaut_payload is None:
                klaut_payload = skipped
            else:
                klaut_payload["report"] = skipped
        else:
            import klaut_gateway

            topic = args.topic or queries_for(top_fail or "")[0]
            klaut_payload["report"] = {
                "type": "report",
                "topic": topic,
                "result": klaut_gateway.create_report(topic),
            }
            klaut_payload["report_ran"] = True

    cycle = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gates_path": str(GATES_PATH),
        "checkpoint_best": str(CKPT_BEST) if CKPT_BEST.is_file() else None,
        "scores": scores,
        "baselines": baselines,
        "gate_results": gate_results,
        "failing_gates": [f["gate"] for f in failing],
        "blocked_gates": [b["gate"] for b in blocked],
        "passing_gates": [p["gate"] for p in passing],
        "top_fail": top_fail,
        "queries": queries_for(top_fail) if top_fail else [],
        "github_queries": github_queries_for(top_fail) if top_fail else [],
        "suggested_hypotheses": suggested_h,
        "harness_status": harness,
        "backlog_ranked": _parse_backlog()[:8],
        "distillation_cap_b": gates_cfg.get("distillation_teacher_max_params_b"),
        "klaut": klaut_payload,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"cycle_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(cycle, indent=2) + "\n")
    (OUT_DIR / "latest.json").write_text(json.dumps({"path": str(out_path), **cycle}, indent=2) + "\n")

    summary = {
        k: cycle[k]
        for k in (
            "failing_gates",
            "blocked_gates",
            "passing_gates",
            "top_fail",
            "queries",
            "github_queries",
            "suggested_hypotheses",
        )
    }
    summary["harness_failing"] = [item["harness"] for item in harness["failing"]]
    summary["harness_unscored"] = [item["harness"] for item in harness["unscored"]]
    summary["harness_blocked"] = [item["harness"] for item in harness["blocked"]]
    if klaut_payload:
        summary["klaut_up"] = klaut_payload.get("klaut_up")
        summary["github_ok"] = klaut_payload.get("github_ok")
        summary["github_hits"] = klaut_payload.get("github_hits")
        summary["hypothesis_added"] = klaut_payload.get("hypothesis_added")
        summary["hits_preview"] = klaut_payload.get("hits_preview")
        summary["intake_path"] = klaut_payload.get("intake_path")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
