#!/usr/bin/env python3
"""One cycle of the NCA auto-research loop.

1. Load benchmark gates and current scoreboard (NCA metrics, local LLM baselines).
2. Mark failing gates and map each failure to a Klaut search query.
3. Optionally call Klaut (search or report) and write a cycle artifact.
4. Emit prioritized hypothesis ids from docs/hypotheses-backlog.md.

Run from repo root on a machine that can reach engine (192.168.10.29) for live
Klaut calls. Offline mode uses SSD cache only.

Example:
  python scripts/auto_research.py --klaut-search
  python scripts/auto_research.py --klaut-report --topic "tool calling distillation 9B"
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


GAP_TO_QUERY = {
    "local_hermes_top1_nca": "session memory tool routing neural network small model",
    "curated_val_top1_nca": "next tool prediction imitation learning agent traces",
    "bfcl_ast_accuracy": "Berkeley function calling leaderboard small model distillation",
    "bfcl_irrelevance_accuracy": "tool calling abstention irrelevance detection",
    "nestful_full_sequence_accuracy": "nested API tool call sequence modeling",
    "tau2_airline_pass1": "tau-bench multi-turn tool use customer service",
    "tau2_retail_pass1": "tau2 bench retail airline tool agent",
    "appworld_dev_tgc": "AppWorld API program synthesis agent benchmark",
}


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
    parser.add_argument("--klaut-search", action="store_true", help="Run Klaut search for top failing gate")
    parser.add_argument("--klaut-report", action="store_true", help="Run Klaut synthesis report (slow)")
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

    klaut_payload = None
    if args.klaut_search and top_fail and not args.dry_run:
        import klaut_gateway

        query = GAP_TO_QUERY.get(top_fail, top_fail.replace("_", " "))
        klaut_payload = {"type": "search", "query": query, "result": klaut_gateway.search_papers(query)}
    if args.klaut_report and not args.dry_run:
        import klaut_gateway

        topic = args.topic or (GAP_TO_QUERY.get(top_fail or "", "neural cellular automata agent tools"))
        klaut_payload = {"type": "report", "topic": topic, "result": klaut_gateway.create_report(topic)}

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
        "suggested_hypotheses": suggested_h,
        "backlog_ranked": _parse_backlog()[:8],
        "distillation_cap_b": gates_cfg.get("distillation_teacher_max_params_b"),
        "klaut": klaut_payload,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"cycle_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(cycle, indent=2) + "\n")
    (OUT_DIR / "latest.json").write_text(json.dumps({"path": str(out_path), **cycle}, indent=2) + "\n")

    print(json.dumps({k: cycle[k] for k in ("failing_gates", "blocked_gates", "passing_gates", "top_fail", "suggested_hypotheses")}, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
