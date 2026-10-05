#!/usr/bin/env python3
"""Register experiment records and build paper SI tables.

Usage:
  python scripts/record_experiment.py backfill
  python scripts/record_experiment.py build-tables
  python scripts/record_experiment.py register --from-hypotheses-summary
  python scripts/record_experiment.py register --file path/to/record.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "paper" / "si"
EXP_DIR = SI / "experiments"
TABLE_DIR = SI / "tables"
SSD = Path("/Volumes/Extreme SSD/nca_research/results")
INDEX_PATH = SI / "experiments_index.json"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _next_id(day: str | None = None) -> str:
    day = day or date.today().strftime("%Y%m%d")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(EXP_DIR.glob(f"EXP-{day}-*.json"))
    n = 1
    if existing:
        last = existing[-1].stem.split("-")[-1]
        n = int(last) + 1
    return f"EXP-{day}-{n:03d}"


def _write_record(record: dict) -> Path:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    path = EXP_DIR / f"{record['id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    _rebuild_index()
    return path


def _rebuild_index() -> None:
    rows = []
    for path in sorted(EXP_DIR.glob("EXP-*.json")):
        rec = json.loads(path.read_text())
        rows.append(
            {
                "id": rec["id"],
                "title": rec.get("title"),
                "status": rec.get("status"),
                "date": rec.get("date"),
                "hypothesis_ids": rec.get("hypothesis_ids"),
                "path": str(path.relative_to(ROOT)),
            }
        )
    INDEX_PATH.write_text(json.dumps({"experiments": rows}, indent=2) + "\n")


def _chip_from_metrics(m: dict) -> dict:
    chip = m.get("chip") or {}
    return {
        "device": m.get("device") or chip.get("brand"),
        "brand": chip.get("brand"),
        "mem_gib": chip.get("mem_gib"),
        "ncpu": chip.get("ncpu"),
        "mps": chip.get("mps_available"),
    }


def backfill() -> list[Path]:
    """Create SI records from known SSD / repo metrics if missing titles."""
    written: list[Path] = []
    existing_titles = {
        json.loads(p.read_text()).get("title") for p in EXP_DIR.glob("EXP-*.json")
    }

    mix = SSD / "text_nca_1d_mix13_metrics.json"
    if mix.is_file():
        m = json.loads(mix.read_text())
        title = "1D TextNCA mix13 primary 150k (Hermes tools)"
        if title not in existing_titles:
            rec = {
                "id": _next_id("20261003"),
                "title": title,
                "status": "primary",
                "date": "2026-10-03",
                "hypothesis_ids": ["H4", "hermes-first-run"],
                "claims_supported": [
                    "1D TextNCA beats flat linear readout on curated Hermes val at 150k steps",
                    "Always-terminal is stronger than NCA on held-out local Hermes sessions",
                ],
                "claims_not_supported": [
                    "Local Hermes generalization without prior matching",
                ],
                "hardware": _chip_from_metrics(m),
                "software": {
                    "torch": (m.get("chip") or {}).get("torch"),
                    "compile": (m.get("microbench") or {}).get("use_compile"),
                },
                "data": {
                    "train_note": m.get("train_note"),
                    "val_note": m.get("val_note"),
                    "train_rows": m.get("train_rows"),
                    "val_rows": m.get("val_rows"),
                    "local_test_rows": m.get("local_test_rows"),
                    "n_classes": m.get("n_classes"),
                    "labels": m.get("labels"),
                    "train_hist": m.get("train_hist"),
                    "val_hist": m.get("val_hist"),
                    "local_test_hist": m.get("local_test_hist"),
                },
                "model": m.get("config"),
                "training": {
                    "optimizer": "AdamW",
                    "total_steps": m.get("step"),
                    "batch": m.get("batch"),
                    "lr": (m.get("config") or {}).get("lr"),
                    "weight_decay": (m.get("config") or {}).get("weight_decay"),
                    "seed": (m.get("config") or {}).get("seed"),
                    "resumed": m.get("resumed"),
                    "loss": "sqrt-inverse-frequency cross-entropy",
                },
                "metrics": {
                    "best_val_top1": m.get("best_val_top1"),
                    "final_val_top1": m.get("final_val_top1"),
                    "best_local_test_top1": m.get("best_local_test_top1"),
                    "final_local_test_top1": m.get("final_local_test_top1"),
                    "best_tracked": m.get("best_tracked"),
                },
                "baselines": {
                    "flat": "same observation, linear head",
                    "always_terminal": "constant predictor",
                },
                "artifacts": {
                    "metrics_json": str(mix),
                    "checkpoint_best": m.get("checkpoint_best"),
                    "checkpoint_last": m.get("checkpoint_last"),
                    "log": m.get("log"),
                    "repo_metrics_copy": "docs/text_nca_1d_mix13_metrics.json",
                    "architecture_doc": "docs/text-nca-architecture.md",
                },
                "repro": {
                    "command": "python -m nca_text.train",
                    "seed": (m.get("config") or {}).get("seed"),
                    "git_commit": _git_commit(),
                    "notes": "Primary 150k mix run; best val at step 26000.",
                },
                "si_notes": "Primary result for H4 on public curated val.",
            }
            written.append(_write_record(rec))

    fine = SSD / "text_nca_1d_fineenvs_metrics.json"
    if fine.is_file():
        m = json.loads(fine.read_text())
        title = "1D TextNCA FineEnvs-prior continuation 150k"
        if title not in existing_titles:
            rec = {
                "id": _next_id("20261005"),
                "title": title,
                "status": "primary",
                "date": "2026-10-05",
                "hypothesis_ids": ["H14", "H16"],
                "claims_supported": [
                    "Verifier-correct FineEnvs rows can be staged into the 13 Hermes labels",
                ],
                "claims_not_supported": [],
                "hardware": _chip_from_metrics(m),
                "software": {"torch": (m.get("chip") or {}).get("torch")},
                "data": {
                    "train_note": m.get("train_note"),
                    "val_note": m.get("val_note"),
                    "train_rows": m.get("train_rows"),
                    "val_rows": m.get("val_rows"),
                    "local_test_rows": m.get("local_test_rows"),
                    "n_classes": m.get("n_classes"),
                },
                "model": m.get("config"),
                "training": {
                    "optimizer": "AdamW",
                    "total_steps": m.get("step"),
                    "batch": m.get("batch"),
                    "sampler": "hermes-prior+fineenvs",
                    "init": "mix13 best weights, fresh AdamW at 1e-4",
                    "seed": (m.get("config") or {}).get("seed"),
                },
                "metrics": {
                    "best_val_top1": m.get("best_val_top1") or m.get("best_tracked"),
                    "final_val_top1": m.get("final_val_top1"),
                    "best_local_test_top1": m.get("best_local_test_top1"),
                    "final_local_test_top1": m.get("final_local_test_top1"),
                },
                "artifacts": {
                    "metrics_json": str(fine),
                    "checkpoint_best": m.get("checkpoint_best"),
                    "checkpoint_last": m.get("checkpoint_last"),
                    "log": m.get("log"),
                    "methods_doc": "docs/multi-harness-and-message-passing.md",
                },
                "repro": {
                    "command": "python -m nca_text.train  # with fineenvs_common_train.jsonl staged",
                    "seed": (m.get("config") or {}).get("seed"),
                    "git_commit": _git_commit(),
                },
                "related_experiments": [],
                "si_notes": "Continuation TensorBoard run tb/train_fineenvs.",
            }
            written.append(_write_record(rec))

    qwen = SSD / "local_next_tool_qwen25_1_5b.json"
    if qwen.is_file():
        m = json.loads(qwen.read_text())
        title = "Baseline Qwen2.5-1.5B-Instruct local Hermes next-tool"
        if title not in existing_titles:
            rec = {
                "id": _next_id("20261004"),
                "title": title,
                "status": "primary",
                "date": "2026-10-04",
                "hypothesis_ids": ["benchmark-gates"],
                "claims_supported": [
                    "Closed-set Qwen2.5-1.5B greedy next-tool top-1 is far below always-terminal and below TextNCA on the same local rows",
                ],
                "hardware": {"note": "local transformers load; see script"},
                "software": {"model_id": m.get("model")},
                "data": {
                    "n": m.get("n"),
                    "protocol": m.get("protocol"),
                    "gold": m.get("gold"),
                    "local_note": m.get("note"),
                },
                "model": {"model_id": m.get("model"), "decode": "greedy max_new_tokens=8"},
                "training": {"trained": False},
                "metrics": {
                    "top1": m.get("top1"),
                    "correct": m.get("correct"),
                    "unparsed": m.get("unparsed"),
                    "per_label_recall": m.get("per_label_recall"),
                    "always_terminal": m.get("always_terminal"),
                },
                "artifacts": {
                    "metrics_json": str(qwen),
                    "script": "scripts/score_local_next_tool.py",
                },
                "repro": {
                    "command": "python scripts/score_local_next_tool.py",
                    "seed": 0,
                    "git_commit": _git_commit(),
                },
                "si_notes": "Paired baseline for gate local_hermes_top1_nca.",
            }
            written.append(_write_record(rec))

    hyp = SSD / "hypothesis_tests" / "summary.json"
    if hyp.is_file():
        s = json.loads(hyp.read_text())
        title = f"Message-passing ablations curated equal-budget ({s.get('total_steps')} steps)"
        if title not in existing_titles:
            status = "smoke" if int(s.get("total_steps") or 0) < 50_000 else "primary"
            variants = {
                v["variant"]: {
                    "best_val_top1": v["best"]["val_top1"],
                    "best_step": v["best"]["step"],
                    "local_top1": v["local_top1"],
                    "n_params": v["n_params"],
                    "seconds": v["seconds"],
                }
                for v in s.get("variants") or []
            }
            rec = {
                "id": _next_id("20261005"),
                "title": title,
                "status": status,
                "date": s.get("date") or "2026-10-05",
                "hypothesis_ids": ["H2", "H3", "H4", "H_halt", "H_long_range"],
                "claims_supported": (
                    [
                        "At 8k steps, cellular baseline beats flat on curated val",
                        "Long-range goal edge slightly beats baseline at 8k",
                    ]
                    if status == "smoke"
                    else []
                ),
                "claims_not_supported": (
                    [
                        "8k-step carry and halt do not beat baseline on reset-val metric",
                        "8k results are not final architectural verdicts (need ≥150k)",
                    ]
                    if status == "smoke"
                    else []
                ),
                "hardware": {"device": s.get("device")},
                "software": {},
                "data": {
                    "train_note": s.get("train_note"),
                    "val_note": s.get("val_note"),
                    "train_rows": s.get("train_rows"),
                    "val_rows": s.get("val_rows"),
                    "n_classes": s.get("n_classes"),
                },
                "model": {
                    "variants": list(variants),
                    "hidden_c": 16,
                    "nca_steps": 4,
                    "seq_len": 5,
                },
                "training": {
                    "total_steps": s.get("total_steps"),
                    "batch": s.get("batch"),
                    "carry_sessions": s.get("carry_sessions"),
                    "carry_window": s.get("carry_window"),
                    "val_every": s.get("val_every"),
                    "seed": 0,
                    "lr": 1e-3,
                },
                "metrics": {
                    "variants": variants,
                    "regeneration": s.get("regeneration"),
                    "val_always_terminal": (s.get("variants") or [{}])[0].get(
                        "val_always_terminal"
                    ),
                },
                "artifacts": {
                    "summary_json": str(hyp),
                    "out_dir": s.get("out_dir"),
                    "repo_doc": "docs/hypothesis-tests.md",
                    "histories": {
                        name: str(SSD / "hypothesis_tests" / f"{name}_history.json")
                        for name in variants
                    },
                },
                "repro": {
                    "command": f"python -m nca_text.hypotheses --steps {s.get('total_steps')}",
                    "seed": 0,
                    "git_commit": _git_commit(),
                    "notes": "Mark smoke if steps < 50000; re-run at 150k for primary claims.",
                },
                "related_experiments": [],
                "si_notes": "Superseded for main-text architecture claims once EXP-* 150k ablations complete.",
            }
            written.append(_write_record(rec))

    return written


def register_from_hypotheses_summary(summary_path: Path | None = None) -> Path:
    summary_path = summary_path or (SSD / "hypothesis_tests" / "summary.json")
    s = json.loads(summary_path.read_text())
    steps = int(s.get("total_steps") or 0)
    status = "smoke" if steps < 50_000 else "primary"
    variants = {
        v["variant"]: {
            "best_val_top1": v["best"]["val_top1"],
            "best_step": v["best"]["step"],
            "local_top1": v["local_top1"],
            "n_params": v["n_params"],
            "seconds": v["seconds"],
            "checkpoint": v.get("checkpoint"),
        }
        for v in s.get("variants") or []
    }
    rec = {
        "id": _next_id(),
        "title": f"Message-passing ablations curated equal-budget ({steps} steps)",
        "status": status,
        "date": s.get("date") or date.today().isoformat(),
        "hypothesis_ids": ["H2", "H3", "H4", "H_halt", "H_long_range", "H6"],
        "claims_supported": [],
        "claims_not_supported": [],
        "hardware": {"device": s.get("device")},
        "software": {},
        "data": {
            "train_rows": s.get("train_rows"),
            "val_rows": s.get("val_rows"),
            "train_note": s.get("train_note"),
            "val_note": s.get("val_note"),
        },
        "model": {"variants": list(variants)},
        "training": {
            "total_steps": steps,
            "batch": s.get("batch"),
            "seed": 0,
            "long_training_floor": 150000,
        },
        "metrics": {"variants": variants, "regeneration": s.get("regeneration")},
        "artifacts": {
            "summary_json": str(summary_path),
            "out_dir": s.get("out_dir"),
        },
        "repro": {
            "command": f"python -m nca_text.hypotheses --steps {steps}",
            "seed": 0,
            "git_commit": _git_commit(),
        },
        "si_notes": "Auto-registered from hypotheses summary.",
    }
    # Fill claims from numbers when primary
    if status == "primary" and "baseline" in variants and "flat" in variants:
        b, f = variants["baseline"]["best_val_top1"], variants["flat"]["best_val_top1"]
        if b > f:
            rec["claims_supported"].append(
                f"Cellular baseline beats flat on curated val ({b:.4f} vs {f:.4f}) at {steps} steps"
            )
        else:
            rec["claims_not_supported"].append(
                f"Cellular baseline does not beat flat at {steps} steps ({b:.4f} vs {f:.4f})"
            )
    return _write_record(rec)


def build_tables() -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in sorted(EXP_DIR.glob("EXP-*.json"))]
    outs: list[Path] = []

    # Master index table
    lines = [
        "# SI Table S1 — Experiment registry",
        "",
        "Auto-generated by `scripts/record_experiment.py build-tables`.",
        "",
        "| ID | Date | Status | Title | Hypotheses |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in records:
        hyps = ", ".join(r.get("hypothesis_ids") or [])
        lines.append(
            f"| `{r['id']}` | {r.get('date')} | {r.get('status')} | {r.get('title')} | {hyps} |"
        )
    path = TABLE_DIR / "S1_experiment_registry.md"
    path.write_text("\n".join(lines) + "\n")
    outs.append(path)

    # Primary next-tool results
    lines = [
        "# SI Table S2 — Next-tool top-1 (primary / preliminary)",
        "",
        "Smoke runs are listed but **not** for main-text claims.",
        "",
        "| ID | Status | Setting | NCA / model | Flat | Always-terminal | Local NCA | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in records:
        m = r.get("metrics") or {}
        if "best_val_top1" in m and isinstance(m["best_val_top1"], dict):
            bv = m["best_val_top1"]
            bl = m.get("best_local_test_top1") or {}
            lines.append(
                f"| `{r['id']}` | {r['status']} | {r['title'][:48]} | "
                f"{_pct(bv.get('nca'))} | {_pct(bv.get('flat'))} | "
                f"{_pct(bv.get('always_terminal'))} | {_pct(bl.get('nca'))} | "
                f"step {bv.get('step', '')} |"
            )
        elif "variants" in m:
            for name, row in (m["variants"] or {}).items():
                lines.append(
                    f"| `{r['id']}` | {r['status']} | ablation `{name}` @ "
                    f"{(r.get('training') or {}).get('total_steps')} | "
                    f"{_pct(row.get('best_val_top1'))} | — | — | "
                    f"{_pct(row.get('local_top1'))} | best step {row.get('best_step')} |"
                )
        elif "top1" in m:
            lines.append(
                f"| `{r['id']}` | {r['status']} | {r['title'][:48]} | "
                f"{_pct(m.get('top1'))} | — | {_pct(m.get('always_terminal'))} | "
                f"{_pct(m.get('top1'))} | Qwen local baseline |"
            )
    path = TABLE_DIR / "S2_next_tool_top1.md"
    path.write_text("\n".join(lines) + "\n")
    outs.append(path)

    # Claims ledger
    lines = [
        "# SI Table S3 — Claims ledger",
        "",
        "| ID | Status | Supported claims | Not supported |",
        "| --- | --- | --- | --- |",
    ]
    for r in records:
        sup = "<br>".join(r.get("claims_supported") or []) or "—"
        nos = "<br>".join(r.get("claims_not_supported") or []) or "—"
        lines.append(f"| `{r['id']}` | {r['status']} | {sup} | {nos} |")
    path = TABLE_DIR / "S3_claims_ledger.md"
    path.write_text("\n".join(lines) + "\n")
    outs.append(path)

    # Master results narrative table for paper/
    master = ROOT / "paper" / "si" / "results_master.md"
    master.write_text(
        "\n".join(
            [
                "# Results master (paper + SI)",
                "",
                f"Updated from {len(records)} experiment records.",
                "",
                "See generated tables:",
                "",
                "- [S1 experiment registry](tables/S1_experiment_registry.md)",
                "- [S2 next-tool top-1](tables/S2_next_tool_top1.md)",
                "- [S3 claims ledger](tables/S3_claims_ledger.md)",
                "",
                "## Inclusion rules",
                "",
                "1. Main-text numbers cite `EXP-*` with `status: primary` (or `preliminary` with explicit caveat).",
                "2. `status: smoke` never supports architecture claims (budget < 50k steps).",
                "3. Every new daemon / hypotheses completion must call "
                "`python scripts/record_experiment.py register --from-hypotheses-summary` "
                "then `build-tables`.",
                "4. Distillation teachers ≤ 9B; record teacher id in `training` / `baselines`.",
                "",
            ]
        )
    )
    outs.append(master)
    return outs


def _pct(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.2f}%"
    except (TypeError, ValueError):
        return "—"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backfill")
    sub.add_parser("build-tables")
    p_reg = sub.add_parser("register")
    p_reg.add_argument("--from-hypotheses-summary", action="store_true")
    p_reg.add_argument("--file", type=Path)
    p_reg.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    if args.cmd == "backfill":
        paths = backfill()
        tables = build_tables()
        print(json.dumps({"wrote": [str(p) for p in paths], "tables": [str(t) for t in tables]}, indent=2))
        return
    if args.cmd == "build-tables":
        tables = build_tables()
        print(json.dumps({"tables": [str(t) for t in tables]}, indent=2))
        return
    if args.cmd == "register":
        if args.file:
            rec = json.loads(args.file.read_text())
            if "id" not in rec:
                rec["id"] = _next_id()
            path = _write_record(rec)
        elif args.from_hypotheses_summary:
            path = register_from_hypotheses_summary(args.summary)
        else:
            raise SystemExit("register needs --file or --from-hypotheses-summary")
        tables = build_tables()
        print(json.dumps({"record": str(path), "tables": [str(t) for t in tables]}, indent=2))


if __name__ == "__main__":
    main()
