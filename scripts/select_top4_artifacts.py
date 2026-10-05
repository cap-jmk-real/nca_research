#!/usr/bin/env python3
"""Select the four best HF publish slots from paper SI + SSD metrics.

Writes results/hf/top4.json (SSD if available, else ./results/hf/).
Does not upload. Exit 0 always when --write succeeds; use --check-push
to fail if any slot lacks a promoter.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "hf_publish.json").read_text())
SSD = Path("/Volumes/Extreme SSD/nca_research/results/hf")
LOCAL = ROOT / "results" / "hf"
SI = ROOT / "paper" / "si" / "experiments"


def _out_dir() -> Path:
    if SSD.parent.is_dir():
        try:
            SSD.mkdir(parents=True, exist_ok=True)
            return SSD
        except OSError:
            pass
    LOCAL.mkdir(parents=True, exist_ok=True)
    return LOCAL


def _load_exps() -> list[dict]:
    rows = []
    if not SI.is_dir():
        return rows
    for path in sorted(SI.glob("EXP-*.json")):
        if path.name.startswith("._"):
            continue
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8", errors="replace")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _iter_variants(exp: dict, m: dict):
    """Normalize SI shapes: list[{variant,...}] or dict[name -> metrics]."""
    variants = exp.get("variants") or m.get("variants") or []
    always_default = m.get("val_always_terminal")
    if isinstance(variants, dict):
        for name, v in variants.items():
            if not isinstance(v, dict):
                continue
            yield {
                "variant": str(name),
                "val_top1": v.get("val_top1", v.get("best_val_top1")),
                "val_top1_always_terminal": v.get(
                    "val_top1_always_terminal", always_default
                ),
                "checkpoint": v.get("checkpoint") or v.get("out"),
                "steps": v.get("steps") or v.get("best_step"),
            }
        return
    if isinstance(variants, list):
        for v in variants:
            if isinstance(v, dict):
                yield v


def _metrics(exp: dict) -> dict:
    m = exp.get("metrics") or exp.get("results") or {}
    best_nca = None
    best_flat = None
    always_default = m.get("val_always_terminal")
    for v in _iter_variants(exp, m):
        name = str(v.get("variant") or "")
        top1 = v.get("val_top1", v.get("best_val_top1"))
        if top1 is None:
            continue
        always = v.get("val_top1_always_terminal", always_default)
        row = {
            "exp_id": exp.get("id") or exp.get("exp_id"),
            "variant": name,
            "val_top1": float(top1),
            "always_terminal": float(always) if always is not None else None,
            "path": v.get("checkpoint") or v.get("out") or exp.get("artifact"),
            "steps": v.get("steps") or exp.get("steps"),
        }
        # Prefer non-flat cellular variants for slot A; flat alone for slot B.
        if name in {"flat", "baseline_flat"}:
            if best_flat is None or row["val_top1"] > best_flat["val_top1"]:
                best_flat = row
        else:
            if best_nca is None or row["val_top1"] > best_nca["val_top1"]:
                best_nca = row
    # Single-model EXPs (mix13 / FineEnvs) without variants block.
    if best_nca is None:
        raw_best = m.get("best_val_top1")
        if isinstance(raw_best, dict):
            if raw_best.get("nca") is not None:
                best_nca = {
                    "exp_id": exp.get("id"),
                    "variant": "textnca",
                    "val_top1": float(raw_best["nca"]),
                    "always_terminal": float(
                        raw_best["always_terminal"]
                        if raw_best.get("always_terminal") is not None
                        else always_default
                    )
                    if (raw_best.get("always_terminal") is not None or always_default is not None)
                    else None,
                    "path": exp.get("artifact"),
                    "steps": raw_best.get("step") or (exp.get("training") or {}).get("steps"),
                }
            if raw_best.get("flat") is not None and best_flat is None:
                best_flat = {
                    "exp_id": exp.get("id"),
                    "variant": "flat",
                    "val_top1": float(raw_best["flat"]),
                    "always_terminal": float(raw_best["always_terminal"])
                    if raw_best.get("always_terminal") is not None
                    else None,
                    "path": exp.get("artifact"),
                    "steps": raw_best.get("step"),
                }
        elif raw_best is not None:
            best_nca = {
                "exp_id": exp.get("id"),
                "variant": "textnca",
                "val_top1": float(raw_best),
                "always_terminal": float(always_default) if always_default is not None else None,
                "path": exp.get("artifact"),
                "steps": (exp.get("training") or {}).get("steps"),
            }
    return {"nca": best_nca, "flat": best_flat, "raw": m}


def _score_nca(row: dict | None) -> float:
    if not row:
        return float("-inf")
    top1 = row["val_top1"]
    always = row.get("always_terminal")
    if always is None:
        return top1
    return top1 - always


def select(exps: list[dict]) -> dict:
    nca_cands = []
    flat_cands = []
    for exp in exps:
        status = str(exp.get("status") or "")
        if status in {"blocked", "claims_not_supported"}:
            continue
        # Accept primary/smoke/accepted/… — only hard rejects above.
        got = _metrics(exp)
        if got["nca"]:
            nca_cands.append(got["nca"])
        if got["flat"]:
            flat_cands.append(got["flat"])

    nca_cands.sort(key=_score_nca, reverse=True)
    flat_cands.sort(key=lambda r: r["val_top1"], reverse=True)

    # Specialist / dataset placeholders until EXP records exist
    specialist = {
        "exp_id": None,
        "note": "Fill when distill/harness EXP exists",
        "path": None,
        "score": None,
    }
    dataset = {
        "exp_id": None,
        "note": "Fill from compile_stats + scrub",
        "path": str(Path("/Volumes/Extreme SSD/nca_research/datasets/specialists")),
        "score": None,
    }

    slots = {
        "A": {
            **CFG["slots"]["A"],
            "winner": nca_cands[0] if nca_cands else None,
            "candidates_considered": len(nca_cands),
        },
        "B": {
            **CFG["slots"]["B"],
            "winner": flat_cands[0] if flat_cands else None,
            "candidates_considered": len(flat_cands),
        },
        "C": {**CFG["slots"]["C"], "winner": specialist},
        "D": {**CFG["slots"]["D"], "winner": dataset},
    }
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "publish_only_top4": True,
        "namespace": CFG["hf_namespace"],
        "slots": slots,
        "n_exps_scanned": len(exps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check-push", action="store_true")
    args = parser.parse_args()

    payload = select(_load_exps())
    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if args.write:
        out = _out_dir() / "top4.json"
        out.write_text(text)
        print(f"wrote {out}", flush=True)
    if args.check_push:
        missing = [
            k
            for k, v in payload["slots"].items()
            if not (v.get("winner") and (v["winner"].get("path") or v["winner"].get("exp_id")))
        ]
        # A/B need real winners; C/D may be pending early
        hard = [k for k in missing if k in {"A", "B"}]
        if hard:
            raise SystemExit(f"top4 missing hard slots: {hard}")


if __name__ == "__main__":
    main()
