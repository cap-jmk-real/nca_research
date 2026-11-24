#!/usr/bin/env python3
"""Write paper/tmlr/supplement_manifest.txt for the anonymised supplementary ZIP."""
from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "tmlr" / "supplement_manifest.txt"
BUNDLE = REPO / "paper" / "tmlr" / "bundle"
STATS_PUBLIC = REPO / "data" / "curated" / "stats_public.json"

SKIP_NAME = re.compile(
    r"(JULIAN_TODO|PAPER_REVIEW|REGRESSION_AUDIT|availability\.md|"
    r"klaut_ai_review|HANDOFF\.md|COVER_LETTER|AUTHOR_INFO)",
    re.I,
)
SKIP_PATH_PARTS = frozenset({"__pycache__", ".git"})

# Result subtrees cited in the paper (S1–S9); omit legacy engine logs with /home/ paths.
RESULT_SUBDIRS = (
    "review_1110",
    "rigour_1110",
    "hpo_1110",
    "ksweep_1124",
    "bfcl_official_1116",
    "baselines_20261026",
    "efficiency_1118",
)
RESULT_FILES = (
    "full_history_benchmarks.json",
    "reviewer_suite_summary.json",
    "learning_curves_n1_10.json",
    "ultra_grokking_results.json",
    "cell_activation_examples.json",
    "jacobian_audit.json",
    "step_norm_distribution_audit.json",
    "topology_diameter_audit.json",
    "parameter_scaling_investigation.json",
    "theorem6_bridge_audit.json",
)


def should_skip(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    if any(part in SKIP_PATH_PARTS for part in path.parts):
        return True
    if "bfcl_official_runner" in rel and path.suffix in (".sh", ".py"):
        return True
    if SKIP_NAME.search(rel):
        return True
    if rel == "data/curated/stats.json":
        return True
    if rel.endswith(("main.pdf", "main_tmlr.pdf")):
        return True
    if rel in ("scripts/check_supplement_anonymity.py", "scripts/write_supplement_manifest.py"):
        return True
    return False


def write_stats_public() -> None:
    src = REPO / "data" / "curated" / "stats.json"
    if not src.is_file():
        return
    data = json.loads(src.read_text())
    data = copy.deepcopy(data)
    data.get("source", {}).pop("local_path", None)
    STATS_PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    STATS_PUBLIC.write_text(json.dumps(data, indent=2) + "\n")


def bundle_formal_proofs() -> None:
    src = REPO / "formal_proofs"
    dst = BUNDLE / "formal_proofs"
    if dst.exists():
        shutil.rmtree(dst)
    if not src.is_dir():
        return
    author = re.compile(r"^\s*Authors?:.*$", re.M)

    def ignore(_dir: str, names: list[str]) -> list[str]:
        return [n for n in names if n == "__pycache__" or n.endswith(".olean")]

    shutil.copytree(src, dst, ignore=ignore)
    for lean in dst.rglob("*.lean"):
        text = lean.read_text(encoding="utf-8")
        lean.write_text(author.sub("", text), encoding="utf-8")


def bundle_efficiency_summary() -> None:
    eff = REPO / "stage0/stage0_results/efficiency_1118/efficiency_summary.json"
    if not eff.is_file():
        return
    out = BUNDLE / "stage0_results/efficiency_1118/efficiency_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(eff.read_text())
    data["source"] = "efficiency_1118 run (host paths omitted in anonymised bundle)"
    out.write_text(json.dumps(data, indent=2) + "\n")


def bundle_build_efficiency_script() -> None:
    src = REPO / "scripts/build_efficiency_1118.py"
    dst = BUNDLE / "scripts/build_efficiency_1118.py"
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    text = text.replace("/home/s4il0r/nca_e_logs/efficiency_1118", "<efficiency_1118 log directory>")
    dst.write_text(text)


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    write_stats_public()
    bundle_formal_proofs()
    bundle_efficiency_summary()
    bundle_build_efficiency_script()
    include: list[str] = []

    def add(p: Path) -> None:
        if p.is_file() and not should_skip(p):
            include.append(p.relative_to(REPO).as_posix())

    for rel in (
        "paper/tmlr/supplement.tex",
        "paper/tmlr/supplement_README.txt",
        "paper/tmlr/Makefile",
        "paper/tmlr/SUBMISSION_PACKAGE.md",
        "paper/tmlr/TMLR_STYLE_LICENSE_APACHE2.txt",
        "paper/tmlr/tmlr.sty",
        "paper/tmlr/tmlr.bst",
        "paper/tmlr/fancyhdr.sty",
        "scripts/check_tmlr_bare_floats.py",
        "scripts/build_protocol_pins_1124.py",
        "data/curated/stats_public.json",
    ):
        add(REPO / rel)
    for f in sorted((REPO / "scripts").glob("build_*.py")):
        if f.name != "build_efficiency_1118.py":
            add(f)
    add(BUNDLE / "scripts/build_efficiency_1118.py")
    for d in (REPO / "paper" / "generated", REPO / "paper" / "figures", REPO / "paper" / "si"):
        if d.is_dir():
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    add(f)
    sr = REPO / "stage0" / "stage0_results"
    for sub in RESULT_SUBDIRS:
        d = sr / sub
        if d.is_dir():
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.suffix != ".pt":
                    add(f)
    for name in RESULT_FILES:
        f = sr / name
        if f.is_file():
            if name == "jacobian_audit.json":
                continue
            add(f)
    if (sr / "review_1110/jacobian_audit.json").is_file():
        add(sr / "review_1110/jacobian_audit.json")
    add(BUNDLE / "stage0_results/efficiency_1118/efficiency_summary.json")
    if (BUNDLE / "formal_proofs").is_dir():
        for f in sorted((BUNDLE / "formal_proofs").rglob("*")):
            if f.is_file():
                include.append(f.relative_to(REPO).as_posix())
    for f in sorted((REPO / "stage0").glob("*.py")):
        if f.name != "score_local_v2c.py":
            add(f)
    for sub in ("efficiency", "hermes_ft"):
        d = REPO / "stage0" / sub
        if d.is_dir():
            for f in sorted(d.rglob("*.py")):
                add(f)
    bfcl_doc = REPO / "stage0/bfcl_official_runner/PROTOCOL.md"
    if bfcl_doc.is_file():
        add(bfcl_doc)
    for req in (REPO / "stage0/bfcl_official_runner").glob("requirements*.txt"):
        add(req)

    OUT.write_text("\n".join(sorted(set(include))) + "\n")
    print(f"manifest: {len(include)} files -> {OUT}")


if __name__ == "__main__":
    main()