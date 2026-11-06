"""Stage 0 shared helpers: curated jsonl -> Examples, fixed splits, metrics."""
from __future__ import annotations
import hashlib, json, os, random, sys
from collections import Counter
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nca_text.data import Example  # noqa: E402

DATA = Path(os.environ.get("NCA_DATA_ROOT", Path.home() / "nca_data"))
SPLITS = DATA / "stage0_splits"
SPLIT_SEED = 20261026


def load_jsonl_examples(path: Path) -> list[Example]:
    out = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out.append(Example(r["session_id"], r["target_tool"], r.get("goal", ""),
                               [(s["tool"], s["result"]) for s in r.get("recent_steps", [])], "curated"))
    return out


def load_split(name: str, split: str) -> list[Example]:
    return load_jsonl_examples(SPLITS / name / f"{split}.jsonl")


def dist(labels) -> dict:
    c = Counter(labels); n = sum(c.values())
    return {k: {"n": v, "pct": round(100 * v / n, 2)} for k, v in c.most_common()}


def metrics(y_true: list[int], ranked: list[list[int]], n_classes: int) -> dict:
    n = len(y_true)
    top1 = sum(r[0] == y for y, r in zip(y_true, ranked)) / n
    top3 = sum(y in r[:3] for y, r in zip(y_true, ranked)) / n
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for y, r in zip(y_true, ranked) if r[0] == c and y == c)
        fp = sum(1 for y, r in zip(y_true, ranked) if r[0] == c and y != c)
        fn = sum(1 for y, r in zip(y_true, ranked) if r[0] != c and y == c)
        if tp + fn == 0:
            continue  # class absent from gold: excluded from macro-F1
        p = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn)
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return {"n": n, "top1": round(100 * top1, 2), "top3": round(100 * top3, 2),
            "macro_f1": round(100 * sum(f1s) / len(f1s), 2)}


def pct(xs, q):
    xs = sorted(xs); i = min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))
    return round(xs[i], 3)


def require_nca_data_root() -> Path:
    """E1/E2 preflight: NCA_DATA_ROOT must be set explicitly (no default), exist, and be writable."""
    raw = os.environ.get("NCA_DATA_ROOT", "").strip()
    if not raw:
        raise SystemExit(
            "NCA_DATA_ROOT is not set. Export a writable data directory on external storage "
            "(no default path). Example: export NCA_DATA_ROOT='/Volumes/Extreme SSD/nca_research'"
        )
    root = Path(raw).expanduser()
    if not root.exists():
        raise SystemExit(f"NCA_DATA_ROOT does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"NCA_DATA_ROOT is not a directory: {root}")
    probe = root / ".nca_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise SystemExit(f"NCA_DATA_ROOT is not writable: {root} ({exc})") from exc
    return root.resolve()


def research_backup_optional() -> bool:
    return os.environ.get("RESEARCH_BACKUP_OPTIONAL", "").strip() == "1"


def maybe_research_backup(step: str) -> None:
    """Run ~/bin/research-backup-now (mandatory on engine; skip only if RESEARCH_BACKUP_OPTIONAL=1)."""
    import subprocess

    hook = Path.home() / "bin" / "research-backup-now"
    optional = research_backup_optional()
    if not hook.is_file():
        if optional:
            return
        raise SystemExit(
            "Missing mandatory backup hook ~/bin/research-backup-now.\n"
            "Install on the engine host, or set RESEARCH_BACKUP_OPTIONAL=1 only for CI/dry-run."
        )
    proc = subprocess.run([str(hook), step], check=False)
    if proc.returncode != 0:
        if optional:
            return
        raise SystemExit(
            f"research-backup-now {step!r} failed with exit code {proc.returncode} "
            "(set RESEARCH_BACKUP_OPTIONAL=1 only for CI/dry-run to skip)."
        )
