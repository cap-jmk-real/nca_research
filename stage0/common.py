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


def maybe_research_backup(step: str) -> None:
    """Invoke ~/bin/research-backup-now when present (engine hook; no-op elsewhere)."""
    import subprocess

    hook = Path.home() / "bin" / "research-backup-now"
    if hook.is_file():
        subprocess.run([str(hook), step], check=False)
