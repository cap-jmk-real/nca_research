#!/usr/bin/env python3
"""Append input-parity train jobs to stage0/queues/*.txt (never rewrite — IDs are line:hash)."""
from __future__ import annotations

import argparse
from pathlib import Path

ARCHS = ("nca1d", "graph", "flat", "gru", "tx")
FEATS = ("hash48", "tfidf_svd128")
SEEDS = (0, 1, 2)
STEPS = 20000


def job(arch: str, feat: str, seed: int) -> str:
    tag = f"parity_{arch}_{feat}"
    feat_arg = f" --text_feat {feat}" if feat != "hash48" else ""
    return (
        f"~/nca_venv/bin/python train_eval.py --arch {arch} --seed {seed} --steps {STEPS}"
        f"{feat_arg} --tag {tag}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--queue",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "stage0" / "queues" / "train.txt",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    lines = [f"# input-parity batch {Path(__file__).name}"]
    for arch in ARCHS:
        for feat in FEATS:
            for seed in SEEDS:
                lines.append(job(arch, feat, seed))
    if args.dry_run:
        print("\n".join(lines))
        return
    with args.queue.open("a") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"appended {len(lines) - 1} jobs to {args.queue}")


if __name__ == "__main__":
    main()
