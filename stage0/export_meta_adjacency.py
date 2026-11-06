#!/usr/bin/env python3
"""E3: export trained meta topology alpha/adjacency + diameters from E1 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage0"))

from graph_topology_export import meta_export_record

RESULTS = REPO / "stage0" / "stage0_results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=RESULTS / "e1_checkpoints",
        help="Directory with seed{N}_meta.pt from E1",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "e3_meta_adjacency_audit.json",
    )
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    import torch

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    records = []
    missing = []
    for seed in seeds:
        ckpt = args.checkpoint_dir / f"seed{seed}_meta.pt"
        if not ckpt.is_file():
            missing.append(str(ckpt))
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = blob.get("best_state") or blob.get("model")
        if state is None:
            raise SystemExit(f"No weights in {ckpt}")
        records.append(
            meta_export_record(
                state,
                seed=seed,
                checkpoint=str(ckpt),
                temperature=args.temperature,
            )
        )
        sidecar = args.checkpoint_dir / f"seed{seed}_meta_adjacency.json"
        sidecar.write_text(json.dumps(records[-1], indent=2))

    out = {
        "experiment": "E3",
        "source_checkpoints": str(args.checkpoint_dir),
        "seeds_requested": seeds,
        "seeds_exported": [r["seed"] for r in records],
        "missing_checkpoints": missing,
        "trained_meta_graphs": records,
        "note": "Directed diameter uses row-softmax A with edge j→i when A[i,j]>0 (nca_text.model matmul convention).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"out": str(args.out), "exported": len(records), "missing": missing}, indent=2))
    if missing and not records:
        raise SystemExit("No meta checkpoints found; run E1 with meta topology first.")


if __name__ == "__main__":
    main()
