#!/usr/bin/env python3
"""Resume-only: re-create the Hermes model artifacts that run_rigour_1110.py skipped while the engine
disk was below 6 GB (12:40-13:20 CET, 2026-11-10), for stage0/run_latency_1110.py.

For each model and seed in ART_SEEDS whose pickle is missing, the configuration recorded in
final/curated/<model>/seed<k>.json is refitted with the same seed and the same code path (fit(final=True)),
and the pickle is written. As a reproducibility check the refit's test top-1 is compared with the stored
final (written to RIG_OUT/artifacts/refit_check.jsonl). Existing pickles are never overwritten.
Usage: run_artifacts_1110.py --worker cpu|gpu
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_rigour_1110 as RG  # noqa: E402


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", choices=("cpu", "gpu"), required=True); ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if a.worker == "gpu" and torch.cuda.is_available() else "cpu")
    if a.worker == "gpu":
        torch.set_num_threads(4)
    models = RG.ALL_CPU if a.worker == "cpu" else RG.ALL_GPU
    data = RG.Data("curated", need_cell=True, need_tab=True, device=dev)
    chk = RG.ART / "refit_check.jsonl"
    for seed in RG.ART_SEEDS:
        for m in models:
            p = RG.ART / m / f"seed{seed}.pkl"
            if p.exists():
                continue
            rec = json.loads((RG.OUT / "final" / "curated" / m / f"seed{seed}.json").read_text())
            t0 = time.time()
            res = RG.fit(m, rec["config"], data, seed, True, dev, a.n_jobs)
            top1 = round(100.0 * float((res["_P"]["test"].argmax(1) == data.y["test"]).mean()), 2)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as fh:
                pickle.dump({"model": m, "config": rec["config"], "obj": res["_obj"], "featuriser": data._gt[seed],
                             "obs_c": int(data.feats(seed)["cell"]["train"].shape[1])}, fh)
            RG.append(chk, {"model": m, "seed": seed, "final_test_top1": rec["test_top1"], "refit_test_top1": top1, "wall_s": round(time.time() - t0, 1)})
            RG.log(f"ART {m} seed {seed} refit={top1} final={rec['test_top1']}")


if __name__ == "__main__":
    main()
