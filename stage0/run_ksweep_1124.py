#!/usr/bin/env python3
"""Recurrence-depth sweep of the tuned tree GNCA (2026-11-24, review gap SPA-15).

For each benchmark the configuration selected for 'gnca_tree' by stage0/run_rigour_1110.py is kept
FIXED and only the number of weight-tied steps K is changed (K in KS, default 1,2,8). The selected K
differs per benchmark (Hermes 2, BFCL 6, ToolBench 4, API-Bank 2), so the sweep also re-runs the
selected K and every comparison is made within this sweep (same code path, same device).
Seeds 0..4, same data, same training loop as run_ablation_1110.py, test evaluated once. No search.
Inputs: RIG_OUT (selected configs, read-only). Output: KS_OUT/<bench>/K<k>/seed<s>.{json,npz}; resumable.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_rigour_1110 as RG  # noqa: E402

KS = tuple(int(k) for k in os.environ.get("KS", "1,2,8").split(","))
OUT = Path(os.environ["KS_OUT"])


def main():
    import torch
    from nca_text.model import GraphNCA
    benches = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else RG.BENCHES
    dev = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("KS_DEVICE", "cuda") == "cuda" else "cpu")
    torch.set_num_threads(int(os.environ.get("KS_THREADS", "2")))
    RG._ensure_torch_classes()
    r = RG.H.R()
    for bench in benches:
        best, _ = RG.chosen(bench, "gnca_tree"); cfg = dict(best["config"])
        data = RG.Data(bench, need_cell=True, need_tab=False, device=dev)
        for seed in RG.SEEDS_FINAL:
            for k in KS:
                fd = OUT / bench / f"K{k}"; fd.mkdir(parents=True, exist_ok=True)
                if (fd / f"seed{seed}.json").exists():
                    continue
                f = data.feats(seed)["cell"]
                Xs = {s: torch.from_numpy(f[s]).to(dev) for s in ("train", "val", "test")}
                torch.manual_seed(seed)
                net = GraphNCA(obs_c=Xs["train"].shape[1], hidden_c=cfg["hidden_c"], n_classes=r.N_CLASSES, steps=k,
                               seq_len=r.SEQ_LEN, topology="tree", update_hidden=cfg["update_hidden"]).to(dev)
                t0 = time.time()
                res = r.train_eval_neural(net, Xs["train"], torch.from_numpy(data.y["train"]).to(dev), Xs["val"], data.y["val"], Xs["test"],
                                          data.y["test"], data.labels_all, steps=cfg["train_steps"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                                          weight_decay=cfg["weight_decay"], label_smoothing=cfg["label_smoothing"], eval_every=100, seed=seed)
                corr = (res["ranked"][:, 0] == data.y["test"]).astype(np.int8)
                np.savez_compressed(fd / f"seed{seed}.npz", test_correct=corr)
                rec = {"K": k, "K_selected": cfg["K"], "seed": seed, "config": cfg, "selected_trial": best["trial"],
                       "val_top1": res["val_top1"], "test_top1": round(100.0 * float(corr.mean()), 2),
                       "n_params": int(res["n_params"]), "wall_s": round(time.time() - t0, 1), "device": str(dev)}
                (fd / f"seed{seed}.json").write_text(json.dumps(rec, indent=1))
                print(f"[{time.strftime('%H:%M:%S')}] KSWEEP {bench}/K{k} seed {seed} test={rec['test_top1']} val={rec['val_top1']}", flush=True)
                del net, Xs
                if dev.type == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
