#!/usr/bin/env python3
"""Component ablations of the tuned tree GNCA (2026-11-10).

For each benchmark the configuration selected for 'gnca_tree' by stage0/run_rigour_1110.py (multi-seed
validation selection) is kept FIXED and one component is changed; seeds 0..4, same data, same training
loop, test evaluated once. No search for the variants (a fixed-configuration ablation).
  sym      : symmetric normalisation Q^{-1/2} S Q^{-1/2} instead of the row-normalised A = Q^{-1} S
  no_lap   : perception [x, Ax] without the (A - I)x block (linearly redundant, see Section 3.3)
  no_gate  : h <- h + delta (no sigmoid gate)
  untied   : separate (W1, W2) per step (message-passing GNN), everything else identical
Output: RIG_OUT/ablation/<bench>/<variant>/seed<k>.{json,npz}; resumable.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import run_rigour_1110 as RG  # noqa: E402

VARIANTS = ("sym", "no_lap", "no_gate", "untied")


def make(variant, cfg, obs_c):
    import torch, torch.nn as nn
    from nca_text.model import GraphNCA, build_adjacency
    RG._ensure_torch_classes()
    r = RG.H.R()
    kw = dict(obs_c=obs_c, hidden_c=cfg["hidden_c"], n_classes=r.N_CLASSES, steps=cfg["K"], seq_len=r.SEQ_LEN,
              topology="tree", update_hidden=cfg["update_hidden"])
    if variant == "untied":
        return RG.UntiedGNN(**kw)
    net = GraphNCA(**kw)
    if variant == "sym":
        A = build_adjacency("tree", r.SEQ_LEN); S = (A > 0).float(); d = S.sum(1).rsqrt()
        net.fixed_adj.copy_(d[:, None] * S * d[None, :])
    elif variant == "no_lap":
        C = net.channels
        net.w1 = nn.Conv1d(2 * C, cfg["update_hidden"], 1)
        orig = net._perceive
        net._perceive = lambda obs, h: orig(obs, h)[:, : 2 * C]
    elif variant == "no_gate":
        def evolve(obs, hidden=None, steps=None, net=net):
            b, _, L = obs.shape
            hidden = obs.new_zeros(b, net.hidden_c, L) if hidden is None else hidden
            traj = [hidden]
            for t in range(net.steps):
                feat = torch.relu(net.w1(net._perceive(obs, hidden + net.step_embed[t].view(1, -1, 1))))
                delta, _ = net.w2(feat).chunk(2, dim=1)
                hidden = hidden + delta; traj.append(hidden)
            return hidden, traj
        net.evolve = evolve
    return net


def main():
    import torch, torch.nn.functional as F
    benches = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else RG.BENCHES
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(4)
    r = RG.H.R()
    for bench in benches:
        best, _ = RG.chosen(bench, "gnca_tree"); cfg = best["config"]
        data = RG.Data(bench, need_cell=True, need_tab=False, device=dev)
        for seed in RG.SEEDS_FINAL:
            for v in VARIANTS:
                fd = RG.OUT / "ablation" / bench / v; fd.mkdir(parents=True, exist_ok=True)
                if (fd / f"seed{seed}.json").exists():
                    continue
                f = data.feats(seed)["cell"]
                Xs = {s: torch.from_numpy(f[s]).to(dev) for s in ("train", "val", "test")}
                torch.manual_seed(seed)
                net = make(v, cfg, Xs["train"].shape[1]).to(dev)
                t0 = time.time()
                res = r.train_eval_neural(net, Xs["train"], torch.from_numpy(data.y["train"]).to(dev), Xs["val"], data.y["val"], Xs["test"],
                                          data.y["test"], data.labels_all, steps=cfg["train_steps"], batch_size=cfg["batch_size"], lr=cfg["lr"],
                                          weight_decay=cfg["weight_decay"], label_smoothing=cfg["label_smoothing"], eval_every=100, seed=seed)
                corr = (res["ranked"][:, 0] == data.y["test"]).astype(np.int8)
                np.savez_compressed(fd / f"seed{seed}.npz", test_correct=corr)
                rec = {"variant": v, "seed": seed, "config": cfg, "selected_trial": best["trial"], "val_top1": res["val_top1"],
                       "test_top1": round(100.0 * float(corr.mean()), 2), "n_params": int(res["n_params"]), "wall_s": round(time.time() - t0, 1)}
                (fd / f"seed{seed}.json").write_text(json.dumps(rec, indent=1))
                print(f"[{time.strftime('%H:%M:%S')}] ABL {bench}/{v} seed {seed} test={rec['test_top1']} val={rec['val_top1']}", flush=True)
                del net, Xs; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
