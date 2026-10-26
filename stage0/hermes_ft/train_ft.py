"""Hermes train/fine-tune grid, CPU only. Out-of-fold test predictions per (model, setup, variant, lr, seed).
Checkpoint selection and lr choice use Hermes-val only."""
import json, sys, os, glob, itertools, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA
from nca_text.data import Example
OUT = DATA / "stage0_results/hermes_ft_20261026"; (OUT / "runs").mkdir(parents=True, exist_ok=True)
SP = DATA / "stage0_splits/hermes_ft"
MODELS = {"nca_main": ("nca1d", "B_nca1d", True), "nca_nocw": ("nca1d", "H1_nocw", False), "nca_graph": ("graph", "T_graph", True),
          "nca_grid": ("nca2d", "T_nca2d", True), "flat": ("flat", "B_flat", True), "tx": ("tx", "B_tx", True)}
STEPS, BATCH, EVAL = 1500, 128, 50
def load(variant):
    p = SP.parent / "hermes_ood/test.jsonl" if variant == "drop" else SP / "other_all.jsonl"
    return [json.loads(l) for l in open(p)]
def ckpt(tag, seed):
    g = [d for d in glob.glob(str(DATA / f"stage0_runs/*_{tag}_curated_s{seed}")) if os.path.exists(d + "/best.pt")]
    return sorted(g)[0] + "/best.pt"
def job(args):
    model, setup, variant, lr, seed, k = args
    name = f"{model}_{setup}_{variant}_lr{lr}_s{seed}_f{k}"; fo = OUT / "runs" / f"{name}.json"
    if fo.exists(): return name
    import torch, torch.nn.functional as F
    torch.set_num_threads(1); torch.manual_seed(seed); rng = np.random.default_rng(seed * 100 + k)
    import train_eval as te
    base = json.load(open(Path(te.__file__).with_name("labels.json")))
    labels = base + (["other"] if variant == "other" else []); te.LABELS = labels
    arch, tag, cw_on = MODELS[model]
    R = load(variant); fold = json.load(open(SP / "folds.json"))
    ex = [Example(r["session_id"], "terminal", r["goal"], [(s["tool"], s["result"]) for s in r["recent_steps"]], "hermes") for r in R]
    X, _ = te.pack(arch, ex); X = torch.from_numpy(X); y = np.array([labels.index(r["target_tool"]) for r in R])
    f = np.array([fold[r["session_id"]] for r in R]); te_ix = np.where(f == k)[0]; va_ix = np.where(f == (k + 1) % 5)[0]; tr_ix = np.where((f != k) & (f != (k + 1) % 5))[0]
    m = te.build(arch, X, 16, 4)
    if setup == "ft":
        sd = torch.load(ckpt(tag, seed), map_location="cpu")["state"]; own = m.state_dict()
        for key, v in sd.items():
            if own[key].shape == v.shape: own[key] = v
            else: own[key][: v.shape[0]] = v  # head grows 13 -> 14 rows for other
        m.load_state_dict(own)
    cnt = np.bincount(y[tr_ix], minlength=len(labels)).astype(np.float32)
    cw = torch.tensor(1 / np.sqrt(cnt + 1)); cw = cw / cw.mean()
    if not cw_on: cw = torch.ones_like(cw)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2); yt = torch.from_numpy(y)
    def pred(ix):
        m.eval()
        with torch.no_grad(): o = m(X[ix])
        m.train(); return o
    best = (-1, 0, None)
    for step in range(1, STEPS + 1):
        for g in opt.param_groups: g["lr"] = lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS)))
        b = torch.from_numpy(rng.choice(tr_ix, BATCH))
        loss = F.cross_entropy(m(X[b]), yt[b], weight=cw); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % EVAL == 0 or (setup == "ft" and step == 1):
            va = float((pred(va_ix).argmax(1).numpy() == y[va_ix]).mean())
            if va > best[0]: best = (va, step, pred(te_ix).argmax(1).numpy())
    fo.write_text(json.dumps({"name": name, "val_top1": best[0], "best_step": best[1], "test_rows": te_ix.tolist(), "test_pred": best[2].tolist(), "n_train": len(tr_ix), "n_val": len(va_ix)}))
    return name
if __name__ == "__main__":
    J = list(itertools.product(MODELS, ["scratch", "ft"], ["drop", "other"], [1e-3, 3e-4], [0, 1, 2], range(5)))
    if len(sys.argv) > 1: J = J[: int(sys.argv[1])]
    with Pool(int(os.environ.get("NPROC", 12))) as p:
        for i, n in enumerate(p.imap_unordered(job, J), 1):
            if i % 20 == 0: print(i, len(J), n, flush=True)
