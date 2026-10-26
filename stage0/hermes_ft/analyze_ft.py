"""Pool out-of-fold predictions, pick lr on Hermes-val, session-bootstrap CIs (paired)."""
import json, sys, glob, collections, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_ft import MODELS, OUT, SP, load, ckpt
from nca_text.data import Example
import train_eval as te
torch.set_num_threads(4); rng = np.random.default_rng(20261026); B = 10000
base = json.load(open(Path(te.__file__).with_name("labels.json"))); fold = json.load(open(SP / "folds.json"))
runs = {}
for p in glob.glob(str(OUT / "runs/*.json")):
    r = json.load(open(p)); m, s, v, lr, sd, k = r["name"].rsplit("_", 5) if False else (None,)*6
    parts = r["name"].split("_"); k = int(parts[-1][1:]); sd = int(parts[-2][1:]); lr = parts[-3][2:]; v = parts[-4]; s = parts[-5]; m = "_".join(parts[:-5])
    runs[(m, s, v, lr, sd, k)] = r
res = {"selection": {}, "variants": {}}
for v in ("drop", "other"):
    R = load(v); labels = base + (["other"] if v == "other" else []); y = np.array([labels.index(r["target_tool"]) for r in R])
    sess = np.array([r["session_id"] for r in R]); us, inv = np.unique(sess, return_inverse=True); mem = [np.where(inv == i)[0] for i in range(len(us))]
    f = np.array([fold[s_] for s_ in sess])
    def ci(x):
        bs = [np.concatenate([x[mem[i]] for i in rng.integers(0, len(us), len(us))]).mean() for _ in range(B)]
        return [round(100*np.percentile(bs, 2.5), 2), round(100*np.percentile(bs, 97.5), 2)]
    floor = np.zeros(len(y))
    for k in range(5):
        tr = (f != k) & (f != (k + 1) % 5); maj = np.bincount(y[tr]).argmax(); floor[f == k] = (y[f == k] == maj)
    V = {"n_rows": len(y), "n_sessions": len(us), "floor_top1": round(100*floor.mean(), 2), "floor_ci": ci(floor), "models": {}}
    C = {}
    # zero-shot reference with curated checkpoints (13 classes, other gold never predicted)
    ex = [Example(r["session_id"], "terminal", r["goal"], [(s_["tool"], s_["result"]) for s_ in r["recent_steps"]], "hermes") for r in R]
    for m, (arch, tag, _) in MODELS.items():
        te.LABELS = base; X, _ = te.pack(arch, ex); X = torch.from_numpy(X); acc = []
        for sd in range(3):
            mm = te.build(arch, X, 16, 4); mm.load_state_dict(torch.load(ckpt(tag, sd), map_location="cpu")["state"]); mm.eval()
            with torch.no_grad(): acc.append((mm(X).argmax(1).numpy() == y).astype(float))
        C[(m, "zeroshot")] = np.stack(acc)
        for s in ("scratch", "ft"):
            lrs = sorted({k[3] for k in runs if k[:3] == (m, s, v)})
            if not lrs: continue
            vals = {lr: np.mean([runs[(m, s, v, lr, sd, k)]["val_top1"] for sd in range(3) for k in range(5) if (m, s, v, lr, sd, k) in runs]) for lr in lrs}
            lr = max(vals, key=vals.get); res["selection"][f"{m}/{s}/{v}"] = {"val_by_lr": {a: round(100*b, 2) for a, b in vals.items()}, "chosen_lr": lr}
            acc = []
            for sd in range(3):
                a = np.full(len(y), np.nan)
                for k in range(5):
                    r = runs.get((m, s, v, lr, sd, k))
                    if r: a[r["test_rows"]] = (np.array(r["test_pred"]) == y[r["test_rows"]])
                acc.append(a)
            C[(m, s)] = np.stack(acc)
    for (m, s), A in C.items():
        if np.isnan(A).any(): V["models"][f"{m}/{s}"] = {"incomplete": int(np.isnan(A).sum())}; continue
        t1 = 100 * A.mean(1); mc = A.mean(0)
        d = {"top1_mean": round(float(t1.mean()), 2), "top1_sd": round(float(t1.std(ddof=1)), 2), "ci": ci(mc), "diff_floor": round(100*(mc-floor).mean(), 2), "diff_floor_ci": ci(mc - floor)}
        if s != "zeroshot" and (m, "zeroshot") in C:
            z = C[(m, "zeroshot")].mean(0); d["diff_zeroshot"] = round(100*(mc-z).mean(), 2); d["diff_zeroshot_ci"] = ci(mc - z)
        if s == "ft" and (m, "scratch") in C:
            z = C[(m, "scratch")].mean(0); d["diff_scratch"] = round(100*(mc-z).mean(), 2); d["diff_scratch_ci"] = ci(mc - z)
        V["models"][f"{m}/{s}"] = d
    best = {}
    for s in ("scratch", "ft"):
        ks = [m for m in MODELS if (m, s) in C and not np.isnan(C[(m, s)]).any()]
        if not ks: continue
        top = max(ks, key=lambda m: C[(m, s)].mean()); best[s] = top
        for m in ks:
            if m != top: x = C[(top, s)].mean(0) - C[(m, s)].mean(0); V["models"][f"{m}/{s}"][f"best_minus_this"] = [round(100*x.mean(), 2), ci(x)]
    V["best"] = best; res["variants"][v] = V
(OUT / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=0)[:9000])
