"""Train/eval one NCA-family or flat model on fixed splits; log train/val/test
curves (grokking), pick best on val, report test + batch-1 latency (GPU, CPU).
Writes only into a new timestamped run dir."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import DATA, load_split, metrics, pct
from nca_text.model import TextNCA, FlatPolicy
from obs_pack import fit_or_load_tfidf, pack_arch

LABELS = json.load(open(Path(__file__).with_name("labels.json")))
VOCAB = {n: i for i, n in enumerate(LABELS)}


class GridNCA(nn.Module):
    """2D NCA over the (goal+steps) x width grid from nca_text.data.build_obs."""
    def __init__(self, obs_c, hidden_c, n_classes, steps, h, w, upd=96):
        super().__init__(); self.hc, self.steps = hidden_c, steps
        c = obs_c + hidden_c
        self.perc = nn.Conv2d(c, c * 3, 3, padding=1, groups=c)
        self.w1 = nn.Conv2d(c * 3, upd, 1); self.w2 = nn.Conv2d(upd, hidden_c * 2, 1, bias=False)
        nn.init.zeros_(self.w2.weight); self.head = nn.Linear(hidden_c * h * w, n_classes)
    def forward(self, obs):
        h = obs.new_zeros(obs.shape[0], self.hc, *obs.shape[2:])
        for _ in range(self.steps):
            d, g = self.w2(F.relu(self.w1(self.perc(torch.cat((obs, h), 1))))).chunk(2, 1)
            h = h + torch.sigmoid(g) * d
        return self.head(h.flatten(1))


class GraphNCA(nn.Module):
    """Message-passing NCA: cells = goal + steps, complete graph, mean aggregation."""
    def __init__(self, obs_c, hidden_c, n_classes, steps, n_nodes, upd=96):
        super().__init__(); self.hc, self.steps = hidden_c, steps
        c = obs_c + hidden_c
        self.w1 = nn.Linear(c * 2, upd); self.w2 = nn.Linear(upd, hidden_c * 2, bias=False)
        nn.init.zeros_(self.w2.weight); self.head = nn.Linear(hidden_c * n_nodes, n_classes)
    def forward(self, obs):  # obs (B, C, L)
        x = obs.transpose(1, 2); h = x.new_zeros(x.shape[0], x.shape[1], self.hc)
        mask = (x[..., -1:] > 0).float()  # presence channel
        for _ in range(self.steps):
            z = torch.cat((x, h), -1)
            msg = (z * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
            d, g = self.w2(F.relu(self.w1(torch.cat((z, msg.expand_as(z)), -1)))).chunk(2, -1)
            h = h + torch.sigmoid(g) * d * mask
        return self.head(h.flatten(1))


class SeqBaseline(nn.Module):
    """Tiny GRU (1x128) or transformer (2 layers, d=128) over the packed cells (Track B2)."""
    def __init__(self, obs_c, n_classes, kind, d=128):
        super().__init__(); self.kind = kind; self.inp = nn.Linear(obs_c, d)
        if kind == "gru": self.enc = nn.GRU(d, d, batch_first=True)
        else:
            self.pos = nn.Parameter(torch.zeros(1, 16, d))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 256, 0.0, batch_first=True), 2)
        self.head = nn.Linear(d, n_classes)
    def forward(self, obs):
        x = self.inp(obs.transpose(1, 2))
        if self.kind == "gru": return self.head(self.enc(x)[0][:, -1])
        return self.head(self.enc(x + self.pos[:, : x.shape[1]]).mean(1))


def pack(arch, ex, text_feat="hash48", goal_tfidf=None):
    return pack_arch(arch, ex, VOCAB, text_feat, goal_tfidf)


def build(arch, obs, hidden, steps):
    n = len(LABELS)
    if arch == "flat": return FlatPolicy(int(np.prod(obs.shape[1:])), n)
    if arch == "nca1d": return TextNCA(obs.shape[1], hidden, n, steps, obs.shape[2])
    if arch == "nca2d": return GridNCA(obs.shape[1], hidden, n, steps, obs.shape[2], obs.shape[3])
    if arch == "graph": return GraphNCA(obs.shape[1], hidden, n, steps, obs.shape[2])
    if arch in ("gru", "tx"): return SeqBaseline(obs.shape[1], n, arch)
    raise SystemExit(arch)


@torch.no_grad()
def evaluate(model, obs, y, dev, bs=2048):
    model.eval(); ranked = []
    for i in range(0, len(y), bs):
        ranked += torch.argsort(model(obs[i:i + bs].to(dev)), 1, descending=True).cpu().tolist()
    model.train(); return metrics(y.tolist(), ranked, len(LABELS))


@torch.no_grad()
def latency(model, obs, dev, n=500):
    model = model.to(dev).eval(); xs = []
    for i in range(n + 50):
        x = obs[i % len(obs)].unsqueeze(0)
        t = time.perf_counter(); x = x.to(dev); model(x).argmax().item(); ms = (time.perf_counter() - t) * 1000
        if i >= 50: xs.append(ms)
    return {"p50_ms": pct(xs, 50), "p95_ms": pct(xs, 95)}


def teacher_soft(path, n_train, temp):
    """Teacher letter log-probs (predictions only, no hidden states) -> soft targets."""
    soft = np.full((n_train, len(LABELS)), np.nan, np.float32)
    for line in open(path):
        r = json.loads(line); lp = np.full(len(LABELS), -30.0, np.float32)
        for k, v in r["logprobs"].items(): lp[ord(k) - 65] = v
        e = np.exp((lp - lp.max()) / temp); soft[r["row"]] = e / e.sum()
    return soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="curated"); ap.add_argument("--arch", default="nca1d")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--nca-steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--const-lr", action="store_true")
    ap.add_argument("--distill", default=""); ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--tag", default="")
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--patience", type=int, default=0, help="stop if train>=99%% and no val gain for N steps")
    ap.add_argument("--text_feat", default="hash48", choices=["hash48", "tfidf_svd128", "hash_wide"],
                    help="goal cell encoding; hash48 = legacy nca_text (default)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"{time.strftime('%Y%m%dT%H%M%S')}_{a.tag or a.arch}_{a.dataset}_s{a.seed}"
    out = DATA / "stage0_runs" / name; out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    goal_tfidf = fit_or_load_tfidf(load_split(a.dataset, "train"), a.seed) if a.text_feat == "tfidf_svd128" else None
    sp = {s: pack(a.arch, load_split(a.dataset, s), a.text_feat, goal_tfidf) for s in ("train", "val", "test")}
    Xtr, ytr = map(torch.from_numpy, sp["train"])
    soft = teacher_soft(a.distill, len(ytr), a.temp) if a.distill else None
    idx_pool = np.arange(len(ytr))
    if soft is not None and a.alpha >= 1.0: idx_pool = idx_pool[~np.isnan(soft[:, 0])]
    if a.train_frac < 1.0: idx_pool = rng.choice(idx_pool, int(len(idx_pool) * a.train_frac), replace=False)
    Xv, yv = map(torch.from_numpy, sp["val"]); Xt, yt = map(torch.from_numpy, sp["test"])
    model = build(a.arch, Xtr, a.hidden, a.nca_steps).to(dev)
    counts = np.bincount(ytr[idx_pool].numpy(), minlength=len(LABELS)).astype(np.float32)
    cw = torch.tensor(1 / np.sqrt(counts + 1), device=dev); cw = cw / cw.mean()
    if a.no_class_weight: cw = torch.ones_like(cw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    soft_t = torch.from_numpy(np.nan_to_num(soft, nan=-1.0)) if soft is not None else None
    Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)
    best, log = {"val_top1": -1}, open(out / "curve.jsonl", "w")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr if a.const_lr else a.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps)))
        for g in opt.param_groups: g["lr"] = lr
        b = torch.from_numpy(rng.choice(idx_pool, a.batch)).to(dev)
        logits = model(Xtr_d[b]); loss = F.cross_entropy(logits, ytr_d[b], weight=cw)
        if soft_t is not None:
            st = soft_t[b.cpu()].to(dev); ok = st[:, 0] >= 0
            if ok.any():
                kd = F.kl_div(F.log_softmax(logits[ok] / a.temp, 1), st[ok], reduction="batchmean") * a.temp ** 2
                loss = (1 - a.alpha) * loss + a.alpha * kd
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            sub = torch.from_numpy(rng.choice(idx_pool, min(20000, len(idx_pool)), replace=False))
            row = {"step": step, "loss": round(loss.item(), 4), "lr": lr, "wall_s": round(time.time() - t0, 1),
                   "train": evaluate(model, Xtr[sub], ytr[sub], dev), "val": evaluate(model, Xv, yv, dev),
                   "test": evaluate(model, Xt, yt, dev)}  # test logged for curves only, never used for selection
            log.write(json.dumps(row) + "\n"); log.flush()
            if not math.isfinite(row["loss"]):
                (out / "STOPPED").write_text(f"nan loss at {step}"); break
            if row["val"]["top1"] > best["val_top1"]:
                best = {"val_top1": row["val"]["top1"], "step": step, "test": row["test"]}
                torch.save({"state": model.state_dict(), "args": vars(a), "step": step}, out / "best.pt")
            if a.patience and row["train"]["top1"] >= 99 and step - best["step"] >= a.patience:
                (out / "STOPPED").write_text(f"early stop at {step}: train>=99, no val gain since {best['step']}"); break
    torch.save({"state": model.state_dict(), "args": vars(a), "step": step}, out / "last.pt")
    ck = torch.load(out / "best.pt", map_location="cpu"); model.load_state_dict(ck["state"])
    res = {"run": name, "arch": a.arch, "dataset": a.dataset, "seed": a.seed, "params": sum(p.numel() for p in model.parameters()),
           "best_step": best["step"], "val_top1": best["val_top1"], "test": evaluate(model, Xt, yt, dev),
           "split_rows": {s: int(len(v[1])) for s, v in sp.items()},
           "latency_gpu": latency(model, Xt, dev) if dev.type == "cuda" else None}
    torch.set_num_threads(1); res["latency_cpu_1thread"] = latency(model.cpu(), Xt, torch.device("cpu"))
    (out / "result.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res))


if __name__ == "__main__":
    main()
