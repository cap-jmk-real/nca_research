"""Paired bootstrap between small models (curated test: row bootstrap; Hermes OOD: session-cluster)
+ calibration check: logit adjustment tau*log(train prior) or terminal bias b, tuned on curated val only."""
import glob, json, os, sys, itertools
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, load_split
from train_eval import pack, build, LABELS
torch.set_num_threads(2); rng = np.random.default_rng(20261026); B = 10000
S = {s: load_split(*s) for s in [("curated","val"),("curated","test"),("hermes_ood","test")]}
sess = np.array([e.session_id for e in S[("hermes_ood","test")]]); us, inv = np.unique(sess, return_inverse=True)
members = [np.where(inv == i)[0] for i in range(len(us))]
prior = np.array(json.load(open(DATA/"stage0_splits/curated/summary.json"))["train"]["labels"][l]["n"] if False else 0 for l in LABELS) if False else None
lab = json.load(open(DATA/"stage0_splits/curated/summary.json"))["train"]["labels"]
prior = np.array([lab.get(l, {"n": 0})["n"] for l in LABELS], float) + 1; logp = np.log(prior / prior.sum())
T = LABELS.index("terminal"); cache = {}; L = {}
for d in sorted(glob.glob(str(DATA / "stage0_runs/*"))):
    tag = Path(d).name.split("_", 1)[1].rsplit("_curated", 1)[0]
    if tag.startswith("G1p") or not os.path.exists(d + "/result.json"): continue
    ck = torch.load(d + "/best.pt", map_location="cpu"); a = ck["args"]
    if a.get("distill"): continue
    out = {}
    for s in S:
        k = (a["arch"], s)
        if k not in cache: X, y = pack(a["arch"], S[s]); cache[k] = (torch.from_numpy(X), y)
        X, y = cache[k]
        m = build(a["arch"], X, a["hidden"], a["nca_steps"]); m.load_state_dict(ck["state"]); m.eval()
        with torch.no_grad(): out[s] = torch.cat([m(X[i:i+1024]) for i in range(0, len(X), 1024)]).numpy()
    L.setdefault(tag, []).append(out)
Y = {s: cache[("nca1d", s)][1] for s in S}
def acc(lg, y): return (lg.argmax(1) == y).astype(float)
def boot(v, cluster):
    if cluster: bs = [np.concatenate([v[members[i]] for i in rng.integers(0, len(us), len(us))]).mean() for _ in range(B)]
    else: bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [round(100*np.percentile(bs, 2.5), 2), round(100*np.percentile(bs, 97.5), 2)]
res = {"paired": {}, "calib": {}}
C = {s: {t: np.mean([acc(r[s], Y[s]) for r in runs], 0) for t, runs in L.items()} for s in S}
for s, cl in ((("curated","test"), False), (("hermes_ood","test"), True)):
    key = s[0]; res["paired"][key] = {}
    for a_, b_ in itertools.combinations(sorted(L), 2):
        d = C[s][a_] - C[s][b_]; ci = boot(d, cl)
        res["paired"][key][f"{a_} - {b_}"] = {"diff_pp": round(100*d.mean(), 2), "ci95": ci, "excl0": ci[0] > 0 or ci[1] < 0}
taus = np.round(np.arange(-1.0, 1.01, 0.1), 2); bs_ = np.round(np.arange(-3, 3.01, 0.25), 2)
for t, runs in L.items():
    vv = ("curated","val")
    def score(f, s): return np.mean([acc(f(r[s]), Y[s]).mean() for r in runs])
    ft = lambda tau: (lambda z: z + tau * logp)
    def fb(b):
        def f(z): z = z.copy(); z[:, T] += b; return z
        return f
    tau = max(taus, key=lambda x: (score(ft(x), vv), -abs(x))); bb = max(bs_, key=lambda x: (score(fb(x), vv), -abs(x)))
    row = {"tau_val": float(tau), "bias_val": float(bb)}
    for nm, f in (("raw", lambda z: z), ("logit_adj", ft(tau)), ("terminal_bias", fb(bb))):
        for s in S:
            per = [100*acc(f(r[s]), Y[s]).mean() for r in runs]
            row[f"{nm}_{s[0]}_{s[1]}"] = [round(float(np.mean(per)), 2), round(float(np.std(per, ddof=1)), 2)]
        h = ("hermes_ood","test"); d = np.mean([acc(f(r[h]), Y[h]) - acc(r[h], Y[h]) for r in runs], 0)
        row[f"{nm}_hermes_gain_ci"] = boot(d, True) if nm != "raw" else None
    res["calib"][t] = row
o = DATA/"stage0_results/paired_calib_20261026"; o.mkdir(parents=True, exist_ok=True); (o/"summary.json").write_text(json.dumps(res, indent=1, default=float))
for k, v in res["paired"].items():
    print(k); [print(" ", p, x) for p, x in v.items() if "H1_nocw" in p or "B_flat" in p]
for t, r in res["calib"].items(): print(t, r)
