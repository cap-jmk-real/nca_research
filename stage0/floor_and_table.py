"""Always-terminal floor on each split + side-by-side table from result files."""
from __future__ import annotations
import glob, json, statistics as st, sys
from common import DATA, load_split
rows = {}
for ds in sys.argv[1:] or ["curated"]:
    for sp in ("val", "test"):
        ex = load_split(ds, sp); rows[f"{ds}/{sp}"] = round(100 * sum(e.label == "terminal" for e in ex) / len(ex), 2)
out = {"floor_always_terminal_top1": rows, "runs": {}, "teachers": {}}
for f in glob.glob(str(DATA / "stage0_runs/*/result.json")):
    r = json.load(open(f)); out["runs"].setdefault(f"{r['arch']}/{r['dataset']}", []).append(r)
for f in glob.glob(str(DATA / "stage0_results/*/teacher_*.json")):
    r = json.load(open(f)); out["teachers"][f"{r['model']}/{r['dataset']}/{r.get('split','test')}"] = r
agg = {}
for k, rs in out["runs"].items():
    t1 = [r["test"]["top1"] for r in rs]
    agg[k] = {"seeds": len(rs), "test_top1_mean": round(st.mean(t1), 2), "test_top1_sd": round(st.pstdev(t1), 2),
              "test_top3_mean": round(st.mean(r["test"]["top3"] for r in rs), 2),
              "macro_f1_mean": round(st.mean(r["test"]["macro_f1"] for r in rs), 2),
              "gpu_p50_ms": st.median(r["latency_gpu"]["p50_ms"] for r in rs if r["latency_gpu"]) if any(r["latency_gpu"] for r in rs) else None,
              "cpu_p50_ms": st.median(r["latency_cpu_1thread"]["p50_ms"] for r in rs),
              "cpu_p95_ms": st.median(r["latency_cpu_1thread"]["p95_ms"] for r in rs)}
out["aggregate"] = agg
print(json.dumps({k: out[k] for k in ("floor_always_terminal_top1", "aggregate")}, indent=1))
print(json.dumps({k: {x: v[x] for x in ("n", "top1", "top3", "macro_f1", "latency_ms_p50", "latency_ms_p95", "no_letter_rows")} for k, v in out["teachers"].items()}, indent=1))
