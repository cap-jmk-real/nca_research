"""Score an Ollama teacher by first-token log-prob over lettered label options.
Usage: score_teachers.py MODEL DATASET OUTDIR [MAX_ROWS] [SPLIT]"""
from __future__ import annotations
import json, random, string, sys, time, urllib.request
from common import load_split, metrics, pct, SPLIT_SEED
from pathlib import Path

URL = "http://127.0.0.1:11434/api/generate"
LABELS = json.load(open(Path(__file__).with_name("labels.json")))
LET = string.ascii_uppercase[: len(LABELS)]

def prompt(ex):
    hist = "\n".join(f"- {t}: {r[:300]}" for t, r in ex.steps) or "(none yet)"
    opts = "\n".join(f"{l}. {n}" for l, n in zip(LET, LABELS))
    return (f"You are routing an AI coding agent. Predict which tool the agent calls next.\n"
            f"User goal:\n{ex.goal[:500]}\n\nRecent steps (tool: result):\n{hist}\n\n"
            f"Options (respond = reply to the user without a tool):\n{opts}\n\n"
            f"Answer with the single letter of the next tool.")

def call(model, p):
    body = {"model": model, "prompt": p, "stream": False, "think": False, "logprobs": True,
            "top_logprobs": 20, "options": {"num_predict": 1, "temperature": 0, "seed": 0}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter(); out = json.load(urllib.request.urlopen(req, timeout=300))
    return out, (time.perf_counter() - t) * 1000

def main():
    model, ds, outdir = sys.argv[1:4]; cap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    split = sys.argv[5] if len(sys.argv) > 5 else "test"
    rows = [(j, e) for j, e in enumerate(load_split(ds, split)) if e.label in LABELS]
    if cap and len(rows) > cap:
        rows = random.Random(SPLIT_SEED).sample(rows, cap)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    call(model, "warmup")
    y, ranked, lat, miss = [], [], [], 0
    with open(out / f"preds_{model.replace(':','_')}_{ds}_{split}.jsonl", "w") as fh:
        for i, (row_idx, ex) in enumerate(rows):
            r, ms = call(model, prompt(ex)); lat.append(ms)
            scores = {}
            for tl in (r.get("logprobs") or [{}])[0].get("top_logprobs", []):
                k = tl["token"].strip().rstrip(".")
                if k in LET and k not in scores: scores[k] = tl["logprob"]
            if not scores: miss += 1
            order = sorted(range(len(LABELS)), key=lambda j: -scores.get(LET[j], -1e9))
            y.append(LABELS.index(ex.label)); ranked.append(order)
            fh.write(json.dumps({"i": i, "session_id": ex.session_id, "row": row_idx, "gold": ex.label, "pred": LABELS[order[0]], "logprobs": scores, "ms": round(ms, 2)}) + "\n")
            if i % 200 == 0: print(model, ds, i, len(rows), flush=True)
    res = {"model": model, "dataset": ds, "split": split, **metrics(y, ranked, len(LABELS)), "no_letter_rows": miss,
           "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95),
           "method": "first-token logprob over lettered options, top_logprobs=20, think=false, batch 1, warm model"}
    (out / f"teacher_{model.replace(':','_')}_{ds}_{split}.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res))

if __name__ == "__main__":
    main()
