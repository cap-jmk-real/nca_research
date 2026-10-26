"""v2 teacher prompt: label priors + few-shot (train only) + recent tool-call sequence.
Usage: score_teachers_v2.py MODEL DATASET OUTDIR [MAX_ROWS] [SPLIT] [K_SHOT]"""
from __future__ import annotations
import json, random, sys
from collections import Counter
from pathlib import Path
import score_teachers as st
from common import load_split, metrics, pct, SPLIT_SEED
LABELS, LET = st.LABELS, st.LET

def build_header(ds, k):
    tr = [e for e in load_split(ds, "train") if e.label in LABELS]
    c = Counter(e.label for e in tr); n = len(tr)
    pri = "\n".join(f"{LET[LABELS.index(l)]}. {l}: {100*c[l]/n:.1f}%" for l in LABELS)
    rng = random.Random(SPLIT_SEED + 7); shots = []
    per = {l: [e for e in tr if e.label == l] for l in LABELS}
    order = [l for l, _ in c.most_common()]
    for l in (order * 3)[:k]:  # cover frequent labels, then repeat by frequency
        if per[l]: shots.append(rng.choice(per[l]))
    ex = "\n\n".join(f"Example {i+1}:\n{body(e)}\nAnswer: {LET[LABELS.index(e.label)]}" for i, e in enumerate(shots))
    return (f"You are routing an AI coding agent. Predict which tool the agent calls next.\n"
            f"Options (respond = reply to the user without a tool):\n" + "\n".join(f"{a}. {b}" for a, b in zip(LET, LABELS)) +
            f"\n\nHow often each option is the correct next tool in this data (label prior):\n{pri}\n"
            f"If unsure, prefer the frequent options; 'terminal' is the most common.\n\n{ex}\n\n")

def body(e):
    seq = " -> ".join(t for t, _ in e.steps[-10:]) or "(none)"
    hist = "\n".join(f"- {t}: {r[:200]}" for t, r in e.steps[-4:]) or "(none yet)"
    return f"Goal: {e.goal[:300]}\nTool-call sequence so far: {seq}\nLast results:\n{hist}"


def body_parity(e):
    """HistGB-matched caps for Azure/Sol parity experiments (do not run without budget)."""
    seq = " -> ".join(t for t, _ in e.steps[-10:]) or "(none)"
    hist = "\n".join(f"- {t}: {r[:200]}" for t, r in e.steps[-4:]) or "(none yet)"
    return f"Goal: {e.goal[:500]}\nTool-call sequence so far: {seq}\nLast results:\n{hist}"

def main():
    model, ds, outdir = sys.argv[1:4]; cap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    split = sys.argv[5] if len(sys.argv) > 5 else "val"; k = int(sys.argv[6]) if len(sys.argv) > 6 else 13
    head = build_header(ds, k)
    rows = [(j, e) for j, e in enumerate(load_split(ds, split)) if e.label in LABELS]
    if cap and len(rows) > cap: rows = random.Random(SPLIT_SEED).sample(rows, cap)
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True); tag = model.replace(':', '_')
    st.call(model, "warmup"); y, ranked, lat, miss = [], [], [], 0
    with open(out / f"preds_v2_{tag}_{ds}_{split}.jsonl", "w") as fh:
        for i, (ri, e) in enumerate(rows):
            r, ms = st.call(model, head + f"Now predict:\n{body(e)}\nAnswer with the single letter of the next tool.\nAnswer:"); lat.append(ms)
            sc = {}
            for tl in (r.get("logprobs") or [{}])[0].get("top_logprobs", []):
                kk = tl["token"].strip().rstrip(".")
                if kk in LET and kk not in sc: sc[kk] = tl["logprob"]
            if not sc: miss += 1
            o = sorted(range(len(LABELS)), key=lambda j: -sc.get(LET[j], -1e9))
            y.append(LABELS.index(e.label)); ranked.append(o)
            fh.write(json.dumps({"i": i, "row": ri, "gold": e.label, "pred": LABELS[o[0]], "logprobs": sc, "ms": round(ms, 2)}) + "\n")
            if i % 50 == 0: print(i, len(rows), flush=True)
    res = {"model": model, "prompt": "v2 priors+%d-shot(train)+tool seq" % k, "dataset": ds, "split": split,
           **metrics(y, ranked, len(LABELS)), "no_letter_rows": miss, "latency_ms_p50": pct(lat, 50), "latency_ms_p95": pct(lat, 95)}
    (out / f"teacher_v2_{tag}_{ds}_{split}.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res))

if __name__ == "__main__": main()
