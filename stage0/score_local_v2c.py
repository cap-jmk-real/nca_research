"""Sol (Azure) teacher on public curated data. Usage: sol_teacher.py SPLIT CAP OUTDIR [--effort low]
Output per row: ranked letters with probabilities (no logprobs on this model)."""
import json, random, re, sys, os, time
from collections import Counter
D = "/home/s4il0r/nca_data/stage0_splits/curated"
LABELS = json.load(open("/home/s4il0r/nca_research/stage0/labels.json")); LET = "ABCDEFGHIJKLM"[:len(LABELS)]
SEED = 20261026
MAXOUT = int(os.environ.get("SOL_MAXOUT", "120")); PACE = float(os.environ.get("SOL_PACE", "62"))
def load(split):
    out = []
    for l in open(f"{D}/{split}.jsonl"):
        r = json.loads(l)
        if r["target_tool"] in LABELS: out.append(r)
    return out
def body(r):
    st = r.get("recent_steps", [])
    seq = " > ".join(x["tool"] for x in st[-10:]) or "(none)"
    last = "\n".join(f"- {x['tool']}: {x['result'][:100]}" for x in st[-1:]) or "(none)"
    return f"Goal: {r['goal'][:160]}\nTools so far: {seq}\nLast results:\n{last}"


def body_parity(r):
    """HistGB-matched caps (500-char goal, 200-char last result). Code only — do not run without budget."""
    st = r.get("recent_steps", [])
    seq = " > ".join(x["tool"] for x in st[-10:]) or "(none)"
    last = "\n".join(f"- {x['tool']}: {x['result'][:200]}" for x in st[-1:]) or "(none)"
    return f"Goal: {r['goal'][:500]}\nTools so far: {seq}\nLast results:\n{last}"
def header(tr):
    c = Counter(r["target_tool"] for r in tr); n = len(tr); rng = random.Random(SEED + 7)
    per = {l: [r for r in tr if r["target_tool"] == l] for l in LABELS}
    shots = [rng.choice(per[l]) for l in LABELS if per[l]]  # >=1 example per label present in train
    missing = [l for l in LABELS if not per[l]]
    rng.shuffle(shots)
    opts = "\n".join(f"{a}. {b} (prior {100*c[b]/n:.1f}%)" for a, b in zip(LET, LABELS))
    ex = "\n".join(f"{' > '.join(x['tool'] for x in r.get('recent_steps', [])[-6:]) or '(start)'} => {LET[LABELS.index(r['target_tool'])]}" for r in shots)
    return ("Predict the next tool an AI coding agent calls. Options with their base rate in this data "
            "(respond = reply to user without a tool):\n" + opts +
            "\nBase rates matter: 'terminal' is right about half the time.\n\nExamples:\n\n" + ex +
            "\n\nReply ONLY with your top 3 options and probabilities, format: K=0.70 F=0.20 M=0.10"), missing
PAT = re.compile(r"\b([A-M])\s*[=:]\s*([0-9.]+)")
def parse(txt):
    p = {}
    for a, v in PAT.findall(txt or ""):
        if a in LET and a not in p:
            try: p[a] = float(v)
            except ValueError: pass
    return p
def metrics(y, ranked, k):
    n = len(y); t1 = sum(r[0] == a for a, r in zip(y, ranked)) / n; t3 = sum(a in r[:3] for a, r in zip(y, ranked)) / n; f = []
    for c in range(k):
        tp = sum(r[0] == c and a == c for a, r in zip(y, ranked)); fp = sum(r[0] == c and a != c for a, r in zip(y, ranked)); fn = sum(r[0] != c and a == c for a, r in zip(y, ranked))
        if tp + fn == 0: continue
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn); f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return {"n": n, "top1": round(100*t1, 2), "top3": round(100*t3, 2), "macro_f1": round(100*sum(f)/len(f), 2)}
def pct(xs, q): xs = sorted(xs); return round(xs[min(len(xs)-1, int(round(q/100*(len(xs)-1))))], 1)

import urllib.request
URL="http://127.0.0.1:11434/api/generate"
def call(model,p):
    b={"model":model,"prompt":p,"stream":False,"think":False,"logprobs":True,"top_logprobs":20,"keep_alive":"10m","options":{"num_predict":1,"temperature":0,"seed":0,"num_ctx":4096}}
    rq=urllib.request.Request(URL,json.dumps(b).encode(),{"Content-Type":"application/json"})
    t=time.perf_counter(); o=json.load(urllib.request.urlopen(rq,timeout=600)); return o,(time.perf_counter()-t)*1000
def call_d1(model,state):
    q={"next":{"type":"choice","instructions":"Which tool does the agent call next?","criteria":{l:(("reply to user without a tool" if l=="respond" else None)) for l in LABELS}}}
    b={"model":model,"state":state,"questions":q,"keep_alive":"10m"}
    rq=urllib.request.Request("http://127.0.0.1:11434/v1/systemone",json.dumps(b).encode(),{"Content-Type":"application/json"})
    t=time.perf_counter(); o=json.load(urllib.request.urlopen(rq,timeout=600)); return o,(time.perf_counter()-t)*1000
def main():
    model,split,cap,outdir=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
    tr=load("train"); head,_=header(tr)
    head=head.split("\n\nReply ONLY")[0]
    rows=load(split); idx=list(range(len(rows)))
    if cap and len(idx)>cap: idx=random.Random(SEED).sample(idx,cap)
    os.makedirs(outdir,exist_ok=True); tag=model.replace(":","_").replace("/","_")
    D1=model.startswith("d1")
    if not D1: call(model,"warmup")
    else: call_d1(model,"warmup")
    y=[];rk=[];lat=[];miss=0
    with open(f"{outdir}/preds_{tag}_{split}.jsonl","w") as fh:
        for i,j in enumerate(idx):
            r=rows[j]
            if D1:
                o,ms=call_d1(model,head+"\n\nNow predict:\n"+body(r)); lat.append(ms)
                pr=o["answers"]["next"]["probabilities"]; sc={LET[LABELS.index(k)]:v for k,v in pr.items() if k in LABELS}
            else:
              o,ms=call(model,head+"\n\nNow predict:\n"+body(r)+"\nAnswer with the single letter of the next tool.\nAnswer:"); lat.append(ms)
              sc={}
              for tl in (o.get("logprobs") or [{}])[0].get("top_logprobs",[]):
                  k=tl["token"].strip().rstrip(".")
                  if k in LET and k not in sc: sc[k]=tl["logprob"]
            if not sc: miss+=1
            od=sorted(range(len(LABELS)),key=lambda c:-sc.get(LET[c],-1e9))
            y.append(LABELS.index(r["target_tool"])); rk.append(od)
            fh.write(json.dumps({"i":i,"row":j,"gold":r["target_tool"],"pred":LABELS[od[0]],"logprobs":sc,"ms":round(ms,1)})+"\n")
            if i%50==0: print(i,len(idx),flush=True)
    res={"model":model,"prompt":"sol_v2c header (priors+1-shot/label) + letter logprob","split":split,**metrics(y,rk,len(LABELS)),"no_letter_rows":miss,"latency_ms_p50":pct(lat,50),"latency_ms_p95":pct(lat,95)}
    open(f"{outdir}/teacher_{tag}_{split}.json","w").write(json.dumps(res,indent=1)); print(json.dumps(res))
if __name__=="__main__": main()
