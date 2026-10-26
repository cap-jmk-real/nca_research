import json,sys,numpy as np
exec(open("paired_calib.py").read().split("res = {")[0])
out={}
h=("hermes_ood","test"); y=Y[h]; term=(y==T).astype(float)
cal=json.load(open(DATA/"stage0_results/paired_calib_20261026/summary.json"))["calib"]
for t,runs in L.items():
    if t.startswith("H1b") or t not in cal: continue
    tau=cal[t]["tau_val"]; v=np.mean([acc(r[h]+tau*logp,y) for r in runs],0)
    out[t]={"adj_minus_floor_pp":round(100*(v-term).mean(),2),"ci":boot(v-term,True)}
print(json.dumps(out)); json.dump(out,open(DATA/"stage0_results/paired_calib_20261026/adj_vs_floor.json","w"),indent=1)
