import sys,json,numpy as np,torch
R="/home/s4il0r/nca_research_git";sys.path[:0]=[R,R+"/stage0"]
from common import load_split,metrics
from features import goal_text, struct_feats
from obs_pack import fit_or_load_tfidf, pack_arch
from nca_text.model import GraphNCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
OUT=R.replace("nca_research_git","nca_e_logs/per_example_eval_1103")
L=json.load(open(R+"/stage0/labels.json"));L2I={t:i for i,t in enumerate(L)};N=len(L);EX=L2I["execute_code"]
sp={s:load_split("bfcl_e1",s) for s in("train","val","test")}
rows=[];summ=[]
def rec(model,seed,split,y,P):
    P=np.asarray(P);rk=np.argsort(-P,1)
    for i in range(len(y)):
        rows.append(dict(model=model,seed=seed,split=split,i=i,true=L[y[i]],top1=L[rk[i,0]],top3=[L[j] for j in rk[i,:3]],p_exec=round(float(P[i,EX]),4),p_true=round(float(P[i,y[i]]),4),true_rank=int(np.where(rk[i]==y[i])[0][0])+1))
    m=metrics(list(y),[list(r) for r in rk],N);mi=y!=EX
    cls=sorted(set(y)|set(rk[:,0]));
    auc=roc_auc_score((y!=EX).astype(int),1-P[:,EX]) if 0<mi.sum()<len(y) and np.ptp(P[:,EX])>0 else None
    summ.append(dict(model=model,seed=seed,split=split,**m,nonexec_top1=int((rk[:,0]!=EX).sum()),minority_correct=int((rk[mi,0]==y[mi]).sum()),minority_n=int(mi.sum()),minority_true_ranks=[int(np.where(rk[i]==y[i])[0][0])+1 for i in np.where(mi)[0]],auroc_nonexec=None if auc is None else round(auc,3),f1_classes=[L[c] for c in sorted(set(y))]))
y={s:np.array([L2I[e.label] for e in sp[s]]) for s in sp}
for seed in (0,1,2):
    torch.manual_seed(seed)
    tf=fit_or_load_tfidf(sp["train"],seed)
    pk={s:pack_arch("nca1d",sp[s],L2I,"tfidf_svd128",tf) for s in sp}
    for topo in ("tree","meta","star","1d"):
        m=GraphNCA(obs_c=pk["train"][0].shape[1],hidden_c=16,n_classes=N,steps=4,seq_len=5,topology=topo)
        b=torch.load(f"{R}/stage0/stage0_results/e1_checkpoints/seed{seed}_{topo}.pt",map_location="cpu",weights_only=False)
        m.load_state_dict(b.get("best_state") or b["model"]);m.eval()
        for s in("val","test"):
            with torch.no_grad(): P=torch.softmax(m(torch.from_numpy(pk[s][0])),-1).numpy()
            rec(f"NCA_{topo}",seed,s,pk[s][1],P)
    X={s:np.hstack([tf.svd.transform(tf.tfidf.transform([goal_text(e,True) for e in sp[s]])),np.stack([struct_feats(e) for e in sp[s]])]) for s in sp}
    for nm,c in(("HGB",HistGradientBoostingClassifier(learning_rate=0.1,max_iter=200,max_leaf_nodes=31,random_state=seed)),("RF",RandomForestClassifier(n_estimators=50,max_depth=15,random_state=seed,n_jobs=2))):
        c.fit(X["train"],y["train"])
        for s in("val","test"):
            P=np.zeros((len(y[s]),N));P[:,c.classes_]=c.predict_proba(X[s]);rec(nm,seed,s,y[s],P)
pri=np.bincount(y["train"],minlength=N)/len(y["train"])
for s in("val","test"): rec("majority_prior",0,s,y[s],np.tile(pri+1e-9*np.arange(N)[::-1],(len(y[s]),1)))
json.dump(rows,open(OUT+"/per_example.json","w"),indent=1);json.dump(summ,open(OUT+"/summary.json","w"),indent=1)
for r in summ:
    if r["split"]=="test": print(r["model"],r["seed"],r["top1"],r["top3"],r["macro_f1"],"nonexecTop1",r["nonexec_top1"],"minor",r["minority_correct"],"/",r["minority_n"],"ranks",r["minority_true_ranks"],"AUC",r["auroc_nonexec"],r["f1_classes"])
for r in rows:
    if r["split"]=="test" and r["true"]!="execute_code" and r["model"].startswith("NCA_tree") and r["seed"]==0: print(r)
