#!/usr/bin/env python3
"""Build paper/REGRESSION_AUDIT_RIGOUR_1110.md (rigour/novelty pass): every past review complaint -> status in the current paper/main.tex.

aiXiv points (349, extracted from 63 aiXiv reviews v1.4-v7.6 into aixiv_points.json) are mapped to a fix
location by theme plus keyword rules (first matching rule wins). Each rule's anchor is a literal string that
must occur in main.tex; line numbers and sections are looked up here, so they track the current text.
Our own reviews, klaut runs and ledgers are listed by hand below.
"""
import json, re, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "paper/main.tex").read_text().splitlines()
POINTS = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "paper/audit/aixiv_points.json").read_text())["points"]

SECS = []
for i, l in enumerate(TEX, 1):
    m = re.match(r"\\(section|subsection)\*?\{([^}]*)\}", l)
    if m: SECS.append((i, m.group(2)))
APP = next(i for i, l in enumerate(TEX, 1) if l.startswith(r"\appendix"))
def loc(anchor):
    hits = [i for i, l in enumerate(TEX, 1) if anchor in l]
    if not hits: raise SystemExit(f"anchor not found: {anchor!r}")
    i = hits[0]
    sec = ([s for s in SECS if s[0] <= i] or [(0, "Abstract")])[-1][1].replace("\\&", "&")
    return f"{'App. ' if i > APP else '§'}{sec}, l.{i}"

# anchors (literal substrings of main.tex)
A = dict(
    scope=r"\label{tab:theorem-scope}", jac=r"\paragraph{Empirical check of the contraction hypothesis.}", appb=r"\section{Perturbation Bound}", lean=r"\section{Lean",
    leak=r"\label{prop:leakage}", tree=r"\label{thm:tree-nca-duality}", rw=r"\section{Related Work}", novelty=r"\textbf{Novelty.}",
    modest=r"The headline is modest", hpo=r"\label{tab:hpo-paired}", hpoapp=r"\section{Hyperparameter Search}",
    budget=r"\textbf{Search budget.}", feats=r"\textbf{Primary tree baselines see fewer features.}",
    notrun=r"\textbf{Baselines not run.}", seeds=r"\textbf{Small test sets.}", stats=r"minimum detectable primary difference",
    holm=r"$p$-values are Holm-adjusted", esc=r"\paragraph{Escalation.} For the tuned tree GNCA", escproto=r"thresholds $\theta\in", esclim=r"\textbf{Escalation.} The weight",
    cpu=r"\paragraph{Inference latency, default configurations.}", lat=r"\paragraph{Latency.}", latlim=r"Latency was measured on one machine",
    bench=r"\label{tab:multibench}", ood=r"\paragraph{Out of distribution.}", oodlim=r"\textbf{Out-of-distribution test on one user.}", splits=r"\paragraph{Data.}",
    turn=r"\textbf{Turn-level evaluation and an offline proxy.}", ses=r"\paragraph{Session-level routing.}",
    main=r"\label{tab:rig-main}", abl=r"\paragraph{Ablations.} Changing", e2e=r"\paragraph{Featurisation cost.}", inf=r"\paragraph{Inference latency.} On one CPU", newbase=r"\paragraph{New baselines.} Three",
    multiseed=r"\paragraph{Main protocol: multi-seed selection", alg=r"\label{alg:router}", prop=r"\label{prop:structure}", delta=r"\label{tab:delta}",
    proposal=r"A protocol for a hosted-LLM baseline", gpu=r"\textbf{Reproducibility on GPU.}", notation=r"$Q = \operatorname{diag}(S\mathbf{1})$", update=r"\subsection{Cellular update rule}",
    macros=r"Every number in", repo=r"github.com", vocab=r"\textbf{Fixed vocabulary and window.}", formal=r"\textbf{Formal results are idealised.}",
    ledger=r"\label{tab:ledger}", sources=r"\label{tab:sources}", params=r"\paragraph{Parameter count.}", rare=r"\paragraph{Rare tools.}",
    llm=r"Qwen3.5-9B", impact=r"\paragraph{Broader impact.}", where=r"\paragraph{Where the GNCA fits.}", env=r"PyTorch~\numEnvTorch",
    factorial=r"\paragraph{Hermes factorial.}", fig1=r"\label{fig:topologies}", contractive=r"contractive variant", sens=r"\paragraph{Protocol sensitivity.}",
)
def L(*keys): return "; ".join(loc(A[k]) for k in keys)

# rules: (theme or None, regex or None, status, anchors, note, regression_note)
R = [
 (None, r"environment|software version|library version", "addressed", ("env",), "Versions in Training paragraph + environment.json", "Partly missing before this pass"),
 (None, r"history turns|window|node features|graph size|\(V=5\)", "partly", ("vocab",), "Fixed 5-cell window and 13 tools stated; other windows/features not tested (needs runs)", ""),
 (None, r"out-of-distribution|\bOOD\b|distribution shift|real-world (traces|deploy)", "addressed", ("ood","oodlim"),
  "All 13 tuned models scored on 1,208 turns from 51 held-out private sessions (S8, session bootstrap, Holm); one-user scope stated", "REGRESSED: Oct draft had the Hermes OOD table; v7.6/v1110 text had dropped it. Restored as limitation (S7) in this pass"),
 ("T10,T16", r"generali[sz]ation", "addressed", ("ood","oodlim","vocab"),
  "All 13 tuned models scored on held-out private sessions (S8); other shifts (new tools, harnesses) stated as untested", "REGRESSED: Oct draft had the Hermes OOD table; v7.6/v1110 text had dropped it. Restored as limitation (S7) in this pass"),
 (None, r"downstream|closed-loop|agent(ic)? (evaluation|pilot|loop|result)|pilot|task success|end-to-end agent", "partly", ("turn","ses"),
  "Session-level offline proxy added (S8); agents not executed, task success not measured (stated)", ""),
 (None, r"\bseeds?\b", "addressed", ("multiseed","main"), "Five test seeds for all 13 models; selection on 3-seed mean validation (S8)", ""),
 (None, r"power|underpowered|detectable", "addressed", ("stats","seeds"), "MDE stated for the 5-seed main comparison (0.6-4.4 pp)", "REGRESSED: v7.6 'power 0.94' had no source and was removed in v1110 without replacement; MDE added in this pass"),
 (None, r"feature extraction|graph construction|full pipeline|featuri", "addressed", ("inf","e2e","lat"), "Inference latency of all tuned models (S8); featurisation timed and reported separately as preprocessing", ""),
 (None, r"environment|software version|library version", "addressed", ("env",), "Versions in Training paragraph + environment.json", "Partly missing before this pass"),
 ("!T08,T06", r"ablat", "addressed", ("abl","factorial"), "One-component ablations (normalisation, (A-I)x block, gate, tying) at the tuned config, 5 seeds, Holm; plus factorial", ""),
 (None, r"GraphSAGE|\bGAT\b|\bGCN\b|LSTM|\bGRU\b|other NCA|fine-tun", "partly", ("newbase","notrun"),
  "Transformer, untied GNN and kNN now run (S8); GAT/GCN/LSTM/GRU/other NCA and fine-tuned function-calling models not run (stated)", ""),
 (None, r"(baseline|compar|versus|vs\.?).*(transformer|\bGNN\b|GraphSAGE|\bGAT\b|\bGCN\b|LSTM|\bGRU\b|sequence model|other NCA|router|retrieval)|(transformer|\bGNN\b|LSTM|\bGRU\b|sequence model|retrieval).*(baseline|compar)|stronger (neural )?baseline|modern router", "addressed", ("newbase","main"),
  "Transformer, untied GNN and kNN retrieval tuned with the same budget, 5 seeds, Holm family (S8)", "Earlier Transformer 65.87% run (old protocol, no result file) not citable; replaced by S8 run"),
 (None, r"XXXNEVER|GraphSAGE|\bGAT\b|\bGCN\b|LSTM|\bGRU\b|sequence model|other NCA|stronger (neural )?baseline|modern router|retrieval", "partly", ("notrun","rw"),
  "Not run under this protocol; stated as limitation; related work extended", "Earlier Transformer 65.87% run (old protocol, no result file) not citable"),
 (None, r"\bLLM\b|GPT|language model baseline|zero-shot", "partly", ("llm","notrun","proposal"), "Two local zero-shot LLMs for cost only; hosted-LLM protocol written, not run (paid API needs approval)", ""),
 (None, r"cost model|λ|lambda|break-even|Pareto", "addressed", ("modest","cpu"), "Cost model and Pareto claim removed; plain latency/size reported", ""),
 (None, r"macro-F1|rare(-| )tool", "addressed", ("rare",), "Rare-tool macro-F1 reported", ""),
 (None, r"dynamic tool|failure mode|false negative|open-vocabulary", "addressed", ("vocab","esclim"), "Fixed vocabulary + escalation scope limitations", ""),
 (None, r"Lean|axiom|sorry|mechani[sz]ed", "addressed", ("lean","scope"), "Lean audit appendix, #print axioms", ""),
 (None, r"compression|18\.2M|parameter count|parameter-matched", "addressed", ("params","cpu"), "Counts from code; RF nodes from code; compression ratio removed", ""),
 (None, r"incremental|limited novelty|lack(s|ing)? of novelty|novelty is (limited|modest)", "partly", ("novelty","rw","delta","alg","prop"),
  "Algorithm box, prior-work delta table and exact structural proposition (unit-tested) added; ablations show components do not drive accuracy; stated as limitation", ""),
 ("T01", None, "addressed", ("rw","delta","novelty"), "Overclaims removed; prior-art delta in Related Work", ""),
 ("T02", None, "addressed", ("multiseed","main","hpoapp","budget"), "Equal-budget search, multi-seed selection, 5 seeds, 13 models (S6, S8)", ""),
 ("T03", r"equivalence|TOST|non-inferior|mixed-effects|LME", "addressed", ("stats","ledger"), "Removed; differences tested only", ""),
 ("T03", None, "addressed", ("stats","holm"), "Seed-clustered paired bootstrap, Holm, DL pooling", ""),
 ("T04", None, "addressed", ("modest","where","main"), "Headline 'competitive and cheap'; default vs tuned scoped in Results, Discussion, Conclusion", "Superiority framing re-checked in this pass (default-vs-tuned qualifiers added)"),
 ("T05", None, "addressed", ("scope","jac","appb","formal"), "Theorem-scope table; measured Jacobian; App. B assumptions", ""),
 ("T06", None, "addressed", ("leak",), "Causal-operator claim removed; leakage proposition with proof", ""),
 ("T07", None, "addressed", ("tree","rw"), "Representability for a symbolic tree only; concordance claim removed", ""),
 ("T08", None, "addressed", ("escproto","esc","esclim"), "β fixed a priori, thresholds fixed, components ablated, entropy recommended", ""),
 ("T09", None, "addressed", ("inf","e2e","lat","cpu","latlim"), "Inference (model call) and featurisation timed separately for all tuned models, 7950X 1 thread; single machine stated", ""),
 ("T10", r"leak|duplicat|split", "addressed", ("splits","bench"), "Per-benchmark held-out splits", ""),
 ("T10", None, "addressed", ("bench","splits"), "Four benchmarks; data and splits described", ""),
 ("T11", None, "partly", ("turn","ses"), "Session-level offline proxy added; agent-level task success not measured", ""),
 ("T12", None, "addressed", ("update","notation"), "Rewrite; notation collisions (Δ, α, τ, t, C) removed in this pass", "Notation collisions found in v1110 text and fixed"),
 ("T13", None, "addressed", ("macros","sources"), "All numbers from generated macros", ""),
 ("T14", None, "addressed", ("repo","sources","env"), "Public repo, result files, environment", ""),
 ("T16", None, "addressed", ("modest","cpu","impact"), "Modest headline; cost framing; limitations section", ""),
 ("T99", None, "addressed", ("vocab","esclim"), "Limitations cover dynamic tools and escalation scope", ""),
]
def classify(p):
    for th, rx, st, anc, note, reg in R:
        if th and th.startswith("!") and p["theme"] in th[1:].split(","): continue
        if th and not th.startswith("!") and p["theme"] not in th.split(","): continue
        if rx and not re.search(rx, p["point"], re.I): continue
        return st, L(*anc), note, reg
    raise SystemExit(f"unmapped {p}")

rows = []
for n, p in enumerate(POINTS, 1):
    st, where, note, reg = classify(p)
    rows.append((f"X{n:03d}", p, st, where, note, reg))

# ---- non-aiXiv sources (by hand) ----
M = [
 # id, point, source, status, anchors, note
 ("P1","Escalation evaluated on K=8 factorial model, not primary","PAPER_REVIEW_V76C(_HPO).md minor","addressed",("esc",),"Escalation on the tuned primary model, all 4 benchmarks, 5 seeds (S8)"),
 ("P2","Factorial uses its own feature packing","PAPER_REVIEW_V76C.md minor","addressed",("factorial","sens"),"Stated in factorial description"),
 ("P3","Only 3 seeds","PAPER_REVIEW_1110/V76C minor","addressed",("multiseed","seeds"),"5 seeds + MDE"),
 ("P4","LLM baselines zero-shot","PAPER_REVIEW_V76C minor","partly",("notrun","proposal"),"Limitation; protocol written, not run (paid API)"),
 ("P5","Fig. 1(d) label touches arrow","PAPER_REVIEW_V76C nit","addressed",("fig1",),"Label offsets fixed in paper/generate_topology_figure.py; checked in rendered PDF"),
 ("P6","Cite on-device function-calling models","PAPER_REVIEW_V76C_HPO minor","addressed",("rw",),"Octopus v2, TinyAgent, Hammer, COLT added and verified"),
 ("P7","Table 5 placement","PAPER_REVIEW_V76C_HPO nit","addressed",("hpo",),"Float placement checked"),
 ("P8","One-seed HPO selection","PAPER_REVIEW_V76C_HPO minor","addressed",("multiseed",),"Top-5 trials re-scored with 3 seeds; selection on mean (S8)"),
 ("P9","HGB default vs search early-stopping mismatch","PAPER_REVIEW_V76C_HPO minor","addressed",("budget",),"Disclosed"),
 ("P10","Tuned-model latency not measured","PAPER_REVIEW_V76C_HPO minor","addressed",("inf","e2e"),"All tuned models timed: inference and featurisation separately (S8)"),
 ("K1","split_labels / number_conflict / Wilson FP","klaut 2026-11-03","addressed",("bench",),"Labels per split; klaut 10.0 now"),
 ("K2","Undifferentiated prior art","klaut 2026-11-03/1110","addressed",("rw",),"Delta stated per line of work"),
 ("K3","Runaway sentences / filenames in text","klaut 2026-11-03","addressed",("sources",),"Paths only in Table 8"),
 ("K4","β/threshold without calibration protocol","klaut (this pass)","addressed",("escproto","esclim"),"Limitation now points to fixed protocol"),
 ("O1","Single seed; no held-out test","draft review-1/2 (Oct)","addressed",("bench","seeds"),"3 seeds, held-out tests"),
 ("O2","Weak/unevaluated LLM baselines; DeepSeek/Llama pending","draft review-1..6","partly",("llm","notrun","proposal"),"Local LLMs only; hosted-LLM protocol written, not run"),
 ("O3","No OOD evaluation (blocker in review-3, fixed in review-5/6)","draft review-3..6","addressed",("ood","oodlim"),"Current models evaluated OOD (S8); earlier regression (S7) now superseded"),
 ("O4","Paired CIs between small models","draft review-3..6","addressed",("holm",),"Table 3"),
 ("O5","Multiplicity","draft review-4..6","addressed",("holm",),"Holm per family"),
 ("O6","Calibration","draft review-4..6","addressed",("esc","abl"),"ECE (15 bins) for GNCA, HGB and MLP, AUROC, selective accuracy (S8)"),
 ("O7","Per-label errors","draft review-2..4","addressed",("rare",),"Macro-F1 / rare tools"),
 ("O8","Private code","draft review-2","addressed",("repo",),"Public repo"),
 ("S1","M1: Transformer 65.87 / Qwen 55.20 unsupported","wt_nca_research REVIEW.md","addressed",("ledger","newbase"),"Removed (ledger); Transformer now run under the protocol (S8)"),
 ("S2","M2: Azure SOTA pilot","wt_nca_research REVIEW.md","addressed",("ledger",),"Not reported; no result file"),
 ("S3","M3: no-class-weight ablation","wt_nca_research REVIEW.md","addressed",("factorial",),"Loss factor in factorial (weighted vs plain)"),
 ("S4","M4: HF dataset revisions","wt_nca_research REVIEW.md","addressed",("bench",),"Dataset sources cited; revisions in repo manifest"),
 ("S5","M5: LICENSE","wt_nca_research REVIEW.md","addressed",("repo",),"MIT in repo"),
 ("L1","PR #8 App. B: Jacobian 0.887/0.890, 62,645 turns","PR #8 ledger","addressed",("ledger","jac"),"Ledger row; measured Jacobian"),
 ("L2","PR #8 App. B: causal lower-triangular operator","PR #8 ledger","addressed",("ledger","leak"),"Ledger row"),
 ("L3","PR #8 App. B: LME, TOST, NI, Cohen's h","PR #8 ledger","addressed",("ledger",),"Ledger row"),
 ("L4","PR #8 App. B: 50-trial Hyperband","PR #8 ledger","addressed",("ledger","hpoapp"),"Ledger row; real search done"),
 ("L5","PR #8 App. B: ±25%/±50% sensitivity analyses","PR #8 ledger","addressed",("ledger",),"REGRESSED (row missing in v1110 ledger); row restored in this pass"),
 ("L6","PR #8 App. B: pre-registered","PR #8 ledger","addressed",("ledger",),"Ledger row"),
 ("L7","PR #8 App. B: Xeon latency, RAM","PR #8 ledger","addressed",("ledger","lat"),"Ledger row"),
 ("L8","PR #8 App. B: RF 18.2M nodes","PR #8 ledger","addressed",("ledger","cpu"),"Ledger row"),
 ("L9","PR #8 App. B: 10k trajectories, grammar decoding, ARTS/dense latencies","PR #8 ledger","addressed",("ledger",),"Ledger rows"),
 ("L10","PR #8 App. B: agent-pilot statistics","PR #8 ledger","addressed",("ledger","turn"),"REGRESSED (row missing in v1110 ledger); row restored in this pass"),
 ("L11","PR #8 App. B: 68.92% from K=8","PR #8 ledger","addressed",("ledger",),"Ledger row"),
 ("L12","PR #8 App. B: wrong citations","PR #8 ledger","addressed",("ledger","rw"),"All 26 refs re-verified this pass"),
 ("L13","PR #8 App. B: zero axioms","PR #8 ledger","addressed",("ledger","lean"),"Ledger row"),
 ("L14","LEDGER_V76C_1110.md: every macro traced to S1-S8","LEDGER_V76C_1110.md","addressed",("sources",),"S8 added this pass"),
]

c = collections.Counter(r[2] for r in rows)
cm = collections.Counter(m[3] for m in M)
# regression check against the previous audit (status may only stay or improve)
OLD = {}
for line in (ROOT / "paper/REGRESSION_AUDIT_1110.md").read_text().splitlines():
    m = re.match(r"\| (\w+)[ :\[].*?\| [^|]* \| (addressed|partly|regressed) \|", line)
    if m: OLD[m.group(1)] = m.group(2)
NEW = {r[0]: r[2] for r in rows} | {m[0]: m[3] for m in M}
RANK = {"regressed": 0, "partly": 1, "addressed": 2}
worse = [k for k in NEW if k in OLD and RANK[NEW[k]] < RANK[OLD[k]]]
better = [k for k in NEW if k in OLD and RANK[NEW[k]] > RANK[OLD[k]]]
missing = [k for k in OLD if k not in NEW]
assert len(OLD) == len(NEW) == len(rows) + len(M), (len(OLD), len(NEW))
out = ["# Regression audit, rigour/novelty pass, 2026-11-10", "",
 "Every complaint from every past review of this manuscript line, with its status in the current `paper/main.tex`, compared with the previous audit (`paper/REGRESSION_AUDIT_1110.md`, PR #9).",
 "Generated by `scripts/build_regression_audit_rigour_1110.py`; section names and line numbers are looked up in the current source, and the script fails if an anchor is missing.", "",
 "Status: **addressed** (fixed in the text or by a new experiment), **partly** (stated honestly as a limitation; a full fix needs experiments not run here or cannot be made in text), **regressed** (status worse than in the previous audit).", "",
 "## Summary", "",
 f"- Items: {len(NEW)} ({len(rows)} aiXiv points from 63 reviews, v1.4-v7.6; {len(M)} own reviews, klaut, earlier drafts and ledgers).",
 f"- aiXiv points: addressed {c['addressed']}, partly {c['partly']}, regressed {len([k for k in worse if k.startswith('X')])}.",
 f"- Other items: addressed {cm['addressed']}, partly {cm['partly']}, regressed {len([k for k in worse if not k.startswith('X')])}.",
 f"- Against the previous audit: {len(better)} items improved (partly -> addressed), {len(worse)} got worse, {len(missing)} missing.",
 "", "## Still open (need experiments or approval not available in this pass)", "",
 "1. Agent-level task success (T11, A16): only an offline session proxy exists.",
 "2. Hosted / few-shot / fine-tuned LLM baselines (O2, P4): protocol in `paper/LLM_BASELINE_PROPOSAL_1110.md`, needs paid-API approval.",
 "3. GAT/GCN/LSTM/GRU/other-NCA and fine-tuned small function-calling baselines.",
 "4. Other windows, cell features and open vocabularies.",
 "5. Novelty (T01, 'incremental'): text now gives an algorithm box, a delta table and an exact proposition, but the components show no accuracy effect.", "",
 "## Own reviews, klaut, earlier drafts and ledgers", "",
 "| Point | Source review | Status | Where fixed (section + line) | Note |", "|---|---|---|---|---|"]
for m in M: out.append(f"| {m[0]}: {m[1]} | {m[2]} | {m[3]} | {L(*m[4])} | {m[5]} |")
out += ["", "## aiXiv review points (all versions)", "",
 "| Point | Source review | Status | Where fixed (section + line) | Note |", "|---|---|---|---|---|"]
for rid, p, st, where, note, rg in rows:
    txt = p["point"].replace("|", "/")
    out.append(f"| {rid} [{p['theme']}, {p['severity']}]: {txt} | aiXiv v{p['version']} (review {p['review_id']}, rating {p['rating']}) | {st} | {where} | {note}{'. ' + rg if rg else ''} |")
(ROOT / "paper/REGRESSION_AUDIT_RIGOUR_1110.md").write_text("\n".join(out) + "\n")
print(len(NEW), dict(c), dict(cm), "improved", len(better), "worse", worse, "missing", missing)
