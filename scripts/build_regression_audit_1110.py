#!/usr/bin/env python3
"""Build paper/REGRESSION_AUDIT_1110.md: every past review complaint -> status in the current paper/main.tex.

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
    sec = [s for s in SECS if s[0] <= i][-1][1].replace("\\&", "&")
    return f"{'App. ' if i > APP else '§'}{sec}, l.{i}"

# anchors (literal substrings of main.tex)
A = dict(
    scope=r"\label{tab:theorem-scope}", jac=r"\paragraph{Empirical check of the contraction hypothesis.}", appb=r"\section{Perturbation Bound}", lean=r"\section{Lean",
    leak=r"\label{prop:leakage}", tree=r"\label{thm:tree-nca-duality}", rw=r"\section{Related Work}", novelty=r"\textbf{Novelty.}",
    modest=r"The headline is modest", hpo=r"\label{tab:hpo-paired}", hpoapp=r"\section{Hyperparameter Search}",
    budget=r"\textbf{Small search budget.}", feats=r"\textbf{Primary tree baselines see fewer features.}",
    notrun=r"\textbf{Baselines not run.}", seeds=r"\textbf{Three seeds, small test sets.}", stats=r"minimum detectable primary difference",
    holm=r"Holm", esc=r"\paragraph{Escalation score.} On the Hermes", escproto=r"thresholds $\theta\in", esclim=r"\textbf{Escalation analysed on one configuration.}",
    cpu=r"\paragraph{CPU cost.}", lat=r"\paragraph{Latency.}", latlim=r"\textbf{One machine for latency,}",
    bench=r"\label{tab:multibench}", ood=r"\textbf{No out-of-distribution test of these models.}", splits=r"\paragraph{Data.}",
    turn=r"\textbf{Turn-level evaluation only.}", notation=r"$Q = \operatorname{diag}(S\mathbf{1})$", update=r"\subsection{Cellular update rule}",
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
 (None, r"out-of-distribution|\bOOD\b|distribution shift|real-world (traces|deploy)", "partly", ("ood","vocab"),
  "OOD pilot on other models reported as limitation; current models not evaluated OOD (needs new run)", "REGRESSED: Oct draft had the Hermes OOD table; v7.6/v1110 text had dropped it. Restored as limitation (S7) in this pass"),
 ("T10,T16", r"generali[sz]ation", "partly", ("ood","vocab"),
  "OOD pilot on other models reported as limitation; current models not evaluated OOD (needs new run)", "REGRESSED: Oct draft had the Hermes OOD table; v7.6/v1110 text had dropped it. Restored as limitation (S7) in this pass"),
 (None, r"downstream|closed-loop|agent(ic)? (evaluation|pilot|loop|result)|pilot|task success|end-to-end agent", "partly", ("turn",),
  "Pilots removed; turn-level only stated; agent-level evaluation needs new experiment", ""),
 (None, r"\bseeds?\b", "partly", ("seeds","budget"), "3 seeds stated; MDE added; 5+ seeds need new runs", ""),
 (None, r"power|underpowered|detectable", "addressed", ("stats","seeds"), "MDE 0.9-4.1 pp stated", "REGRESSED: v7.6 'power 0.94' had no source and was removed in v1110 without replacement; MDE added in this pass"),
 (None, r"feature extraction|graph construction|full pipeline|featuri", "partly", ("lat","latlim"), "Featurisation excluded, stated; end-to-end timing needs new measurement", ""),
 (None, r"environment|software version|library version", "addressed", ("env",), "Versions in Training paragraph + environment.json", "Partly missing before this pass"),
 ("!T08,T06", r"ablat", "partly", ("factorial","esc"), "Factorial varies depth, topology, loss, carry, halting and contractive variant; perception stack and gate are not ablated separately", ""),
 (None, r"(baseline|compar|versus|vs\.?).*(transformer|\bGNN\b|GraphSAGE|\bGAT\b|\bGCN\b|LSTM|\bGRU\b|sequence model|other NCA|router|retrieval)|(transformer|\bGNN\b|LSTM|\bGRU\b|sequence model|retrieval).*(baseline|compar)|stronger (neural )?baseline|modern router", "partly", ("notrun","rw"),
  "Not run under this protocol; stated as limitation; related work extended", "Earlier Transformer 65.87% run (old protocol, no result file) not citable"),
 (None, r"XXXNEVER|GraphSAGE|\bGAT\b|\bGCN\b|LSTM|\bGRU\b|sequence model|other NCA|stronger (neural )?baseline|modern router|retrieval", "partly", ("notrun","rw"),
  "Not run under this protocol; stated as limitation; related work extended", "Earlier Transformer 65.87% run (old protocol, no result file) not citable"),
 (None, r"\bLLM\b|GPT|language model baseline|zero-shot", "partly", ("llm","notrun"), "Two local zero-shot LLMs for cost only; limitation", ""),
 (None, r"cost model|λ|lambda|break-even|Pareto", "addressed", ("modest","cpu"), "Cost model and Pareto claim removed; plain latency/size reported", ""),
 (None, r"macro-F1|rare(-| )tool", "addressed", ("rare",), "Rare-tool macro-F1 reported", ""),
 (None, r"dynamic tool|failure mode|false negative|open-vocabulary", "addressed", ("vocab","esclim"), "Fixed vocabulary + escalation scope limitations", ""),
 (None, r"Lean|axiom|sorry|mechani[sz]ed", "addressed", ("lean","scope"), "Lean audit appendix, #print axioms", ""),
 (None, r"compression|18\.2M|parameter count|parameter-matched", "addressed", ("params","cpu"), "Counts from code; RF nodes from code; compression ratio removed", ""),
 (None, r"incremental|limited novelty|lack(s|ing)? of novelty|novelty is (limited|modest)", "partly", ("novelty","rw"),
  "Claims narrowed; novelty stated as limitation; cannot be raised by text", ""),
 ("T01", None, "addressed", ("rw","novelty"), "Overclaims removed; prior-art delta in Related Work", ""),
 ("T02", None, "addressed", ("hpo","hpoapp","budget"), "Equal-budget random search for all models (S6)", ""),
 ("T03", r"equivalence|TOST|non-inferior|mixed-effects|LME", "addressed", ("stats","ledger"), "Removed; differences tested only", ""),
 ("T03", None, "addressed", ("stats","holm"), "Seed-clustered paired bootstrap, Holm, DL pooling", ""),
 ("T04", None, "addressed", ("modest","where","hpo"), "Headline 'competitive and cheap'; default vs tuned scoped in Results, Discussion, Conclusion", "Superiority framing re-checked in this pass (default-vs-tuned qualifiers added)"),
 ("T05", None, "addressed", ("scope","jac","appb","formal"), "Theorem-scope table; measured Jacobian; App. B assumptions", ""),
 ("T06", None, "addressed", ("leak",), "Causal-operator claim removed; leakage proposition with proof", ""),
 ("T07", None, "addressed", ("tree","rw"), "Representability for a symbolic tree only; concordance claim removed", ""),
 ("T08", None, "addressed", ("escproto","esc","esclim"), "β fixed a priori, thresholds fixed, components ablated, entropy recommended", ""),
 ("T09", None, "addressed", ("lat","cpu","latlim"), "Measured on 7950X, 1 thread; single machine stated", ""),
 ("T10", r"leak|duplicat|split", "addressed", ("splits","bench"), "Per-benchmark held-out splits", ""),
 ("T10", None, "addressed", ("bench","splits"), "Four benchmarks; data and splits described", ""),
 ("T11", None, "partly", ("turn",), "Pilots removed; agent-level evaluation open", ""),
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
 ("P1","Escalation evaluated on K=8 factorial model, not primary","PAPER_REVIEW_V76C(_HPO).md minor","partly",("esclim",),"Stated as limitation; re-run on primary needs experiment"),
 ("P2","Factorial uses its own feature packing","PAPER_REVIEW_V76C.md minor","addressed",("factorial","sens"),"Stated in factorial description"),
 ("P3","Only 3 seeds","PAPER_REVIEW_1110/V76C minor","partly",("seeds",),"Limitation + MDE"),
 ("P4","LLM baselines zero-shot","PAPER_REVIEW_V76C minor","partly",("notrun",),"Limitation"),
 ("P5","Fig. 1(d) label touches arrow","PAPER_REVIEW_V76C nit","addressed",("fig1",),"Label offsets fixed in paper/generate_topology_figure.py; checked in rendered PDF"),
 ("P6","Cite on-device function-calling models","PAPER_REVIEW_V76C_HPO minor","addressed",("rw",),"Octopus v2, TinyAgent, Hammer, COLT added and verified"),
 ("P7","Table 5 placement","PAPER_REVIEW_V76C_HPO nit","addressed",("hpo",),"Float placement checked"),
 ("P8","One-seed HPO selection","PAPER_REVIEW_V76C_HPO minor","partly",("budget",),"Limitation; multi-seed selection needs runs"),
 ("P9","HGB default vs search early-stopping mismatch","PAPER_REVIEW_V76C_HPO minor","addressed",("budget",),"Disclosed"),
 ("P10","Tuned-model latency not measured","PAPER_REVIEW_V76C_HPO minor","partly",("budget",),"Disclosed; measurement needed"),
 ("K1","split_labels / number_conflict / Wilson FP","klaut 2026-11-03","addressed",("bench",),"Labels per split; klaut 10.0 now"),
 ("K2","Undifferentiated prior art","klaut 2026-11-03/1110","addressed",("rw",),"Delta stated per line of work"),
 ("K3","Runaway sentences / filenames in text","klaut 2026-11-03","addressed",("sources",),"Paths only in Table 8"),
 ("K4","β/threshold without calibration protocol","klaut (this pass)","addressed",("escproto","esclim"),"Limitation now points to fixed protocol"),
 ("O1","Single seed; no held-out test","draft review-1/2 (Oct)","addressed",("bench","seeds"),"3 seeds, held-out tests"),
 ("O2","Weak/unevaluated LLM baselines; DeepSeek/Llama pending","draft review-1..6","partly",("llm","notrun"),"Local LLMs only; API runs never produced result files"),
 ("O3","No OOD evaluation (blocker in review-3, fixed in review-5/6)","draft review-3..6","partly",("ood",),"REGRESSED then restored as limitation (S7)"),
 ("O4","Paired CIs between small models","draft review-3..6","addressed",("holm",),"Table 3"),
 ("O5","Multiplicity","draft review-4..6","addressed",("holm",),"Holm per family"),
 ("O6","Calibration","draft review-4..6","partly",("esc",),"AUROC/selective accuracy only; no ECE/reliability (reported earlier on old models)"),
 ("O7","Per-label errors","draft review-2..4","addressed",("rare",),"Macro-F1 / rare tools"),
 ("O8","Private code","draft review-2","addressed",("repo",),"Public repo"),
 ("S1","M1: Transformer 65.87 / Qwen 55.20 unsupported","wt_nca_research REVIEW.md","addressed",("ledger","notrun"),"Removed (ledger); Transformer not run - limitation"),
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
 ("L14","LEDGER_V76C_1110.md: every macro traced to S1-S7","LEDGER_V76C_1110.md","addressed",("sources",),"S7 added this pass"),
]

c = collections.Counter(r[2] for r in rows)
cm = collections.Counter(m[3] for m in M)
reg = [r for r in rows if r[5].startswith("REGRESSED")]
out = ["# Regression audit, 2026-11-10", "",
 "Every complaint from every past review of this manuscript line, with its status in the current `paper/main.tex`.",
 "Generated by `scripts/build_regression_audit_1110.py`; section names and line numbers are looked up in the current source.", "",
 "Status: **addressed** (fixed in the text), **partly** (stated honestly as a limitation; a full fix needs new experiments or cannot be made in text), **regressed** (resolved in an earlier version, lost in the current text). No item is left in the *regressed* state; items that were regressed are flagged in the note column and were fixed in this pass.", "",
 "## Summary", "",
 f"- aiXiv points: {len(rows)} from 63 reviews (v1.4-v7.6; v7.7-v9.0 have no reviews). addressed {c['addressed']}, partly {c['partly']}, regressed 0.",
 f"- Own reviews, klaut, earlier drafts, ledgers: {len(M)} items. addressed {cm['addressed']}, partly {cm['partly']}, regressed 0.",
 f"- Regressions found and fixed in this pass: OOD result dropped (O3; {sum('OOD' in r[5] for r in rows)} aiXiv points), power statement dropped ({sum('power' in r[5] for r in rows)} aiXiv points), two PR #8 ledger rows dropped (L5, L10), notation collisions (Δ, α, τ, t, C), superiority wording without default/tuned scope.",
 "", "## Open points that need new experiments (ask Julian before running)", "",
 "1. Agent-level task-success evaluation (T11, A16).", "2. OOD evaluation of the current models on held-out sessions (O3, T10).",
 "3. 5+ seeds for all models; multi-seed HPO selection (T03, A14, P8).", "4. End-to-end latency including featurisation; latency of tuned models (T09, P10).",
 "5. Transformer / GNN / other-NCA / retrieval / fine-tuned small function-calling baselines under this protocol (T02, M1).",
 "6. Escalation on the primary/tuned models; calibration (ECE) (P1, O6).", "7. Stronger LLM baselines (few-shot or fine-tuned) (O2).",
 "8. Novelty (T01, 'incremental'): cannot be fixed in text.", "",
 "## Own reviews, klaut, earlier drafts and ledgers", "",
 "| Point | Source review | Status | Where fixed (section + line) | Note |", "|---|---|---|---|---|"]
for m in M: out.append(f"| {m[0]}: {m[1]} | {m[2]} | {m[3]} | {L(*m[4])} | {m[5]} |")
out += ["", "## aiXiv review points (all versions)", "",
 "| Point | Source review | Status | Where fixed (section + line) | Note |", "|---|---|---|---|---|"]
for rid, p, st, where, note, rg in rows:
    txt = p["point"].replace("|", "/")
    out.append(f"| {rid} [{p['theme']}, {p['severity']}]: {txt} | aiXiv v{p['version']} (review {p['review_id']}, rating {p['rating']}) | {st} | {where} | {note}{'. ' + rg if rg else ''} |")
(ROOT / "paper/REGRESSION_AUDIT_1110.md").write_text("\n".join(out) + "\n")
print(len(rows), dict(c), dict(cm), "regressions flagged:", len(reg))
