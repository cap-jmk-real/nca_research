#!/usr/bin/env python3
"""Macros + table for the official BFCL v4 end-to-end check (baseline qwen3.5-9b vs. the same model with
the GNCA respond gate). Input: stage0/stage0_results/bfcl_official_1116/summary.json (compacted from the engine run
official_bfcl_v4_20261111; every value is read from the official evaluate score files, the gate decision
logs and the parse-fallback logs). Output: paper/generated/{numbers_bfcl_official.tex,tab_bfcl_official.tex}.
No number is typed by hand."""
import json, math
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
S = json.loads((REPO / "stage0/stage0_results/bfcl_official_1116/summary.json").read_text())
GEN = REPO / "paper/generated"
B, G = S["new"]["baseline_no_gnca"], S["new"]["gnca_gate"]
P, GT = S["paired_exact_mcnemar_unadjusted"], S["gate"]
OLD = S["old_pp15_mtcheck"]
ORDER = [("simple_python","simple Python"),("simple_java","simple Java"),("simple_javascript","simple JavaScript"),("multiple","multiple"),
 ("parallel","parallel"),("parallel_multiple","parallel multiple"),("irrelevance","irrelevance"),
 ("live_simple","live simple"),("live_multiple","live multiple"),("live_parallel","live parallel"),("live_parallel_multiple","live parallel multiple"),
 ("live_irrelevance","live irrelevance"),("live_relevance","live relevance"),
 ("multi_turn_base","MT base"),("multi_turn_miss_func","MT miss func"),("multi_turn_miss_param","MT miss param"),("multi_turn_long_context","MT long context"),
 ("web_search_base","web search base"),("web_search_no_snippet","web search no snippet"),
 ("memory_kv","memory KV"),("memory_vector","memory vector"),("memory_rec_sum","memory rec.\\ sum.")]
assert len(ORDER) == 22 and all(k in B and k in G for k, _ in ORDER)
def pc(x): return f"{100*x:.1f}"
def pv(p): return "$<$0.001" if p < 0.001 else f"{p:.3f}"
def cam(s): return "".join(w.capitalize() for w in s.replace("_", " ").split())
M = {}
def put(k, v): M[k] = v
for c, D, tag in (("baseline_no_gnca", B, "Base"), ("gnca_gate", G, "Gate")):
    for k, _ in ORDER:
        put(f"bfAcc{tag}{cam(k)}", pc(D[k]["acc"]))
    for grp, keys in {"MT": [k for k,_ in ORDER if k.startswith("multi_turn")], "Mem": [k for k,_ in ORDER if k.startswith("memory")],
                      "Web": ["web_search_base","web_search_no_snippet"]}.items():
        n = sum(D[k]["total"] for k in keys); co = sum(D[k]["correct"] for k in keys)
        put(f"bf{grp}Acc{tag}", pc(co / n)); put(f"bf{grp}N", str(n))
    allc = sum(D[k]["correct"] for k, _ in ORDER); alln = sum(D[k]["total"] for k, _ in ORDER)
    put(f"bfPooledAcc{tag}", pc(allc / alln)); put("bfNEntries", str(alln))
    put(f"bfOverall{tag}", S["overall"][c]["Overall Acc"].rstrip("%"))
    for key, name in (("Multi Turn Acc","MTSub"),("Non-Live AST Acc","NonLive"),("Live Acc","Live"),("Memory Acc","MemSub"),("Web Search Acc","WebSub")):
        put(f"bf{name}{tag}", S["overall"][c][key].rstrip("%"))
    mt = [k for k,_ in ORDER if k.startswith("multi_turn")]
    put(f"bfIE{tag}", str(sum(D[k]["inference_error"] for k in mt)))
    put(f"bfEmpty{tag}", str(sum(D[k]["empty_turn"] for k in mt)))
    put(f"bfResErr{tag}", str(sum(D[k]["result_file_inference_errors"] for k, _ in ORDER)))
    pf = S["parse_fallback"][c]
    put(f"bfFallback{tag}", str(sum(r[2] for r in pf)))
    put(f"bfFallbackWithheld{tag}", str(sum(r[2] for r in pf if r[1] == "withheld")))
mtk = [k for k,_ in ORDER if k.startswith("multi_turn")]
memk = [k for k,_ in ORDER if k.startswith("memory")]
put("bfGateWithheldMT", str(sum(GT[k]["withheld_rows"] for k in mtk))); put("bfGateRowsMT", str(sum(GT[k]["decision_rows"] for k in mtk)))
put("bfGateEntWithheldMT", str(sum(GT[k]["entries_with_withheld"] for k in mtk))); put("bfGateEntMT", str(sum(GT[k]["entries"] for k in mtk)))
put("bfGateWithheldAll", str(sum(GT[k]["withheld_rows"] for k, _ in ORDER))); put("bfGateRowsAll", str(sum(GT[k]["decision_rows"] for k, _ in ORDER)))
sl = [k for k,_ in ORDER if not (k.startswith("multi_turn") or k.startswith("memory") or k.startswith("web"))]
put("bfGateWithheldSingle", str(sum(GT[k]["withheld_rows"] for k in sl))); put("bfGateRowsSingle", str(sum(GT[k]["decision_rows"] for k in sl)))
put("bfNDiscBase", str(P["_all_entries"]["base_only_correct"])); put("bfNDiscGate", str(P["_all_entries"]["gate_only_correct"]))
for grp, keys in (("MT", mtk), ("Mem", memk), ("Single", sl)):
    put(f"bf{grp}DiscBase", str(sum(P[k]["base_only_correct"] for k in keys))); put(f"bf{grp}DiscGate", str(sum(P[k]["gate_only_correct"] for k in keys)))
# old (pp 1.5, interim multi_turn check) vs new, same categories
for c, tag, D in (("baseline_no_gnca","Base",B),("gnca_gate","Gate",G)):
    for k in mtk:
        if k in OLD[c]:
            o = OLD[c][k]
            put(f"bfOld{tag}{cam(k)}", pc(o["acc"])); put(f"bfOld{tag}{cam(k)}N", str(o["total"]))
            put(f"bfOld{tag}{cam(k)}IE", str(o["inference_error"]))
            put(f"bfOld{tag}{cam(k)}Empty", str(o["empty_turn"]))
    put(f"bfOldIE{tag}", str(sum(OLD[c][k]["inference_error"] for k in OLD[c])))
    put(f"bfOldIEcats{tag}", str(len(OLD[c])))
# Holm over the 22 categories (two-sided exact McNemar p, unadjusted in the summary)
ps = sorted((P[k]["p_exact"], k) for k, _ in ORDER); surv = []
for i, (pp, k) in enumerate(ps):
    if pp * (len(ps) - i) <= 0.05: surv.append(k)
    else: break
put("bfHolmN", str(len(surv)))
nm = dict(ORDER); put("bfHolmList", ", ".join(nm[k] for k in surv if k in nm))
out = ["% AUTO-GENERATED by scripts/build_bfcl_official_1116.py -- do not edit by hand"]
out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
(GEN / "numbers_bfcl_official.tex").write_text("\n".join(out) + "\n")
rows = ["% AUTO-GENERATED by scripts/build_bfcl_official_1116.py", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{@{}lrrrrrrrrrrr@{}}", "\\toprule",
 "& & \\multicolumn{2}{c}{Accuracy (\\%)} & & \\multicolumn{2}{c}{Entries only one gets right} & \\multicolumn{2}{c}{Inference errors$^a$} & \\multicolumn{2}{c}{Empty-turn failures$^b$} & Gate withheld$^c$ \\\\",
 "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
 "Category & $n$ & Baseline & Gate & $p$ & Base & Gate & Base & Gate & Base & Gate & steps (entries) \\\\", "\\midrule"]
for i, (k, name) in enumerate(ORDER):
    if k in ("multi_turn_base", "web_search_base", "memory_kv") or k == "live_simple" or k == "simple_python":
        if i: rows.append("\\midrule")
    b, g = B[k], G[k]
    mt = k.startswith("multi_turn")
    ie = (str(b["inference_error"]), str(g["inference_error"])) if mt else (str(b["result_file_inference_errors"]), str(g["result_file_inference_errors"]))
    em = (str(b["empty_turn"]), str(g["empty_turn"])) if mt else ("--", "--")
    gt = f"{GT[k]['withheld_rows']} ({GT[k]['entries_with_withheld']})"
    rows.append(f"{name} & {b['total']} & {pc(b['acc'])} & {pc(g['acc'])} & {pv(P[k]['p_exact'])} & {P[k]['base_only_correct']} & {P[k]['gate_only_correct']} & {ie[0]} & {ie[1]} & {em[0]} & {em[1]} & {gt} \\\\")
rows += ["\\bottomrule", "\\end{tabular}}"]
(GEN / "tab_bfcl_official.tex").write_text("\n".join(rows) + "\n")
print(len(M), "macros")
