#!/usr/bin/env python3
"""Rewrite synthetic train goals into real BFCL goal format (Julian 2026-11-03, option a).

Real goals = question text + names of ALL offered functions (_question_text, <=500 chars).
Here: (1) the klaut EU Azure deployment rewrites each synthetic question so it names no
function/identifier; (2) an offered-function list is appended, copied from a random REAL
TRAIN row of another class (keeps the real list-size/name distribution), with any name that
maps to the synthetic row's label removed; the correct function is never named.
Leak check: per class, rate of rows whose goal contains the correct function name must be
<= the rate in real train rows of that class. Then exact + TF-IDF cosine>0.9 dedupe vs val/test.
"""

from __future__ import annotations

import argparse, hashlib, json, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1v2_synth_generate import DEPLOYMENT, ENDPOINT, PRICE_IN, PRICE_OUT, chat  # noqa: E402
from bfcl_label_map import map_bfcl_function  # noqa: E402


def split_fn(goal: str) -> tuple[str, str]:
    toks = goal.split(); fn = toks[-1]
    while len(toks) > 1 and toks[-1] == fn:
        toks.pop()
    return " ".join(toks), fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True); ap.add_argument("--train-fns", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--budget-eur", type=float, default=2.0)
    args = ap.parse_args()
    from common import load_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    src = [json.loads(l) for l in args.inp.open()]
    tf = json.load(args.train_fns.open()); real_rows = tf["rows"]; real_rate = tf["gold_in_goal_rate"]
    rng = random.Random(20261104)
    log = args.out.with_suffix(".batches.jsonl"); log.unlink(missing_ok=True)
    spent = 0.0; rewritten: dict[int, str] = {}
    for b0 in range(0, len(src), args.batch):
        chunk = src[b0 : b0 + args.batch]
        qs = [{"i": b0 + k, "text": split_fn(r["goal"])[0]} for k, r in enumerate(chunk)]
        prompt = ("Rewrite each user request so it contains NO function, tool, API or command identifier names "
                  "(no snake_case/dotted identifiers, no tool names like read_file/todo_add/cmd_controller). Keep the "
                  "meaning, concrete details (file names, paths, shell commands the user wants run are fine) and natural tone. "
                  'Return JSON {"items":[{"i":int,"text":str}]} with the same i values.\n' + json.dumps(qs, ensure_ascii=False))
        seed = 7000 + b0
        resp = chat([{"role": "user", "content": prompt}], 0.3, seed, max_tokens=4000)
        u = resp.get("usage", {}); cost = (u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT) / 1e6
        spent += cost
        got = {int(x["i"]): str(x["text"]).strip() for x in json.loads(resp["choices"][0]["message"]["content"]).get("items", [])}
        rewritten.update(got)
        with log.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "batch": b0 // args.batch, "deployment": DEPLOYMENT, "endpoint": ENDPOINT,
                                "temperature": 0.3, "seed": seed, "prompt": prompt, "usage": u, "cost_eur_est": round(cost, 5),
                                "spent_eur_est": round(spent, 4), "returned": len(got)}) + "\n")
        if spent > args.budget_eur:
            raise SystemExit("budget exceeded")

    held = [e.goal for s in ("val", "test") for e in load_split("bfcl_e1v2", s)]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(held + [r["goal"] for r in src]); H = vec.transform(held)
    held_exact = {" ".join(g.lower().split()) for g in held}
    out, drops, acc = [], Counter(), []
    leak = defaultdict(lambda: [0, 0])
    for i, r in enumerate(src):
        lab = r["target_tool"]; _, gold = split_fn(r["goal"])
        q = rewritten.get(i)
        if not q:
            drops["no_rewrite"] += 1; continue
        if gold.lower() in q.lower():
            drops["gold_in_question"] += 1; continue
        donor = rng.choice([x for x in real_rows if x["label"] != lab and x["fns"]])
        offered = [f for f in donor["fns"] if map_bfcl_function(f) != lab and f.lower() != gold.lower()]
        goal = " ".join([q] + offered)[:500]
        norm = " ".join(goal.lower().split())
        if norm in held_exact:
            drops["exact_dup_val_test"] += 1; continue
        v = vec.transform([goal])
        if cosine_similarity(v, H).max() > 0.9:
            drops["near_dup_val_test"] += 1; continue
        if acc and cosine_similarity(v, vec.transform(acc)).max() > 0.9:
            drops["near_dup_synth"] += 1; continue
        leak[lab][1] += 1; leak[lab][0] += int(gold.lower() in goal.lower())
        acc.append(goal)
        out.append({**r, "goal": goal, "gold_function_hidden": gold, "offered_functions": offered,
                    "distractor_donor_label": donor["label"], "format": "bfcl_question+offered_functions_v2"})
    check = {lab: {"synth": [a, n, round(a / n, 3)], "real_train": real_rate.get(lab), "pass": (a / n) <= (real_rate.get(lab, [0, 0, 0])[2])}
             for lab, (a, n) in leak.items()}
    args.out.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(json.dumps({"rows": len(out), "labels": Counter(x["target_tool"] for x in out), "drops": drops,
                      "leak_check": check, "spent_eur_est": round(spent, 4)}, indent=1))


if __name__ == "__main__":
    main()
