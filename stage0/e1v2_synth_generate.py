#!/usr/bin/env python3
"""Synthetic TRAIN rows for under-represented bfcl_e1v2 classes via the klaut EU Azure deployment.

Rules (Julian, 2026-11-03): train only; val/test stay 100% real and are never shown
to the model; prompts use only TRAIN rows as style examples; drop exact and
near-duplicates (TF-IDF cosine > 0.9) vs val and test (and vs earlier synthetic rows);
rows labelled is_synthetic/source; per-batch log of prompt, deployment, temperature,
seed, tokens and cost estimate; hard budget cap. The key comes from the environment
(KLAUT_AZURE_OPENAI_API_KEY) and is never printed or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ENDPOINT = "https://klaut-research-aoai.openai.azure.com/"
DEPLOYMENT = "gpt-4.1-mini"
API_VERSION = "2025-01-01-preview"
# gpt-4.1-mini list price (USD per 1M tokens); conservative EUR≈USD
PRICE_IN, PRICE_OUT = 0.40, 1.60
LABEL_HINT = {
    "search_files": "look up / fetch / search / list / query information",
    "terminal": "run a shell or terminal command",
    "write_file": "create, write, save or update a file or record",
    "read_file": "open or read a specific file",
    "todo": "create or manage a task / todo item",
}


def chat(messages, temperature, seed, max_tokens=1200):
    key = os.environ["KLAUT_AZURE_OPENAI_API_KEY"]
    url = f"{ENDPOINT}openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    body = json.dumps({"messages": messages, "temperature": temperature, "seed": seed, "max_tokens": max_tokens,
                       "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(url, data=body, headers={"api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-class", type=int, default=200)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--budget-eur", type=float, default=3.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from common import SPLITS, load_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    train = load_split("bfcl_e1v2", "train")
    held = [e.goal for s in ("val", "test") for e in load_split("bfcl_e1v2", s)]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([e.goal for e in train] + held)
    H = vec.transform(held)
    held_exact = {" ".join(g.lower().split()) for g in held}
    counts = Counter(e.label for e in train)
    log_path = args.out.with_suffix(".batches.jsonl")
    rows, accepted_texts, spent = [], [], 0.0
    rng = random.Random(20261103)
    for label, hint in LABEL_HINT.items():
        need = max(0, args.target_per_class - counts.get(label, 0))
        pool = [e.goal for e in train if e.label == label] or [e.goal for e in train]
        b = 0
        while need > 0 and spent < args.budget_eur:
            shots = rng.sample(pool, min(5, len(pool)))
            prompt = (f"Generate {args.batch} diverse, realistic user requests to an AI agent whose correct next action is "
                      f"'{label}' ({hint}). Each request should mention one plausible function name. Vary domain and length. "
                      f"Style examples (do not copy):\n" + "\n".join(f"- {s[:300]}" for s in shots) +
                      '\nReturn JSON {"items": [{"goal": str, "function_name": str}]}.')
            seed = 1000 * len(rows) + b
            resp = chat([{"role": "user", "content": prompt}], args.temperature, seed)
            u = resp.get("usage", {})
            cost = (u.get("prompt_tokens", 0) * PRICE_IN + u.get("completion_tokens", 0) * PRICE_OUT) / 1e6
            spent += cost
            items = json.loads(resp["choices"][0]["message"]["content"]).get("items", [])
            kept = drop = 0
            from bfcl_label_map import map_bfcl_function
            for it in items:
                g = f"{it.get('goal', '').strip()} {it.get('function_name', '').strip()}".strip()[:500]
                norm = " ".join(g.lower().split())
                ok = bool(g) and norm not in held_exact and map_bfcl_function(it.get("function_name", "")) in (label, "execute_code")
                if ok:
                    v = vec.transform([g])
                    ok = cosine_similarity(v, H).max() <= 0.9 and (not accepted_texts or cosine_similarity(v, vec.transform(accepted_texts)).max() <= 0.9)
                if not ok:
                    drop += 1; continue
                sid = "synth_" + hashlib.sha256(g.encode()).hexdigest()[:16]
                rows.append({"session_id": sid, "step_index": 0, "goal": g, "recent_steps": [], "target_tool": label,
                             "is_synthetic": True, "source": f"azure:{DEPLOYMENT}:{API_VERSION}:swedencentral"})
                accepted_texts.append(g); kept += 1; need -= 1
            with log_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "label": label, "batch": b, "deployment": DEPLOYMENT, "endpoint": ENDPOINT,
                                    "temperature": args.temperature, "seed": seed, "prompt": prompt, "usage": u,
                                    "cost_eur_est": round(cost, 5), "spent_eur_est": round(spent, 4), "kept": kept, "dropped": drop}) + "\n")
            b += 1
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"rows": len(rows), "by_label": Counter(r["target_tool"] for r in rows), "spent_eur_est": round(spent, 4)}))


if __name__ == "__main__":
    main()
