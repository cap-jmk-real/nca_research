#!/usr/bin/env python3
"""Deterministic train / held-out split of official BFCL v4 entry ids for GNCA-gate training.

Split unit = entry id (one entry = one independent session/conversation), hashed with a fixed salt,
so no session appears in both sides. Held-out ids are NEVER used for training; the trained gate is
evaluated vs. baseline on held-out ids only. Pure CPU, reads bundled bfcl_eval data only.
"""
import hashlib, json, sys
from pathlib import Path

D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SALT = "gnca-bfcl-split-v1"
CATS = ["simple_python","simple_java","simple_javascript","multiple","parallel","parallel_multiple",
        "irrelevance","live_simple","live_multiple","live_parallel","live_parallel_multiple",
        "live_irrelevance","live_relevance","multi_turn_base","multi_turn_miss_func",
        "multi_turn_miss_param","multi_turn_long_context","memory"]
split = {}
for c in CATS:
    f = D / f"BFCL_v4_{c}.json"
    ids = [json.loads(l)["id"] for l in f.read_text().splitlines() if l.strip()]
    tr, ho = [], []
    for i in ids:
        (ho if int(hashlib.sha256((SALT + i).encode()).hexdigest(), 16) % 2 else tr).append(i)
    split[c] = {"train": tr, "heldout": ho}
    print(f"{c:28s} total={len(ids):4d} train={len(tr):4d} heldout={len(ho):4d}")
OUT.write_text(json.dumps({"salt": SALT, "split": split}, indent=1))
