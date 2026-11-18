#!/usr/bin/env python3
"""GNCA System-1 router service for the official BFCL v4 run (runs in the paper env, nca_venv).

Loads the Hermes-trained Tree K=8 label-smoothed GraphNCA checkpoint (rule_depth=2, hidden=16,
seq_len=5) with the seed-0 TF-IDF+SVD128 goal featurizer it was trained with, and serves
POST /route {"goal": str, "steps": [[tool_label, result_text], ...]} -> {"top1", "probs"}.
Labels are the 13-label Hermes vocabulary (stage0/labels.json). CPU only, deterministic.
"""
from __future__ import annotations

import argparse, json, os, sys, hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.environ.get("NCA_REPO", "/home/s4il0r/nca_bfcl_official_1111"))
ap.add_argument("--checkpoint", default=os.environ.get("GNCA_CHECKPOINT"))
ap.add_argument("--tfidf-seed", type=int, default=0)
ap.add_argument("--port", type=int, default=int(os.environ.get("GNCA_ROUTER_PORT", "18777")))
ap.add_argument("--selftest", action="store_true", help="evaluate on curated test split and exit")
args = ap.parse_args()

sys.path.insert(0, args.repo); sys.path.insert(0, str(Path(args.repo) / "stage0"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from nca_text.data import Example  # noqa: E402
from nca_text.model import GraphNCA  # noqa: E402
from obs_pack import GoalTfidf, pack_arch, _feats_dir  # noqa: E402
import pickle  # noqa: E402

torch.set_num_threads(1)
LABELS = json.load(open(Path(args.repo) / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
with open(_feats_dir() / f"tfidf_svd128_seed{args.tfidf_seed}.pkl", "rb") as f:
    blob = pickle.load(f)
GT = GoalTfidf(blob["tfidf"], blob["svd"], args.tfidf_seed)
_X0, _ = pack_arch("nca1d", [Example("probe", "respond", "probe", [])], L2I, "tfidf_svd128", GT)
MODEL = GraphNCA(obs_c=_X0.shape[1], hidden_c=16, n_classes=len(LABELS), steps=8, rule_depth=2,
                 seq_len=5, topology="tree")
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
if isinstance(state, dict) and "state" in state:
    state = state["state"]
MODEL.load_state_dict(state)
MODEL.eval()
CKPT_SHA = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()


@torch.no_grad()
def route(examples):
    X, _ = pack_arch("nca1d", examples, L2I, "tfidf_svd128", GT)
    p = torch.softmax(MODEL(torch.from_numpy(X).float()), dim=-1).numpy()
    return p


if args.selftest:
    from common import load_split
    test = load_split("curated", "test")
    y = np.array([L2I[e.label] for e in test if e.label in L2I])
    p = route(test)
    print(json.dumps({"checkpoint": args.checkpoint, "sha256": CKPT_SHA, "n_test": int(len(y)),
                      "curated_test_top1": round(100 * float((p.argmax(1) == y).mean()), 2),
                      "pred_respond_rate": round(100 * float((p.argmax(1) == L2I["respond"]).mean()), 2)}))
    sys.exit(0)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"ok": True, "checkpoint": args.checkpoint, "sha256": CKPT_SHA, "labels": LABELS}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        steps = [(str(t), str(r)) for t, r in req.get("steps", [])]
        p = route([Example("q", "respond", str(req.get("goal", "")), steps)])[0]
        body = json.dumps({"top1": LABELS[int(p.argmax())], "probs": {l: round(float(v), 5) for l, v in zip(LABELS, p)}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    print(f"GNCA router on 127.0.0.1:{args.port} ckpt={args.checkpoint} sha256={CKPT_SHA}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
