#!/usr/bin/env python3
"""E1 v2: BFCL next-tool classification on the stratified ``bfcl_e1v2`` split.

Differences to v1 (run_e1_benchmark.py):
* split ``bfcl_e1v2`` (manifest: stage0/data_manifests/bfcl_e1v2.json), checked by hash at start;
* the TF-IDF+SVD feature pipeline is fitted on train for each seed and pickled
  next to the checkpoints (``seed{N}_features.pkl``); it is never read from the
  shared ``stage0_feats`` cache (that cache was keyed by seed only and caused
  v1 reload mismatches);
* evaluated weights = ``best_state`` (highest val top-1, earliest step on ties),
  saved separately as ``seed{N}_{topo}_best.pt``; reported NCA metrics are
  computed by reloading that file + the feature pickle on CPU (``eval_nca_from_disk``),
  so re-running the same function reproduces them exactly;
* sklearn rankings mapped through ``clf.classes_``; top-1 = ranked[:, 0];
* majority-class baseline (from train) row; macro-F1 over gold-test classes and
  over train∪test classes; per-example predictions; paired bootstrap of each
  model vs majority and NCA vs HGB/RF (per seed, on the same test indices).
No model selection on test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "stage0"))

SPLIT_NAME = "bfcl_e1v2"
MANIFEST = REPO_DIR / "stage0" / "data_manifests" / f"{SPLIT_NAME}.json"
RESULTS_DIR = REPO_DIR / "stage0" / "stage0_results" / "e1v2"
LABELS = json.load(open(REPO_DIR / "stage0" / "labels.json"))
L2I = {t: i for i, t in enumerate(LABELS)}
I2L = {i: t for t, i in L2I.items()}
N_CLASSES = len(LABELS)
SEQ_LEN = 5
HIDDEN_C = 16
NCA_STEPS = 4


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_split(splits_dir: Path) -> dict:
    man = json.loads(MANIFEST.read_text())
    for s, info in man["splits"].items():
        got = sha256_file(splits_dir / f"{s}.jsonl")
        if got != info["sha256"]:
            raise SystemExit(f"split {s} hash mismatch vs manifest: {got} != {info['sha256']}")
    return man


def fit_features(train, seed: int):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    from features import goal_text
    from obs_pack import GoalTfidf, SVD_DIM, TFIDF_KW

    texts = [goal_text(ex, True) for ex in train]
    tfidf = TfidfVectorizer(**TFIDF_KW)
    xt = tfidf.fit_transform(texts)
    svd = TruncatedSVD(SVD_DIM, random_state=seed).fit(xt)
    return GoalTfidf(tfidf, svd, seed)


def save_features(gt, path: Path) -> str:
    with path.open("wb") as f:
        pickle.dump({"tfidf": gt.tfidf, "svd": gt.svd, "seed": gt.seed}, f)
    return sha256_file(path)


def load_features(path: Path):
    from obs_pack import GoalTfidf

    with path.open("rb") as f:
        b = pickle.load(f)
    return GoalTfidf(b["tfidf"], b["svd"], b["seed"])


def build_model(obs_c: int, topo: str):
    from nca_text.model import GraphNCA

    return GraphNCA(obs_c=obs_c, hidden_c=HIDDEN_C, n_classes=N_CLASSES, steps=NCA_STEPS, seq_len=SEQ_LEN, topology=topo)


def ranked_logits(model, X, bs: int = 2048) -> np.ndarray:
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lg = model(X[i : i + bs]).double()
            # stable order: descending logit, ties -> lower label id
            out.append(np.argsort(-lg.cpu().numpy(), axis=1, kind="stable"))
    return np.concatenate(out) if out else np.zeros((0, N_CLASSES), dtype=np.int64)


def eval_nca_from_disk(best_path: Path, feat_path: Path, examples, topo: str) -> tuple[np.ndarray, np.ndarray]:
    """Canonical NCA evaluation: CPU, float32, weights + features loaded from disk."""
    import torch

    from obs_pack import pack_arch

    torch.use_deterministic_algorithms(True)
    gt = load_features(feat_path)
    X, y = pack_arch("nca1d", examples, L2I, "tfidf_svd128", gt)
    X = torch.from_numpy(X).float()
    blob = torch.load(best_path, map_location="cpu", weights_only=False)
    assert blob["feature_sha256"] == sha256_file(feat_path), "feature pipeline does not match checkpoint"
    model = build_model(X.shape[1], topo)
    model.load_state_dict(blob["state"])
    return ranked_logits(model, X), np.asarray(y)


def tab_features(gt, examples):
    from features import goal_text, struct_feats

    G = [goal_text(e, True) for e in examples]
    return np.hstack([gt.svd.transform(gt.tfidf.transform(G)), np.stack([struct_feats(e) for e in examples])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--topologies", type=str, default="tree,meta,star,1d")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--synthetic-jsonl", type=Path, default=None,
                    help="Synthetic rows appended to TRAIN only (ablation real+synthetic); val/test stay 100%% real")
    args = ap.parse_args()
    if args.dry_run:
        args.steps, args.eval_every, args.seeds, args.n_boot = min(args.steps, 1000), 250, "0", 1000

    from common import load_split, maybe_research_backup, require_nca_data_root, SPLITS

    require_nca_data_root()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    import eval_utils as eu
    from obs_pack import pack_arch

    man = verify_split(SPLITS / SPLIT_NAME)
    variant = "run_synth" if args.synthetic_jsonl else "run"
    res = args.results_dir / ("dry_run" if args.dry_run else variant)
    ck = res / "checkpoints"; pred_dir = res / "predictions"
    ck.mkdir(parents=True, exist_ok=True); pred_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    topos = [t.strip() for t in args.topologies.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = {s: load_split(SPLIT_NAME, s) for s in ("train", "val", "test")}
    synth_info = None
    if args.synthetic_jsonl:
        from common import load_jsonl_examples
        syn = load_jsonl_examples(args.synthetic_jsonl)
        real_ids = {e.session_id for s in sp for e in sp[s]}
        assert not ({e.session_id for e in syn} & real_ids), "synthetic ids collide with real ids"
        synth_info = {"path": str(args.synthetic_jsonl), "sha256": sha256_file(args.synthetic_jsonl), "rows": len(syn)}
        sp["train"] = sp["train"] + syn
    test_ids = [e.session_id for e in sp["test"]]
    y = {s: np.array([L2I[e.label] for e in sp[s]]) for s in sp}
    labels_all = sorted(set(y["train"].tolist()) | set(y["test"].tolist()))
    summary = {"experiment": "E1v2", "split": SPLIT_NAME, "split_manifest_sha256": sha256_file(MANIFEST),
               "n": {s: len(sp[s]) for s in sp}, "steps": args.steps, "seeds": seeds, "dry_run": args.dry_run,
               "evaluated_weights": "best val top-1 state (earliest on ties), reloaded from disk on CPU",
               "macro_f1_note": "macro_f1_gold averages over classes present in gold test; macro_f1_all over classes in train∪test",
               "device_train": str(dev), "synthetic_train": synth_info, "models": {}, "paired": {}}
    out_path = res / "e1v2_benchmark.json"
    t_start = time.time()

    def record(name, seed, ranked, extra=None):
        m = eu.full_metrics(y["test"], ranked, labels_all)
        m.update(eu.bootstrap_top1(ranked[:, 0] == y["test"], args.n_boot, seed))
        m["predictions_sha256"] = eu.write_predictions(pred_dir / f"{name}_seed{seed}.jsonl", test_ids, y["test"], ranked, I2L)
        summary["models"].setdefault(name, []).append({"seed": seed, "test": m, **(extra or {})})
        return ranked[:, 0] == y["test"]

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        correct: dict[str, np.ndarray] = {}
        feat_path = ck / f"seed{seed}_features.pkl"
        if feat_path.is_file():
            gt = load_features(feat_path)
        else:
            gt = fit_features(sp["train"], seed)
            save_features(gt, feat_path)
        feat_sha = sha256_file(feat_path)

        maj_r = eu.majority_ranked(y["train"], len(y["test"]), N_CLASSES)
        correct["Majority"] = record("Majority", seed, maj_r, {"label": I2L[int(maj_r[0, 0])]})

        Xtab = {s: tab_features(gt, sp[s]) for s in sp}
        for name, clf in (
            ("HistGradientBoosting", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=30 if args.dry_run else 200, max_leaf_nodes=31, random_state=seed)),
            ("RandomForest", RandomForestClassifier(n_estimators=10 if args.dry_run else 50, max_depth=15, random_state=seed, n_jobs=4)),
        ):
            clf.fit(Xtab["train"], y["train"])
            vr = eu.rank_from_proba(clf.predict_proba(Xtab["val"]), clf.classes_)
            tr = eu.rank_from_proba(clf.predict_proba(Xtab["test"]), clf.classes_)
            with (ck / f"seed{seed}_{name}.pkl").open("wb") as f:
                pickle.dump(clf, f)
            correct[name] = record(name, seed, tr, {"val_top1": round(100 * float((vr[:, 0] == y["val"]).mean()), 2)})

        Xs = {s: torch.from_numpy(pack_arch("nca1d", sp[s], L2I, "tfidf_svd128", gt)[0]).float() for s in sp}
        Xtr_d = Xs["train"].to(dev); ytr_d = torch.from_numpy(y["train"]).to(dev)
        counts = np.bincount(y["train"], minlength=N_CLASSES).astype(np.float32)
        cw = torch.tensor(1.0 / np.sqrt(counts + 1), device=dev); cw = cw / cw.mean()
        n_train = len(y["train"]); batch = min(args.batch, n_train)

        for topo in topos:
            key = f"GraphNCA_{topo}"
            ck_path = ck / f"seed{seed}_{topo}.pt"; best_path = ck / f"seed{seed}_{topo}_best.pt"
            model = build_model(Xs["train"].shape[1], topo).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            start, best_val, best_step = 1, -1.0, 0
            if ck_path.is_file():
                b = torch.load(ck_path, map_location=dev, weights_only=False)
                assert b["feature_sha256"] == feat_sha
                model.load_state_dict(b["model"]); opt.load_state_dict(b["optimizer"])
                start, best_val, best_step = b["step"] + 1, b["best_val"], b["best_step"]
                rng = b["rng"]; torch.set_rng_state(b["torch_rng"])
                print(f"resume {key} seed={seed} from step {start}", flush=True)
            t0 = time.time()
            for step in range(start, args.steps + 1):
                lr = 1e-3 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps)))
                for g in opt.param_groups:
                    g["lr"] = lr
                bi = torch.from_numpy(rng.choice(n_train, batch)).to(dev)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(Xtr_d[bi]), ytr_d[bi], weight=cw)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                if step % args.eval_every == 0 or step == args.steps:
                    vr = ranked_logits(model, Xs["val"].to(dev)); model.train()
                    v = 100 * float((vr[:, 0] == y["val"]).mean())
                    if v > best_val:
                        best_val, best_step = v, step
                        torch.save({"state": {k: t.detach().cpu().clone() for k, t in model.state_dict().items()},
                                    "step": step, "val_top1": v, "topology": topo, "seed": seed,
                                    "feature_sha256": feat_sha, "split_manifest_sha256": summary["split_manifest_sha256"]}, best_path)
                    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "best_val": best_val,
                                "best_step": best_step, "rng": rng, "torch_rng": torch.get_rng_state(), "feature_sha256": feat_sha}, ck_path)
                    if step % (args.eval_every * 20) == 0 or step == args.steps:
                        print(f"{key} seed={seed} step {step}/{args.steps} loss={loss.item():.3f} val={v:.2f} best={best_val:.2f}@{best_step} ({time.time()-t0:.0f}s)", flush=True)
            r1, _ = eval_nca_from_disk(best_path, feat_path, sp["test"], topo)
            r2, _ = eval_nca_from_disk(best_path, feat_path, sp["test"], topo)
            assert np.array_equal(r1, r2), "reload eval not reproducible"
            correct[key] = record(key, seed, r1, {"val_top1": round(best_val, 2), "best_step": best_step,
                                                  "best_checkpoint": str(best_path), "best_checkpoint_sha256": sha256_file(best_path),
                                                  "feature_sha256": feat_sha, "train_time_s": round(time.time() - t0, 1)})
            if topo == "meta":
                from graph_topology_export import meta_export_record
                blob = torch.load(best_path, map_location="cpu", weights_only=False)
                (ck / f"seed{seed}_meta_adjacency.json").write_text(json.dumps(meta_export_record(blob["state"], seed=seed, checkpoint=str(best_path)), indent=2))
            out_path.write_text(json.dumps(summary, indent=2))
            maybe_research_backup(f"e1v2-seed{seed}-{topo}")

        pair = {}
        for a in correct:
            if a != "Majority":
                pair[f"{a}_minus_Majority"] = eu.paired_bootstrap(correct[a], correct["Majority"], args.n_boot, seed)
        for t in topos:
            for b in ("HistGradientBoosting", "RandomForest"):
                pair[f"GraphNCA_{t}_minus_{b}"] = eu.paired_bootstrap(correct[f"GraphNCA_{t}"], correct[b], args.n_boot, seed)
        summary["paired"][f"seed{seed}"] = pair
        summary["elapsed_s"] = round(time.time() - t_start, 1)
        out_path.write_text(json.dumps(summary, indent=2))
        maybe_research_backup(f"e1v2-seed{seed}")
    print(f"E1v2 complete -> {out_path} ({time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
