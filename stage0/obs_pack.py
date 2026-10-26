"""Neural observation packing with optional --text_feat (default = nca_text hash48)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from common import DATA
from features import goal_text, struct_feats
from nca_text.data import (
    HASH_C,
    HASH_DIM,
    GRID_H,
    GRID_W,
    MAX_STEPS,
    TRACE_STEPS,
    SEQ_LEN,
    build_obs,
    build_trace_obs,
    hash_features,
)
from hash_text import hash_wide

TFIDF_KW = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "dtype": np.float32,
}
SVD_DIM = 128
GOAL_TF_DIM = SVD_DIM + 4


class GoalTfidf:
    def __init__(self, tfidf: TfidfVectorizer, svd: TruncatedSVD, seed: int) -> None:
        self.tfidf = tfidf
        self.svd = svd
        self.seed = seed

    def encode_goal(self, ex) -> np.ndarray:
        x = self.tfidf.transform([goal_text(ex, True)])
        v = self.svd.transform(x)[0].astype(np.float32)
        s = struct_feats(ex)[-4:].astype(np.float32)
        return np.concatenate([v, s])


def _feats_dir() -> Path:
    d = DATA / "stage0_feats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fit_or_load_tfidf(train_examples: list, seed: int) -> GoalTfidf:
    path = _feats_dir() / f"tfidf_svd128_seed{seed}.pkl"
    if path.is_file():
        with path.open("rb") as f:
            blob = pickle.load(f)
        return GoalTfidf(blob["tfidf"], blob["svd"], seed)
    texts = [goal_text(ex, True) for ex in train_examples]
    tfidf = TfidfVectorizer(**TFIDF_KW)
    xt = tfidf.fit_transform(texts)
    svd = TruncatedSVD(SVD_DIM, random_state=seed).fit(xt)
    obj = GoalTfidf(tfidf, svd, seed)
    with path.open("wb") as f:
        pickle.dump({"tfidf": tfidf, "svd": svd}, f)
    return obj


def _cell_base_dim(text_feat: str) -> int:
    if text_feat == "tfidf_svd128":
        return GOAL_TF_DIM
    if text_feat == "hash_wide":
        return 512
    return HASH_DIM


def pack_trace(examples: list, vocab: dict[str, int], text_feat: str, goal_tfidf: GoalTfidf | None):
    if text_feat == "hash48":
        return build_trace_obs(examples, vocab)[:2]
    base = _cell_base_dim(text_feat)
    n_classes = len(vocab)
    obs_c = base + n_classes + 1
    usable = [ex for ex in examples if ex.label in vocab]
    if not usable:
        return np.zeros((0, obs_c, SEQ_LEN), np.float32), np.zeros((0,), np.int64)
    obs_all = np.zeros((len(usable), obs_c, SEQ_LEN), np.float32)
    labels = np.empty(len(usable), np.int64)
    for n, ex in enumerate(usable):
        obs = obs_all[n]
        if text_feat == "tfidf_svd128":
            assert goal_tfidf is not None
            obs[:GOAL_TF_DIM, 0] = goal_tfidf.encode_goal(ex)
        elif text_feat == "hash_wide":
            obs[:512, 0] = hash_wide(goal_text(ex, True))
        obs[-1, 0] = 1.0
        steps = ex.steps[-TRACE_STEPS:]
        start = SEQ_LEN - len(steps)
        for i, (tool, text) in enumerate(steps):
            col = start + i
            obs[:HASH_DIM, col] = hash_features(f"{tool} {text}".strip(), HASH_DIM)
            idx = vocab.get(tool)
            if idx is not None:
                obs[base + idx, col] = 1.0
            obs[-1, col] = 1.0
        labels[n] = vocab[ex.label]
    return obs_all, labels


def pack_grid(examples: list, vocab: dict[str, int], text_feat: str, goal_tfidf: GoalTfidf | None):
    if text_feat == "hash48":
        return build_obs(examples, vocab)[:2]
    hash_dim = HASH_C * GRID_W
    base = _cell_base_dim(text_feat)
    obs_c = HASH_C + len(vocab)
    usable = [ex for ex in examples if ex.label in vocab]
    if not usable:
        return np.zeros((0, obs_c, GRID_H, GRID_W), np.float32), np.zeros((0,), np.int64)
    obs_all = np.zeros((len(usable), obs_c, GRID_H, GRID_W), np.float32)
    labels = np.empty(len(usable), np.int64)
    for n, ex in enumerate(usable):
        obs = obs_all[n]
        if text_feat == "tfidf_svd128":
            assert goal_tfidf is not None
            g = goal_tfidf.encode_goal(ex)
            padded = np.zeros(hash_dim, np.float32)
            padded[: min(len(g), hash_dim)] = g[:hash_dim]
            obs[:HASH_C, 0] = padded.reshape(HASH_C, GRID_W)
        elif text_feat == "hash_wide":
            g = hash_wide(goal_text(ex, True))
            padded = np.zeros(hash_dim, np.float32)
            padded[: min(len(g), hash_dim)] = g[:hash_dim]
            obs[:HASH_C, 0] = padded.reshape(HASH_C, GRID_W)
        steps = ex.steps[-MAX_STEPS:]
        start = GRID_H - len(steps)
        for i, (tool, text) in enumerate(steps):
            row = start + i
            blob = hash_features(f"{tool} {text}".strip(), hash_dim)
            obs[:HASH_C, row] = blob.reshape(HASH_C, GRID_W)
            idx = vocab.get(tool)
            if idx is not None:
                obs[HASH_C + idx, row, :] = 1.0
        labels[n] = vocab[ex.label]
    return obs_all, labels


def pack_arch(arch: str, examples: list, vocab: dict[str, int], text_feat: str, goal_tfidf: GoalTfidf | None):
    if arch == "nca2d":
        return pack_grid(examples, vocab, text_feat, goal_tfidf)
    return pack_trace(examples, vocab, text_feat, goal_tfidf)
