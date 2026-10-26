"""Shared goal/result text and struct features (must match legacy sota_v2 / sota_gbdt)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_LABELS = json.load(open(Path(__file__).resolve().parent / "labels.json"))
_L = {t: i for i, t in enumerate(_LABELS)}
_K = len(_LABELS) + 1


def label_struct_k() -> tuple[dict[str, int], int]:
    return _L, _K


def goal_text(ex, with_result: bool) -> str:
    """Same string as sota_v2 ``text(e, res)`` / sota_gbdt ``G`` with result."""
    return (ex.goal or "") + (
        (" || " + " ".join(str(r)[:200] for _, r in list(ex.steps)[-1:])) if with_result else ""
    )


def shared_text(ex, include_result: bool = True) -> str:
    return goal_text(ex, include_result)


def struct_feats(ex, clean_result_scalars: bool = False) -> np.ndarray:
    """Positional one-hot of last 4 tools + 4 scalars (sota_v2 ``feats``)."""
    st = list(ex.steps)[-4:]
    v = np.zeros(4 * _K + 4, dtype=np.float32)
    for j in range(4):
        k = len(st) - 1 - j
        if k >= 0:
            v[j * _K + _L.get(st[k][0], len(_LABELS))] = 1
        else:
            v[j * _K + len(_LABELS)] = 1
    v[4 * _K] = len(st)
    v[4 * _K + 1] = np.log1p(len(str(st[-1][1]))) if st else 0
    v[4 * _K + 2] = float(any("error" in str(r).lower() for _, r in st[-1:]))
    v[4 * _K + 3] = np.log1p(len(ex.goal or ""))
    if clean_result_scalars:
        v[4 * _K + 1] = 0
        v[4 * _K + 2] = 0
    return v


def shared_scalars(ex, include_result_scalars: bool = True) -> list[float]:
    return struct_feats(ex, clean_result_scalars=not include_result_scalars)[4 * _K :].tolist()
