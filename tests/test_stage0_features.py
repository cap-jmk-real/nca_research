"""features.py must match legacy sota_v2 / sota_gbdt strings and struct vectors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage0"))

from nca_text.data import Example
from features import goal_text, struct_feats

LABELS = json.load(open(ROOT / "stage0" / "labels.json"))
L = {t: i for i, t in enumerate(LABELS)}
K = len(LABELS) + 1


def legacy_text(ex, res: bool) -> str:
    return (ex.goal or "") + (
        (" || " + " ".join(str(r)[:200] for _, r in list(ex.steps)[-1:])) if res else ""
    )


def legacy_feats(ex) -> np.ndarray:
    st = list(ex.steps)[-4:]
    v = np.zeros(4 * K + 4, dtype=np.float32)
    for j in range(4):
        k = len(st) - 1 - j
        if k >= 0:
            v[j * K + L.get(st[k][0], len(LABELS))] = 1
        else:
            v[j * K + len(LABELS)] = 1
    v[4 * K] = len(st)
    v[4 * K + 1] = np.log1p(len(str(st[-1][1]))) if st else 0
    v[4 * K + 2] = float(any("error" in str(r).lower() for _, r in st[-1:]))
    v[4 * K + 3] = np.log1p(len(ex.goal or ""))
    return v


FIXTURES = [
    Example("s1", "terminal", "Run tests", [("read_file", "ok")]),
    Example(
        "s2",
        "search_files",
        "Find config",
        [("terminal", '{"error": "fail"}'), ("todo", "x" * 250)],
    ),
    Example("s3", "respond", "", []),
]


@pytest.mark.parametrize("ex", FIXTURES)
def test_goal_text_matches_legacy(ex: Example) -> None:
    assert goal_text(ex, True) == legacy_text(ex, True)
    assert goal_text(ex, False) == legacy_text(ex, False)


@pytest.mark.parametrize("ex", FIXTURES)
def test_struct_feats_matches_legacy(ex: Example) -> None:
    np.testing.assert_array_equal(struct_feats(ex), legacy_feats(ex))


def test_noresult_clean_zeros_result_scalars() -> None:
    ex = FIXTURES[1]
    v = struct_feats(ex, clean_result_scalars=True)
    assert v[4 * K + 1] == 0
    assert v[4 * K + 2] == 0
