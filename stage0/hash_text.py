"""Wide hash bag for ``--text_feat hash_wide`` (goal cell only)."""

from __future__ import annotations

import re
import zlib

import numpy as np

_WS = re.compile(r"\s+")
HASH_WIDE_DIM = 512


def hash_wide(text: str, dim: int = HASH_WIDE_DIM, max_chars: int = 200) -> np.ndarray:
    vec = np.zeros(dim, np.float32)
    raw = (text or "").lower()[:max_chars]
    toks = [t for t in _WS.split(raw.strip()) if t]
    if not toks and text:
        toks = [text[:32].lower()]
    items = list(toks) + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    for tok in items:
        digest = zlib.crc32(tok.encode("utf-8", "ignore")) & 0xFFFFFFFF
        vec[digest % dim] += 1.0
    norm = float(np.sqrt(np.dot(vec, vec)))
    if norm > 0:
        vec /= norm
    return vec
