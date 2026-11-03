"""Export learned GraphNCA meta adjacency (alpha / row-softmax A) and directed diameters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def directed_diameter(A: np.ndarray, eps: float = 1e-9) -> int:
    """Shortest-path hop count; edge j→i when A[i,j]>eps."""
    n = A.shape[0]
    dist = np.full((n, n), 10**9, dtype=np.int64)
    for i in range(n):
        dist[i, i] = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] > eps:
                dist[i, j] = 1
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[k, None, :])
    d = int(dist.max())
    return d if d < 10**8 else -1


def eccentricity_from_node(A: np.ndarray, source: int = 0, eps: float = 1e-9) -> int:
    n = A.shape[0]
    dist = [10**9] * n
    dist[source] = 0
    for _ in range(n):
        changed = False
        for i in range(n):
            for j in range(n):
                if A[i, j] > eps and dist[j] + 1 < dist[i]:
                    dist[i] = dist[j] + 1
                    changed = True
        if not changed:
            break
    return int(max(dist))


def adjacency_from_state(state: dict, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha logits, row-softmax A) from a GraphNCA state dict."""
    import torch
    import torch.nn.functional as F

    alpha_t = state.get("alpha")
    if alpha_t is None:
        raise KeyError("state dict has no 'alpha' (not a meta topology checkpoint)")
    if not isinstance(alpha_t, torch.Tensor):
        alpha_t = torch.tensor(alpha_t)
    alpha = alpha_t.detach().cpu().numpy().astype(np.float64)
    A = F.softmax(alpha_t / temperature, dim=-1).detach().cpu().numpy().astype(np.float64)
    return alpha, A


def meta_export_record(state: dict, *, seed: int, checkpoint: str, temperature: float = 1.0) -> dict:
    alpha, A = adjacency_from_state(state, temperature)
    return {
        "seed": seed,
        "checkpoint": checkpoint,
        "temperature": temperature,
        "alpha": alpha.round(6).tolist(),
        "adjacency_row_softmax": A.round(6).tolist(),
        "directed_diameter": directed_diameter(A),
        "eccentricity_from_goal_node_0": eccentricity_from_node(A, 0),
    }
