"""Exact structural properties of the implemented GraphNCA (nca_text/model.py), checked numerically.

P1  Weight tying: the parameter count grows by exactly H per extra step (the step embedding e_n);
    an untied K-layer GNN with the same operator grows by a full update network per step.
P2  Automorphism equivariance: for every permutation P with P A P^T = A, the hidden trajectory
    satisfies evolve(P o) = P evolve(o). On the tree the only non-trivial automorphism swaps the
    two most recent steps (cells 3 and 4).
P3  Locality: h_i^(k) does not depend on the observation of cell j when dist_A(i, j) > k.
Run: python -m pytest tests/test_gnca_properties_1110.py  or  python tests/test_gnca_properties_1110.py  (CPU, < 10 s)
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nca_text.model import GraphNCA, build_adjacency  # noqa: E402

OBS_C, H, NCLS, V = 146, 16, 13, 5


def _net(topology="tree", K=4, seed=0):
    torch.manual_seed(seed)
    net = GraphNCA(obs_c=OBS_C, hidden_c=H, n_classes=NCLS, steps=K, seq_len=V, topology=topology).double()
    with torch.no_grad():  # W2 is zero-initialised (identity map); randomise so the test is not trivial
        net.w2.weight.normal_(0, 0.3)
        net.step_embed.normal_(0, 0.3)
    return net.eval()


def _hops(A):
    S = (A > 0).int()
    D = torch.full((V, V), 99)
    for i in range(V):
        D[i, i] = 0
    R = torch.eye(V, dtype=torch.int)
    for k in range(1, V):
        R = ((R @ S) > 0).int()
        D = torch.where((R > 0) & (D == 99), torch.full_like(D, k), D)
    return D


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_p1_weight_tying_parameter_count():
    base = n_params(_net(K=2))
    for K in (3, 4, 6, 8):
        assert n_params(_net(K=K)) - base == (K - 2) * H


def test_p2_automorphism_equivariance():
    for topo in ("tree", "star", "1d"):
        A = build_adjacency(topo, V).double()
        auts = []
        for perm in itertools.permutations(range(V)):
            P = torch.eye(V, dtype=torch.double)[list(perm)]
            if torch.allclose(P @ A @ P.T, A):
                auts.append(list(perm))
        if topo == "tree":
            assert sorted(auts) == [[0, 1, 2, 3, 4], [0, 1, 2, 4, 3]]
        net = _net(topo)
        o = torch.randn(3, OBS_C, V, dtype=torch.double)
        h, _ = net.evolve(o)
        for perm in auts:
            hp, _ = net.evolve(o[:, :, perm])
            assert torch.allclose(hp, h[:, :, perm], atol=1e-10)


def test_p3_locality_receptive_field():
    for topo in ("tree", "1d", "star"):
        A = build_adjacency(topo, V)
        D = _hops(A)
        net = _net(topo, K=4)
        o = torch.randn(2, OBS_C, V, dtype=torch.double)
        _, traj = net.evolve(o)
        for j in range(V):
            o2 = o.clone(); o2[:, :, j] += torch.randn(2, OBS_C, dtype=torch.double)
            _, traj2 = net.evolve(o2)
            for k in range(1, 5):
                for i in range(V):
                    changed = not torch.allclose(traj[k][:, :, i], traj2[k][:, :, i], atol=1e-12)
                    # h_i^(k) depends on o_j iff dist(i, j) <= k (the "if" part holds for generic weights)
                    assert changed == bool(D[i, j] <= k), (topo, i, j, k, int(D[i, j]))


def test_p1_untied_parameter_count():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage0"))
    import run_rigour_1110 as RG
    RG._ensure_torch_classes()
    Dm, Dc = 96, OBS_C + H
    n_upd = Dm * (3 * Dc + 1) + 2 * H * Dm
    for K in (2, 4, 8):
        tied = GraphNCA(obs_c=OBS_C, hidden_c=H, n_classes=NCLS, steps=K, seq_len=V, topology="tree")
        untied = RG.UntiedGNN(obs_c=OBS_C, hidden_c=H, n_classes=NCLS, steps=K, seq_len=V, topology="tree")
        assert n_params(untied) - n_params(tied) == (K - 1) * n_upd


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
