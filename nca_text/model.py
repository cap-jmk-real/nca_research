"""1D trace NCA and a flat readout on the same observation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _perception_filters(channels: int) -> torch.Tensor:
    """Fixed 1D identity, gradient, and Laplacian. Shape (3C, 1, 3)."""
    ident = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    grad = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
    lap = torch.tensor([0.25, -0.5, 0.25], dtype=torch.float32)
    weight = torch.zeros(channels * 3, 1, 3, dtype=torch.float32)
    for c in range(channels):
        weight[c, 0] = ident
        weight[channels + c, 0] = grad
        weight[2 * channels + c, 0] = lap
    return weight


class TextNCA(nn.Module):
    """Shared 1D rule along the trace. Observation channels stay fixed.

    Hidden channels are updated in place for ``steps`` iterations and reset
    by the caller between decision rows. The last convolution is zero-initialized
    so the first forward is a no-op residual.
    """

    def __init__(
        self,
        obs_c: int,
        hidden_c: int,
        n_classes: int,
        steps: int,
        seq_len: int,
        update_hidden: int = 96,
    ) -> None:
        super().__init__()
        self.obs_c = obs_c
        self.hidden_c = hidden_c
        self.channels = obs_c + hidden_c
        self.steps = steps
        self.seq_len = seq_len
        self.register_buffer("filters", _perception_filters(self.channels))
        self.step_embed = nn.Parameter(torch.zeros(steps, hidden_c))
        self.w1 = nn.Conv1d(self.channels * 3, update_hidden, kernel_size=1)
        # delta and gate, hidden channels only
        self.w2 = nn.Conv1d(update_hidden, hidden_c * 2, kernel_size=1, bias=False)
        nn.init.zeros_(self.w2.weight)
        self.head = nn.Linear(hidden_c * seq_len, n_classes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_c, L)
        b, _, length = obs.shape
        hidden = obs.new_zeros(b, self.hidden_c, length)
        for t in range(self.steps):
            h = hidden + self.step_embed[t].view(1, -1, 1)
            x = torch.cat((obs, h), dim=1)
            perceived = F.conv1d(x, self.filters, padding=1, groups=self.channels)
            update = self.w2(F.relu(self.w1(perceived)))
            delta, gate_logit = update.chunk(2, dim=1)
            hidden = hidden + torch.sigmoid(gate_logit) * delta
        return self.head(hidden.flatten(1))


class FlatPolicy(nn.Module):
    """Same observation tensor, no cellular update. Hypothesis-4 baseline."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs.flatten(1))
