"""Vectorized conv NCA and a flat baseline with the same readout width."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _perception_filters(channels: int) -> torch.Tensor:
    """Fixed identity + Sobel filters, one set per channel. Shape (3C, 1, 3, 3)."""
    ident = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    ) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.float32,
    ) / 8.0
    weight = torch.zeros(channels * 3, 1, 3, 3, dtype=torch.float32)
    for c in range(channels):
        weight[c, 0] = ident
        weight[channels + c, 0] = sobel_x
        weight[2 * channels + c, 0] = sobel_y
    return weight


class TextNCA(nn.Module):
    """Shared local update. Observation channels stay fixed; hidden channels move."""

    def __init__(
        self,
        obs_c: int,
        hidden_c: int,
        n_classes: int,
        steps: int,
        grid_h: int,
        grid_w: int,
        update_hidden: int = 48,
    ) -> None:
        super().__init__()
        self.obs_c = obs_c
        self.hidden_c = hidden_c
        self.channels = obs_c + hidden_c
        self.steps = steps
        self.register_buffer("filters", _perception_filters(self.channels))
        self.w1 = nn.Conv2d(self.channels * 3, update_hidden, kernel_size=1)
        self.w2 = nn.Conv2d(update_hidden, self.channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.w2.weight)
        # Every hidden cell, not only the last row. The observation channels
        # stay out of the head so the score has to come through the update.
        self.head = nn.Linear(hidden_c * grid_h * grid_w, n_classes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_c, H, W) float32, already on device
        b, _, h, w = obs.shape
        hidden = obs.new_zeros(b, self.hidden_c, h, w)
        x = torch.cat((obs, hidden), dim=1)
        for _ in range(self.steps):
            perceived = F.conv2d(x, self.filters, padding=1, groups=self.channels)
            delta = self.w2(F.relu(self.w1(perceived)))
            hidden_delta = delta[:, self.obs_c :]
            x = torch.cat((obs, x[:, self.obs_c :] + hidden_delta), dim=1)
        hidden = x[:, self.obs_c :]
        return self.head(hidden.flatten(1))


class FlatPolicy(nn.Module):
    """Same observation tensor, no spatial update. Hypothesis-4 baseline."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs.flatten(1))
