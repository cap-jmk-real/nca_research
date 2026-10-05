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

    Hidden channels update for ``steps`` iterations. By default they start at
    zero each call; pass ``hidden`` / ``return_hidden=True`` to carry them
    across decision rows (research-design hypothesis 2).

    Optional ``learned_halt`` soft-stops further updates (Adaptive Message
    Passing analogue). Optional ``long_range_goal`` adds a shortcut from the
    goal cell into every position (rewiring / BraiNCA-style long edge).
    """

    def __init__(
        self,
        obs_c: int,
        hidden_c: int,
        n_classes: int,
        steps: int,
        seq_len: int,
        update_hidden: int = 96,
        learned_halt: bool = False,
        long_range_goal: bool = False,
    ) -> None:
        super().__init__()
        self.obs_c = obs_c
        self.hidden_c = hidden_c
        self.channels = obs_c + hidden_c
        self.steps = steps
        self.seq_len = seq_len
        self.learned_halt = learned_halt
        self.long_range_goal = long_range_goal
        self.register_buffer("filters", _perception_filters(self.channels))
        self.step_embed = nn.Parameter(torch.zeros(steps, hidden_c))
        perc_in = self.channels * 3
        if long_range_goal:
            # Broadcast goal cell (obs + hidden) into every position.
            perc_in = perc_in + self.channels
            self.goal_scale = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Conv1d(perc_in, update_hidden, kernel_size=1)
        # delta and gate, hidden channels only
        self.w2 = nn.Conv1d(update_hidden, hidden_c * 2, kernel_size=1, bias=False)
        nn.init.zeros_(self.w2.weight)
        if learned_halt:
            self.halt = nn.Linear(hidden_c, 1)
            nn.init.zeros_(self.halt.weight)
            nn.init.constant_(self.halt.bias, 1.0)  # start near "keep going"
        self.head = nn.Linear(hidden_c * seq_len, n_classes)

    def _perceive(self, obs: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat((obs, h), dim=1)
        perceived = F.conv1d(x, self.filters, padding=1, groups=self.channels)
        if not self.long_range_goal:
            return perceived
        goal = torch.cat((obs[:, :, 0], h[:, :, 0]), dim=1).unsqueeze(-1)
        goal = goal.expand(-1, -1, obs.shape[-1])
        # Zero-init scale so the first forward matches the local-only rule.
        return torch.cat((perceived, torch.tanh(self.goal_scale) * goal), dim=1)

    def evolve(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Run the cellular rule; return the final hidden state."""
        b, _, length = obs.shape
        if hidden is None:
            hidden = obs.new_zeros(b, self.hidden_c, length)
        else:
            hidden = hidden
        n_steps = self.steps if steps is None else int(steps)
        alive = obs.new_ones(b)
        for t in range(n_steps):
            embed = self.step_embed[min(t, self.steps - 1)].view(1, -1, 1)
            h = hidden + embed
            perceived = self._perceive(obs, h)
            update = self.w2(F.relu(self.w1(perceived)))
            delta, gate_logit = update.chunk(2, dim=1)
            residual = torch.sigmoid(gate_logit) * delta
            hidden = hidden + alive.view(b, 1, 1) * residual
            if self.learned_halt and t + 1 < n_steps:
                halt_logit = self.halt(hidden.mean(dim=-1))
                halt_p = torch.sigmoid(halt_logit).squeeze(-1)
                alive = alive * (1.0 - halt_p)
        return hidden

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
        steps: int | None = None,
    ):
        # obs: (B, obs_c, L)
        state = self.evolve(obs, hidden=hidden, steps=steps)
        logits = self.head(state.flatten(1))
        if return_hidden:
            return logits, state
        return logits


class FlatPolicy(nn.Module):
    """Same observation tensor, no cellular update. Hypothesis-4 baseline."""

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs.flatten(1))
