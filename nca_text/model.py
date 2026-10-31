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


def build_adjacency(topology: str, seq_len: int = 5) -> torch.Tensor:
    """Constructs row-stochastic adjacency matrix for candidate spatial topologies."""
    A = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    if topology == "1d":
        for i in range(seq_len):
            A[i, i] = 1.0
            if i > 0:
                A[i, i - 1] = 1.0
            if i < seq_len - 1:
                A[i, i + 1] = 1.0
    elif topology == "star":
        # Node 0 (goal) is connected to all nodes; all nodes connect to 0 and self.
        for i in range(seq_len):
            A[i, i] = 1.0
            A[0, i] = 1.0
            A[i, 0] = 1.0
    elif topology == "tree":
        # Binary hierarchical call-tree DAG on seq_len nodes:
        # Parent of i > 0 is (i - 1) // 2
        for i in range(seq_len):
            A[i, i] = 1.0
            if i > 0:
                parent = (i - 1) // 2
                A[parent, i] = 1.0
                A[i, parent] = 1.0
    elif topology == "complete":
        A.fill_(1.0)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Row normalize so each row sums to 1.0
    A = A / A.sum(dim=-1, keepdim=True)
    return A


class GraphNCA(nn.Module):
    """Graph Neural Cellular Automata with multi-topology support, architecture meta-learning,
    and configurable deep recurrence / rule depth.
    
    Topologies supported:
      - '1d': 1D nearest-neighbor lattice
      - 'star': Goal-broadcasting star topology
      - 'tree': Hierarchical call-tree DAG routing
      - 'meta': Continuous bilevel architecture meta-learning over edge weights alpha
    """
    def __init__(
        self,
        obs_c: int,
        hidden_c: int,
        n_classes: int,
        steps: int = 4,
        rule_depth: int = 2,
        seq_len: int = 5,
        update_hidden: int = 96,
        topology: str = "tree",
        learned_halt: bool = False,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.obs_c = obs_c
        self.hidden_c = hidden_c
        self.channels = obs_c + hidden_c
        self.steps = steps
        self.rule_depth = rule_depth
        self.seq_len = seq_len
        self.topology = topology
        self.learned_halt = learned_halt
        self.temperature = temperature
        
        if topology == "meta":
            # Learnable continuous edge logits alpha in R^{V x V}
            init_adj = build_adjacency("tree", seq_len)
            self.alpha = nn.Parameter(torch.log(init_adj + 1e-4))
        else:
            self.register_buffer("fixed_adj", build_adjacency(topology, seq_len))
            
        self.step_embed = nn.Parameter(torch.zeros(steps, hidden_c))
        perc_in = self.channels * 3
        
        # Local cell update network with configurable rule_depth
        if rule_depth <= 2:
            self.w1 = nn.Conv1d(perc_in, update_hidden, kernel_size=1)
            self.w2 = nn.Conv1d(update_hidden, hidden_c * 2, kernel_size=1, bias=False)
            nn.init.zeros_(self.w2.weight)
            self.deep_layers = None
        else:
            # Deep update network with residual pre-activation and group normalization
            self.w1 = nn.Conv1d(perc_in, update_hidden, kernel_size=1)
            layers = []
            for _ in range(rule_depth - 2):
                layers.append(nn.Sequential(
                    nn.GroupNorm(1, update_hidden),
                    nn.GELU(),
                    nn.Conv1d(update_hidden, update_hidden, kernel_size=1),
                ))
            self.deep_layers = nn.ModuleList(layers)
            self.w2 = nn.Conv1d(update_hidden, hidden_c * 2, kernel_size=1, bias=False)
            nn.init.zeros_(self.w2.weight)
        
        if learned_halt:
            self.halt = nn.Linear(hidden_c, 1)
            nn.init.zeros_(self.halt.weight)
            nn.init.constant_(self.halt.bias, 1.0)
            
        # Hierarchical multi-head readout
        self.tool_head = nn.Linear(hidden_c * seq_len, n_classes)
        self.arg_proj = nn.Bilinear(hidden_c, hidden_c, 32)
        self.abstain_head = nn.Linear(hidden_c, 1)

    def get_adjacency(self) -> torch.Tensor:
        if self.topology == "meta":
            return F.softmax(self.alpha / self.temperature, dim=-1)
        return self.fixed_adj

    def _perceive(self, obs: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_c, V), h: (B, hidden_c, V)
        x = torch.cat((obs, h), dim=1) # (B, C, V)
        A = self.get_adjacency() # (V, V)
        x_perm = x.permute(0, 2, 1) # (B, V, C)
        agg = torch.matmul(A, x_perm).permute(0, 2, 1) # (B, C, V)
        laplacian = agg - x
        perceived = torch.cat((x, agg, laplacian), dim=1) # (B, 3C, V)
        return perceived

    def evolve(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, _, length = obs.shape
        if hidden is None:
            hidden = obs.new_zeros(b, self.hidden_c, length)
        n_steps = self.steps if steps is None else int(steps)
        alive = obs.new_ones(b)
        trajectory = [hidden]
        for t in range(n_steps):
            embed = self.step_embed[min(t, self.steps - 1)].view(1, -1, 1)
            h = hidden + embed
            perceived = self._perceive(obs, h)
            feat = F.relu(self.w1(perceived))
            if self.deep_layers is not None:
                for block in self.deep_layers:
                    feat = feat + block(feat)
            update = self.w2(feat)
            delta, gate_logit = update.chunk(2, dim=1)
            residual = torch.sigmoid(gate_logit) * delta
            hidden = hidden + alive.view(b, 1, 1) * residual
            trajectory.append(hidden)
            if self.learned_halt and t + 1 < n_steps:
                halt_logit = self.halt(hidden.mean(dim=-1))
                halt_p = torch.sigmoid(halt_logit).squeeze(-1)
                alive = alive * (1.0 - halt_p)
        return hidden, trajectory

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
        steps: int | None = None,
    ):
        state, traj = self.evolve(obs, hidden=hidden, steps=steps)
        logits = self.tool_head(state.flatten(1))
        if return_hidden:
            return logits, state
        return logits
