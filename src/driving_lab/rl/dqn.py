from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import random
import os
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

# Keep the optional RL stack from oversubscribing the same MKL runtime used by
# the NumPy behavior-cloning baseline when pytest imports both stacks.
torch.set_num_threads(1)

from .replay import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    input_dim: int = 11
    action_dim: int = 5
    hidden_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 10000
    device: str = "auto"


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, action_dim))

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        return self.layers(x)


class DQNAgent:
    def __init__(self, config: DQNConfig | None = None):
        self.config = config or DQNConfig()
        requested = self.config.device
        self.device = torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")
        self.network = QNetwork(self.config.input_dim, self.config.action_dim, self.config.hidden_size).to(self.device)
        self.target = QNetwork(self.config.input_dim, self.config.action_dim, self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.replay_capacity)

    def act(self, state, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.config.action_dim)
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
            return int(self.network(tensor).argmax(dim=1).item())

    def train_step(self) -> float | None:
        if len(self.replay) < 1:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        q = self.network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target = rewards + self.config.gamma * (1.0 - dones) * self.target(next_states).max(dim=1).values
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        torch.save({"config": asdict(self.config), "network": self.network.state_dict(), "target": self.target.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto"):
        payload = torch.load(path, map_location="cpu")
        config = DQNConfig(**{**payload["config"], "device": device})
        agent = cls(config)
        agent.network.load_state_dict(payload["network"]); agent.target.load_state_dict(payload.get("target", payload["network"]))
        return agent
