from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.rl import DQNAgent, DQNConfig, ReplayBuffer, QNetwork


def test_replay_buffer_capacity_and_q_network_shape():
    buffer = ReplayBuffer(2)
    for index in range(3):
        buffer.add(np.zeros(11), index % 5, 1.0, np.ones(11), False)
    assert len(buffer) == 2
    network = QNetwork(11, 5, hidden_size=16)
    assert tuple(network(np.zeros((1, 11), dtype=np.float32)).shape) == (1, 5)


def test_agent_checkpoint_round_trip(tmp_path):
    agent = DQNAgent(DQNConfig(input_dim=11, action_dim=5, device="cpu"))
    path = tmp_path / "dqn.pt"
    agent.save(path)
    restored = DQNAgent.load(path, device="cpu")
    assert restored.config.action_dim == 5
