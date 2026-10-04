from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.rl import DQNAgent, DQNConfig, DQNTrainer, SafetyShieldPolicy, initialize_from_behavior_cloning
from driving_lab.types import Action, ActionType, Observation


def test_cpu_smoke_training_and_behavior_initialization():
    agent = DQNAgent(DQNConfig(input_dim=11, action_dim=5, device="cpu", batch_size=2, replay_capacity=8))
    trainer = DQNTrainer(agent, seed=2)
    metrics = trainer.train_steps(4)
    assert metrics["steps"] == 4
    initialize_from_behavior_cloning(agent, agent.network.state_dict())


def test_shield_records_override():
    class Unsafe:
        def act(self, observation):
            return Action(ActionType.ACCELERATE)

    shield = SafetyShieldPolicy(Unsafe())
    observation = Observation( speed=-14.0, lane=1, front_distance=(100.0, 5.0, 100.0), front_speed=(None, -2.0, None), ttc=(10.0, 0.5, 10.0))
    assert shield.act(observation).kind is ActionType.BRAKE
    assert shield.last_override is True
