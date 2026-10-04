from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.controllers.mpc import MPCConfig, MPCPolicy
from driving_lab.types import Action, ActionType, Observation


def _observation(front=100.0, ttc=10.0, speed=12.0):
    return Observation(speed=-speed, lane=1, front_distance=(100.0, front, 100.0), front_speed=(None, 6.0, None), ttc=(10.0, ttc, 10.0))


def test_mpc_brakes_for_immediate_collision_risk():
    policy = MPCPolicy(config=MPCConfig(horizon=3))
    action = policy.act(_observation(front=8.0, ttc=0.5, speed=14.0))
    assert action.kind is ActionType.BRAKE
    assert policy.last_plan


def test_mpc_action_is_deterministic_and_horizon_one_has_plan():
    policy = MPCPolicy(config=MPCConfig(horizon=1))
    first = policy.act(_observation())
    second = policy.act(_observation())
    assert first == second
    assert len(policy.last_plan) == 1
