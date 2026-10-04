from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.environment import HighwayEnv
from driving_lab.policies import RulePolicy
from driving_lab.types import Action, ActionType, BehaviorMode


def test_environment_traffic_enters_follow_or_brake_mode():
    env = HighwayEnv()
    observation = env.reset(seed=5, scenario="slow_lead")
    seen = set()
    for _ in range(80):
        transition = env.step(Action(ActionType.CRUISE))
        observation = transition.next_observation
        seen.update(vehicle.mode for vehicle in env.vehicles)
        if transition.done:
            break
    assert seen & {BehaviorMode.FOLLOW, BehaviorMode.BRAKE, BehaviorMode.EMERGENCY_BRAKE}


def test_rear_approach_traffic_does_not_remain_passive_cruise_only():
    env = HighwayEnv()
    observation = env.reset(seed=6, scenario="rear_approach")
    modes = []
    for _ in range(35):
        transition = env.step(Action(ActionType.CRUISE))
        observation = transition.next_observation
        modes.append(env.vehicles[0].mode)
        if transition.done:
            break
    assert any(mode is not BehaviorMode.CRUISE for mode in modes)
