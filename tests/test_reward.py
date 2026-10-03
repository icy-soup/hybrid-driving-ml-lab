import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.config import EnvironmentConfig  # noqa: E402
from driving_lab.environment import HighwayEnv  # noqa: E402
from driving_lab.rewards import RewardConfig, compute_reward  # noqa: E402
from driving_lab.types import Action, ActionType, VehicleState  # noqa: E402


def test_reward_breakdown_exposes_each_component():
    breakdown = compute_reward(
        RewardConfig(),
        speed=12.0,
        ttc=1.5,
        overtake=True,
        lane_change=True,
        collision=False,
        shield_override=True,
    )

    assert breakdown.components["alive"] > 0
    assert breakdown.components["progress"] > 0
    assert breakdown.components["overtake"] > 0
    assert breakdown.components["ttc_warning"] < 0
    assert breakdown.components["lane_change"] < 0
    assert breakdown.components["shield_override"] < 0
    assert breakdown.total == sum(breakdown.components.values())


def test_collision_ends_episode_and_applies_terminal_penalty():
    env = HighwayEnv(EnvironmentConfig(max_steps=100))
    env.reset(seed=1, scenario="obstacle")

    transition = None
    for _ in range(100):
        transition = env.step(Action(ActionType.BRAKE))
        if transition.done:
            break

    assert transition is not None
    assert transition.done is True
    assert transition.info["collision"] is True
    assert transition.info["reward_components"]["collision"] < 0


def test_overtake_event_is_counted_once_without_collision():
    env = HighwayEnv(EnvironmentConfig(max_steps=20))
    env.reset(seed=1, scenario="empty")
    env.vehicles = [VehicleState("passed", 0, 4.0, 4.0)]

    first = env.step(Action(ActionType.CRUISE))
    second = env.step(Action(ActionType.CRUISE))

    assert first.info["events"]["overtakes"] == 1
    assert second.info["events"]["overtakes"] == 1
    assert first.info["collision"] is False


def test_shield_override_is_visible_in_reward_components():
    env = HighwayEnv(EnvironmentConfig(max_steps=20))
    env.reset(seed=1, scenario="empty")

    transition = env.step(Action(ActionType.BRAKE), shield_override=True)

    assert transition.info["shield_override"] is True
    assert transition.info["reward_components"]["shield_override"] < 0
