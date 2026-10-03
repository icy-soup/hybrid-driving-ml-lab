import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.config import EnvironmentConfig  # noqa: E402
from driving_lab.environment import HighwayEnv  # noqa: E402
from driving_lab.types import Action, ActionType  # noqa: E402


def _vehicle_snapshot(env):
    return [(v.vehicle_id, v.lane, round(v.x, 5), round(v.speed, 5)) for v in env.vehicles]


def test_reset_with_same_seed_recreates_observation_and_traffic():
    env = HighwayEnv(EnvironmentConfig(max_steps=20))
    first = env.reset(seed=7, scenario="random")
    first_traffic = _vehicle_snapshot(env)
    env.step(Action(ActionType.CRUISE))

    second = env.reset(seed=7, scenario="random")

    assert first.to_features().tolist() == second.to_features().tolist()
    assert first_traffic == _vehicle_snapshot(env)


def test_lane_actions_are_clamped_at_road_edges():
    env = HighwayEnv(EnvironmentConfig(max_steps=20))
    env.reset(seed=1, scenario="empty")

    env.ego = replace(env.ego, lane=0)
    env.step(Action(ActionType.LANE_LEFT))
    assert env.ego.lane == 0

    env.ego = replace(env.ego, lane=2)
    env.step(Action(ActionType.LANE_RIGHT))
    assert env.ego.lane == 2


def test_slow_lead_scenario_exposes_a_front_vehicle():
    env = HighwayEnv(EnvironmentConfig(max_steps=20))
    observation = env.reset(seed=3, scenario="slow_lead")

    assert observation.front_distance[observation.lane] < float("inf")
    assert observation.front_speed[observation.lane] is not None
    assert observation.front_speed[observation.lane] > observation.speed


def test_rear_approach_scenario_exposes_a_nearby_rear_vehicle():
    env = HighwayEnv(EnvironmentConfig(max_steps=20))
    observation = env.reset(seed=3, scenario="rear_approach")

    lane = observation.lane
    assert any(distance < 160.0 for distance in observation.rear_distance)
    assert observation.rear_distance[lane] < float("inf")


def test_environment_module_does_not_import_pygame():
    source = Path(__file__).parents[1] / "src" / "driving_lab" / "environment.py"
    assert "import pygame" not in source.read_text(encoding="utf-8")
