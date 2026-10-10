import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.config import EnvironmentConfig  # noqa: E402
from driving_lab.environment import HighwayEnv  # noqa: E402
from driving_lab.types import Action, ActionType, VehicleState  # noqa: E402


def _same_lane_gaps(vehicles):
    gaps = []
    for lane in range(3):
        xs = sorted((vehicle.x for vehicle in vehicles if vehicle.lane == lane), reverse=True)
        gaps.extend(front - rear for front, rear in zip(xs, xs[1:]))
    return gaps


def test_random_traffic_starts_with_safe_same_lane_spacing():
    env = HighwayEnv()
    for seed in range(20):
        env.reset(seed=seed, scenario="random")
        assert all(gap >= env.config.traffic_min_gap for gap in _same_lane_gaps(env.vehicles))


def test_following_traffic_is_clamped_instead_of_overlapping():
    env = HighwayEnv(EnvironmentConfig(traffic_min_gap=72.0))
    env.reset(seed=7, scenario="empty")
    env.vehicles = [
        VehicleState("front", 1, 150.0, 8.0),
        VehicleState("rear", 1, 100.0, 16.0),
    ]
    env._previous_vehicle_x = {vehicle.vehicle_id: vehicle.x for vehicle in env.vehicles}

    env.step(Action(ActionType.CRUISE))

    assert _same_lane_gaps(env.vehicles) == [72.0]
    assert next(vehicle for vehicle in env.vehicles if vehicle.vehicle_id == "rear").speed == 8.0
