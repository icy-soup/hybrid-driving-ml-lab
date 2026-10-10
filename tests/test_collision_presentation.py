from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.environment import HighwayEnv  # noqa: E402
from driving_lab.events import classify_collision  # noqa: E402
from driving_lab.perception import VehicleGeometry  # noqa: E402
from driving_lab.pygame_app import build_snapshot  # noqa: E402
from driving_lab.types import Action, ActionType, VehicleState  # noqa: E402


def test_ego_keeps_legacy_sports_car_identity():
    env = HighwayEnv()
    env.reset(seed=0, scenario="empty")

    assert env.ego.vehicle_type == "跑车"
    assert env.ego.color == "银色"
    assert build_snapshot(env, last_action=4, total_reward=0.0)["ego"]["vehicle_type"] == "跑车"


def test_traffic_collision_stays_in_environment_for_presentation():
    env = HighwayEnv()
    env.reset(seed=0, scenario="slow_lead")
    env.vehicles = [
        VehicleState("traffic-a", 0, 5.0, 0.0),
        VehicleState("traffic-b", 0, 0.0, 0.0),
    ]
    env._previous_vehicle_x = {vehicle.vehicle_id: vehicle.x for vehicle in env.vehicles}

    transition = env.step(Action(ActionType.CRUISE))

    assert transition.info["collision_events"]
    crashed_ids = {vehicle.vehicle_id for vehicle in env.vehicles if vehicle.crashed}
    assert {"traffic-a", "traffic-b"} <= crashed_ids
    assert all(vehicle.explosion_frames > 0 for vehicle in env.vehicles if vehicle.crashed)

    before = max(vehicle.explosion_frames for vehicle in env.vehicles if vehicle.crashed)
    env.step(Action(ActionType.CRUISE))
    after = max(vehicle.explosion_frames for vehicle in env.vehicles if vehicle.crashed)
    assert 0 < after < before


def test_collision_snapshot_contains_explosion_state_and_participants():
    env = HighwayEnv()
    env.reset(seed=0, scenario="empty")
    env.vehicles = [VehicleState("traffic-a", 0, 0.0, 0.0, crashed=True, explosion_frames=12)]

    snapshot = build_snapshot(env, last_action=4, total_reward=0.0)

    vehicle = snapshot["vehicles"][0]
    assert vehicle["crashed"] is True
    assert vehicle["exploding"] is True
    assert vehicle["explosion_frames"] == 12


def test_collision_classification_is_invariant_to_argument_order():
    rear = VehicleGeometry("rear", 0.0, 50.0, 1, 10.0, 4.0, 12.0)
    front = VehicleGeometry("front", 8.0, 50.0, 1, 10.0, 4.0, 8.0)

    first = classify_collision(rear, front, step=1, scenario="test")
    second = classify_collision(front, rear, step=1, scenario="test")

    assert first.collision_type == second.collision_type == "rear_end"
    assert first.gap == second.gap
    assert first.relative_speed == second.relative_speed == 4.0
