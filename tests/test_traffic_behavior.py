from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.config import EnvironmentConfig
from driving_lab.perception import VehicleGeometry, perceive
from driving_lab.traffic import TrafficDriver, TrafficProfile, TrafficWorld
from driving_lab.types import BehaviorMode, VehicleState


def _perception(speed=10.0, front=None, left_front=None, left_rear=None, right_front=None):
    ego = VehicleGeometry("ego", 0.0, 50.0, 1, 10.0, 4.0, speed)
    vehicles = []
    if front:
        vehicles.append(VehicleGeometry("front", front, 50.0, 1, 10.0, 4.0, 6.0))
    if left_front:
        vehicles.append(VehicleGeometry("lf", left_front, 0.0, 0, 10.0, 4.0, 10.0))
    if left_rear:
        vehicles.append(VehicleGeometry("lr", left_rear, 0.0, 0, 10.0, 4.0, 14.0))
    if right_front:
        vehicles.append(VehicleGeometry("rf", right_front, 100.0, 2, 10.0, 4.0, 10.0))
    return perceive(ego, vehicles, EnvironmentConfig()), ego


def test_driver_cruises_at_profile_speed_when_lane_is_clear():
    perception, vehicle = _perception(speed=8.0)
    decision = TrafficDriver(TrafficProfile(desired_speed=12.0)).decide(vehicle, perception, 0.1)
    assert decision.target_speed == pytest.approx(12.0)
    assert decision.behavior_mode is BehaviorMode.CRUISE


def test_driver_follows_and_brakes_for_low_ttc_vehicle():
    perception, vehicle = _perception(speed=12.0, front=12.0)
    decision = TrafficDriver(TrafficProfile(desired_speed=14.0)).decide(vehicle, perception, 0.1)
    assert decision.target_speed < vehicle.speed
    assert decision.behavior_mode in (BehaviorMode.FOLLOW, BehaviorMode.BRAKE, BehaviorMode.EMERGENCY_BRAKE)


def test_emergency_brake_matches_front_speed_instead_of_gap_speed():
    perception, vehicle = _perception(speed=12.0, front=18.0)
    # The front vehicle travels at 6 px/frame.  A short gap must command a
    # target close to that speed; using gap/headway here would command 12 and
    # keep closing into the obstacle.
    decision = TrafficDriver(TrafficProfile(desired_speed=14.0)).decide(vehicle, perception, 0.1)
    assert decision.behavior_mode is BehaviorMode.EMERGENCY_BRAKE
    assert decision.target_speed <= 6.0


def test_driver_can_prepare_lane_change_only_when_front_and_rear_windows_are_clear():
    perception, vehicle = _perception(speed=12.0, front=35.0)
    decision = TrafficDriver(TrafficProfile(desired_speed=14.0, lane_change_threshold=40.0)).decide(vehicle, perception, 0.1)
    assert decision.target_lane == 0
    assert decision.behavior_mode is BehaviorMode.PREPARE_LANE_CHANGE

    blocked, vehicle = _perception(speed=12.0, front=35.0, left_rear=-30.0)
    blocked_decision = TrafficDriver(TrafficProfile(desired_speed=14.0, lane_change_threshold=40.0)).decide(vehicle, blocked, 0.1)
    assert blocked_decision.target_lane != 0
    assert blocked_decision.target_lane in (vehicle.lane, 2)


def test_traffic_world_does_not_integrate_same_lane_vehicles_into_penetration():
    world = TrafficWorld(EnvironmentConfig(traffic_min_gap=20.0))
    ego = VehicleState("ego", lane=1, x=0.0, speed=10.0, y=50.0)
    vehicles = [
        VehicleState("front", lane=1, x=30.0, speed=10.0, y=50.0),
        VehicleState("rear", lane=1, x=15.0, speed=20.0, y=50.0),
    ]
    result = world.step_traffic(vehicles, ego, time_step=0.5)
    front, rear = result.vehicles
    assert front.x - rear.x >= 20.0 + (front.length + rear.length) / 2.0 - 1e-9


def test_traffic_world_performs_continuous_lane_change_in_fixed_scenario():
    world = TrafficWorld(EnvironmentConfig())
    ego = VehicleState("ego", lane=1, x=0.0, speed=8.0, y=50.0)
    vehicles = [
        VehicleState("lead", lane=1, x=100.0, speed=4.0, y=50.0),
        VehicleState("follower", lane=1, x=35.0, speed=12.0, y=50.0),
    ]

    result = world.step_traffic(vehicles, ego, scenario="slow_lead", time_step=0.25)
    follower = next(vehicle for vehicle in result.vehicles if vehicle.vehicle_id == "follower")

    assert result.decisions["follower"].target_lane == 0
    assert follower.y == pytest.approx(49.25)
    assert follower.mode is BehaviorMode.LANE_CHANGE


def test_lane_change_safety_window_scales_with_current_speed():
    profile = TrafficProfile(desired_speed=30.0, lane_change_threshold=40.0)
    driver = TrafficDriver(profile)

    low_perception, low_vehicle = _perception(speed=8.0, front=35.0, left_front=150.0, right_front=150.0)
    high_perception, high_vehicle = _perception(speed=30.0, front=35.0, left_front=150.0, right_front=150.0)

    low_decision = driver.decide(low_vehicle, low_perception, 0.1)
    high_decision = driver.decide(high_vehicle, high_perception, 0.1)

    assert low_decision.target_lane != low_vehicle.lane
    assert high_decision.target_lane == high_vehicle.lane
