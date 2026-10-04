from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.config import EnvironmentConfig
from driving_lab.perception import VehicleGeometry, perceive
from driving_lab.types import BehaviorMode


def test_front_distance_is_bumper_to_bumper_and_ttc_uses_closing_speed():
    ego = VehicleGeometry("ego", x=0.0, y=50.0, lane=1, length=10.0, width=4.0, speed=20.0)
    front = VehicleGeometry("front", x=50.0, y=50.0, lane=1, length=10.0, width=4.0, speed=10.0)
    snapshot = perceive(ego, [front], EnvironmentConfig())
    assert snapshot.front_center.vehicle_id == "front"
    assert snapshot.front_center.gap == pytest.approx(40.0)
    assert snapshot.front_center.relative_speed == pytest.approx(10.0)
    assert snapshot.front_center.ttc == pytest.approx(4.0)


def test_five_sensor_regions_and_lane_occupancy_use_lateral_overlap():
    ego = VehicleGeometry("ego", x=0.0, y=50.0, lane=1, length=10.0, width=4.0, speed=10.0)
    vehicles = [
        VehicleGeometry("left-front", 30.0, 0.0, 0, 10.0, 4.0, 10.0),
        VehicleGeometry("right-front", 25.0, 100.0, 2, 10.0, 4.0, 10.0),
        VehicleGeometry("left-rear", -30.0, 0.0, 0, 10.0, 4.0, 10.0),
        VehicleGeometry("right-rear", -25.0, 100.0, 2, 10.0, 4.0, 10.0),
    ]
    snapshot = perceive(ego, vehicles, EnvironmentConfig())
    assert snapshot.front_left.vehicle_id == "left-front"
    assert snapshot.front_right.vehicle_id == "right-front"
    assert snapshot.rear_left.vehicle_id == "left-rear"
    assert snapshot.rear_right.vehicle_id == "right-rear"
    assert snapshot.lane_occupancy == (True, False, True)


def test_exact_lateral_boundary_is_not_a_same_lane_overlap():
    ego = VehicleGeometry("ego", 0.0, 50.0, 1, 10.0, 4.0, 10.0)
    boundary = VehicleGeometry("boundary", 20.0, 46.0, 0, 10.0, 4.0, 10.0)
    snapshot = perceive(ego, [boundary], EnvironmentConfig())
    assert snapshot.lane_occupancy[0] is False
    assert snapshot.front_left.vehicle_id is None
    assert math.isinf(snapshot.front_left.gap)


def test_cruising_vehicle_has_infinite_ttc():
    ego = VehicleGeometry("ego", 0.0, 50.0, 1, 10.0, 4.0, 10.0)
    front = VehicleGeometry("front", 50.0, 50.0, 1, 10.0, 4.0, 10.0)
    snapshot = perceive(ego, [front], EnvironmentConfig())
    assert snapshot.front_center.mode is BehaviorMode.CRUISE
    assert math.isinf(snapshot.front_center.ttc)
