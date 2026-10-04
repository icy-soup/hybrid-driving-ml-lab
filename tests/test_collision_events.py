from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.events import classify_collision, EpisodeDiagnostics
from driving_lab.perception import VehicleGeometry


def test_rear_end_event_contains_geometry_and_reason():
    front = VehicleGeometry("front", 0.0, 50.0, 1, 10.0, 4.0, 6.0)
    rear = VehicleGeometry("rear", 0.0, 50.0, 1, 10.0, 4.0, 12.0)
    event = classify_collision(rear, front, step=4, scenario="slow_lead")
    assert event.collision_type == "rear_end"
    assert event.vehicle_a == "rear"
    assert event.gap == pytest.approx(-10.0)
    assert event.reason


def test_side_collision_is_classified_by_lateral_overlap():
    a = VehicleGeometry("a", 0.0, 48.0, 1, 10.0, 4.0, 8.0)
    b = VehicleGeometry("b", 0.0, 52.0, 2, 10.0, 4.0, 8.0)
    event = classify_collision(a, b, step=1, scenario="random")
    assert event.collision_type == "side"


def test_diagnostics_summary_is_json_ready():
    diagnostics = EpisodeDiagnostics("random", seed=7)
    diagnostics.record_step(1, {"speed": 8.0})
    assert diagnostics.summary()["steps"] == 1
    assert diagnostics.summary()["minimum_ttc"] is None


def test_cli_exposes_diagnose_command():
    from driving_lab.cli import build_parser

    args = build_parser().parse_args(["diagnose", "--scenario", "empty", "--seed", "3"])
    assert args.command == "diagnose"
