"""Small deterministic traffic cases for contract and regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from driving_lab.types import VehicleState


def make_case(name: str) -> list[VehicleState]:
    """Return a fresh, deterministic vehicle list for a named audit case."""
    cases = {
        "empty": [],
        "following": [
            VehicleState("traffic-front", lane=1, x=120.0, speed=7.0),
        ],
        "rear_approach": [
            VehicleState("traffic-rear", lane=1, x=-80.0, speed=14.0),
        ],
        "lane_change_conflict": [
            VehicleState("target-front", lane=0, x=80.0, speed=8.0),
            VehicleState("target-rear", lane=0, x=-45.0, speed=12.0),
        ],
        "collision": [
            VehicleState("collision-front", lane=1, x=24.0, speed=6.0),
        ],
    }
    try:
        return list(cases[name])
    except KeyError as exc:
        raise ValueError(f"unknown traffic case: {name}") from exc
