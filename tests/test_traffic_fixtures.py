from __future__ import annotations

import pytest

from tests.fixtures.traffic_cases import make_case


@pytest.mark.parametrize(
    ("name", "expected_ids"),
    [
        ("empty", []),
        ("following", ["traffic-front"]),
        ("rear_approach", ["traffic-rear"]),
        ("lane_change_conflict", ["target-front", "target-rear"]),
        ("collision", ["collision-front"]),
    ],
)
def test_cases_have_stable_ids_and_order(name, expected_ids):
    vehicles = make_case(name)
    assert [vehicle.vehicle_id for vehicle in vehicles] == expected_ids


def test_relative_positions_are_explicit_and_fresh():
    vehicles = make_case("following")
    assert vehicles[0].lane == 1
    assert vehicles[0].x > 0
    vehicles[0] = vehicles[0].__class__("changed", 1, 999.0, 1.0)
    assert make_case("following")[0].vehicle_id == "traffic-front"


def test_unknown_case_is_rejected():
    with pytest.raises(ValueError, match="unknown traffic case"):
        make_case("not-a-case")
