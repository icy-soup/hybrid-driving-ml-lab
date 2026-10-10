from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.legacy_ui_adapter import LegacyUIConfig, map_vehicle


def test_adapter_uses_original_lane_centers_and_ego_anchor():
    config = LegacyUIConfig()
    mapped = map_vehicle({"vehicle_id": "car-1", "x": 0.0, "lane": 1, "mode": "CRUISE"}, config)
    assert mapped.left == config.ego_left
    assert mapped.bottom == 250


def test_adapter_keeps_screen_entry_and_exit_geometry():
    config = LegacyUIConfig()
    entering = map_vehicle({"vehicle_id": "car-1", "x": -150.0, "lane": 0, "mode": "CRUISE"}, config)
    exiting = map_vehicle({"vehicle_id": "car-2", "x": 950.0, "lane": 2, "mode": "CRUISE"}, config)
    assert entering.left == config.ego_left - 150.0
    assert entering.bottom == 200
    assert exiting.left > config.road_width


def test_adapter_preserves_continuous_lateral_position_during_lane_change():
    config = LegacyUIConfig()
    mapped = map_vehicle({"vehicle_id": "car-1", "x": 0.0, "lane": 1, "y": 25.0}, config)

    assert mapped.bottom == 225.0
