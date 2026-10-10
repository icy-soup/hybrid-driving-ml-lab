"""Mapping from canonical environment coordinates to the original UI geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyUIConfig:
    road_width: int = 910
    panel_width: int = 240
    road_height: int = 450
    ego_left: float = 455.0
    lane_centers: tuple[float, float, float] = (200.0, 250.0, 300.0)
    entry_margin: float = 150.0
    remove_margin: float = 400.0


@dataclass(frozen=True)
class LegacyVehicleDraw:
    vehicle_id: str
    left: float
    bottom: float
    lane: int
    mode: str
    crashed: bool = False
    vehicle_type: str | None = None
    color: str | None = None
    sprite_state: int = 1


def map_vehicle(vehicle: dict, config: LegacyUIConfig | None = None) -> LegacyVehicleDraw:
    config = config or LegacyUIConfig()
    lane = max(0, min(len(config.lane_centers) - 1, int(vehicle.get("lane", 1))))
    logical_y = vehicle.get("y")
    bottom = (
        200.0 + float(logical_y)
        if logical_y is not None
        else config.lane_centers[lane]
    )
    return LegacyVehicleDraw(
        vehicle_id=str(vehicle.get("vehicle_id", "traffic")),
        left=config.ego_left + float(vehicle.get("x", 0.0)),
        bottom=bottom,
        lane=lane,
        mode=str(vehicle.get("mode", "CRUISE")),
        crashed=bool(vehicle.get("crashed", False)),
        vehicle_type=vehicle.get("vehicle_type"),
        color=vehicle.get("color"),
        sprite_state=int(vehicle.get("sprite_state", 1)),
    )
