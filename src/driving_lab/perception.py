"""Geometry-based five-region perception for the highway model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .config import EnvironmentConfig
from .types import BehaviorMode


@dataclass(frozen=True)
class VehicleGeometry:
    vehicle_id: str
    x: float
    y: float
    lane: int
    length: float
    width: float
    speed: float
    mode: BehaviorMode = BehaviorMode.CRUISE


@dataclass(frozen=True)
class SensorReading:
    vehicle_id: str | None
    gap: float = math.inf
    relative_speed: float = 0.0
    ttc: float = math.inf
    mode: BehaviorMode = BehaviorMode.CRUISE


@dataclass(frozen=True)
class PerceptionSnapshot:
    front_center: SensorReading
    front_left: SensorReading
    front_right: SensorReading
    rear_left: SensorReading
    rear_right: SensorReading
    lane_occupancy: tuple[bool, ...]


def _lateral_overlap(a: VehicleGeometry, b: VehicleGeometry) -> bool:
    return abs(a.y - b.y) < (a.width + b.width) / 2.0


def _front_gap(ego: VehicleGeometry, other: VehicleGeometry) -> float:
    return other.x - other.length / 2.0 - (ego.x + ego.length / 2.0)


def _rear_gap(ego: VehicleGeometry, other: VehicleGeometry) -> float:
    return ego.x - ego.length / 2.0 - (other.x + other.length / 2.0)


def _ttc(gap: float, closing_speed: float) -> float:
    if gap < 0 or closing_speed <= 0:
        return 0.0 if gap < 0 else math.inf
    return gap / closing_speed


def _nearest(readings: list[SensorReading]) -> SensorReading:
    return min(readings, key=lambda item: item.gap) if readings else SensorReading(None)


def perceive(
    ego: VehicleGeometry,
    vehicles: Sequence[VehicleGeometry],
    config: EnvironmentConfig,
) -> PerceptionSnapshot:
    """Classify nearby vehicles into paper-inspired front/rear lane regions."""
    front: dict[str, list[SensorReading]] = {"center": [], "left": [], "right": []}
    rear: dict[str, list[SensorReading]] = {"left": [], "right": []}
    occupancy = [False] * config.lanes
    for vehicle in vehicles:
        if vehicle.vehicle_id == ego.vehicle_id:
            continue
        distance = abs(vehicle.x - ego.x)
        if distance > config.detection_distance:
            continue
        if 0 <= vehicle.lane < config.lanes:
            lane_center = vehicle.lane * config.lane_width
            # Occupancy follows the lane corridor, not a guessed sprite index.
            # At an exact corridor boundary the vehicle belongs to neither
            # adjacent lane until its geometry crosses that boundary.
            if abs(vehicle.y - lane_center) < config.lane_width / 2.0:
                occupancy[vehicle.lane] = True
        if vehicle.x >= ego.x:
            gap = _front_gap(ego, vehicle)
            if gap < 0:
                continue
            closing = ego.speed - vehicle.speed
            reading = SensorReading(vehicle.vehicle_id, gap, closing, _ttc(gap, closing), vehicle.mode)
            if _lateral_overlap(ego, vehicle):
                front["center"].append(reading)
            elif vehicle.lane == ego.lane - 1 and abs(vehicle.y - ego.y) > (vehicle.width + ego.width) / 2.0:
                front["left"].append(reading)
            elif vehicle.lane == ego.lane + 1 and abs(vehicle.y - ego.y) > (vehicle.width + ego.width) / 2.0:
                front["right"].append(reading)
        else:
            gap = _rear_gap(ego, vehicle)
            if gap < 0:
                continue
            reading = SensorReading(vehicle.vehicle_id, gap, vehicle.speed - ego.speed, math.inf, vehicle.mode)
            if vehicle.lane == ego.lane - 1:
                rear["left"].append(reading)
            elif vehicle.lane == ego.lane + 1:
                rear["right"].append(reading)
    return PerceptionSnapshot(
        front_center=_nearest(front["center"]),
        front_left=_nearest(front["left"]),
        front_right=_nearest(front["right"]),
        rear_left=_nearest(rear["left"]),
        rear_right=_nearest(rear["right"]),
        lane_occupancy=tuple(occupancy),
    )
