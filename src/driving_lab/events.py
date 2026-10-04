"""Typed collision events and serializable episode diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .perception import VehicleGeometry


@dataclass(frozen=True)
class CollisionEvent:
    vehicle_a: str
    vehicle_b: str
    collision_type: str
    step: int
    scenario: str
    lane_a: int
    lane_b: int
    gap: float
    relative_speed: float
    ttc: float
    reason: str


def classify_collision(a: VehicleGeometry, b: VehicleGeometry, *, step: int, scenario: str) -> CollisionEvent:
    gap = b.x - b.length / 2.0 - (a.x + a.length / 2.0)
    if a.lane == b.lane:
        collision_type = "rear_end"
        reason = "same_lane_longitudinal_overlap"
    else:
        collision_type = "side"
        reason = "lateral_overlap_during_lane_interaction"
    closing = a.speed - b.speed
    ttc = gap / closing if closing > 0 and gap >= 0 else (0.0 if gap < 0 else math.inf)
    return CollisionEvent(a.vehicle_id, b.vehicle_id, collision_type, step, scenario, a.lane, b.lane, gap, closing, ttc, reason)


@dataclass
class EpisodeDiagnostics:
    scenario: str
    seed: int | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    collisions: list[CollisionEvent] = field(default_factory=list)

    def record_step(self, step: int, snapshot: dict[str, Any]) -> None:
        self.steps.append({"step": int(step), **_json_safe(snapshot)})

    def record_collision(self, event: CollisionEvent) -> None:
        self.collisions.append(event)

    def summary(self) -> dict[str, Any]:
        ttcs = [item.get("ttc") for item in self.steps if isinstance(item.get("ttc"), (int, float))]
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "steps": len(self.steps),
            "collision_count": len(self.collisions),
            "collision_types": _counts(event.collision_type for event in self.collisions),
            "minimum_ttc": min(ttcs) if ttcs else None,
        }

    def write_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for item in self.steps:
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            for event in self.collisions:
                handle.write(json.dumps({"type": "collision", **_json_safe(asdict(event))}, ensure_ascii=False, allow_nan=False) + "\n")


def _counts(values):
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
