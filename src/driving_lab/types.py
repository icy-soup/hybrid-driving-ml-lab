"""Stable data contracts shared by environments, policies and experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class ActionType(IntEnum):
    """Discrete actions kept compatible with the original training data."""

    ACCELERATE = 0
    BRAKE = 1
    LANE_LEFT = 2
    LANE_RIGHT = 3
    CRUISE = 4


class BehaviorMode(IntEnum):
    """Explainable traffic behavior states used by perception and drivers."""

    CRUISE = 0
    FOLLOW = 1
    BRAKE = 2
    PREPARE_LANE_CHANGE = 3
    LANE_CHANGE = 4
    LANE_CHANGE_ABORT = 5
    EMERGENCY_BRAKE = 6
    CRASHED = 7


@dataclass(frozen=True)
class Action:
    """A single high-level decision returned by a policy."""

    kind: ActionType

    @property
    def index(self) -> int:
        return int(self.kind)


@dataclass(frozen=True)
class VehicleState:
    """Minimal kinematic state used by the headless environment."""

    vehicle_id: str
    lane: int
    x: float
    speed: float
    crashed: bool = False
    y: float | None = None
    target_lane: int | None = None
    desired_speed: float | None = None
    acceleration: float = 0.0
    length: float = 10.0
    width: float = 4.0
    mode: BehaviorMode = BehaviorMode.CRUISE
    # Optional legacy Pygame identity; appended to preserve positional callers.
    vehicle_type: str | None = None
    color: str | None = None
    sprite_state: int = 1
    # Number of render frames for which the legacy explosion overlay remains visible.
    explosion_frames: int = 0

    @property
    def exploding(self) -> bool:
        return self.explosion_frames > 0


@dataclass(frozen=True)
class Observation:
    """Ego-centric observation with an 11-dimensional legacy feature view."""

    speed: float
    lane: int
    front_distance: tuple[float, float, float]
    front_speed: tuple[float | None, float | None, float | None]
    ttc: tuple[float, float, float] = (math.inf, math.inf, math.inf)
    rear_distance: tuple[float, float, float] = (math.inf, math.inf, math.inf)
    rear_speed: tuple[float | None, float | None, float | None] = (None, None, None)
    crashed_front: tuple[bool, bool, bool] = (False, False, False)
    left_rear_present: bool = False
    right_rear_present: bool = False

    def to_features(self) -> np.ndarray:
        """Return the original 11-feature representation for model compatibility."""

        lane = max(0, min(2, int(self.lane)))
        speed_feature = (self.speed - (-36.0)) / ((-6.0) - (-36.0)) * 2.0 - 1.0
        lane_onehot = [0.0, 0.0, 0.0]
        lane_onehot[lane] = 1.0

        distance = self.front_distance[lane]
        relative_speed = 0.0
        if self.front_speed[lane] is not None:
            relative_speed = self.speed - float(self.front_speed[lane])
        normalized_distance = min(1.0, distance / 500.0) if math.isfinite(distance) else 1.0
        normalized_relative_speed = max(-1.0, min(1.0, relative_speed / 20.0))
        lane_ttc = self.ttc[lane]
        normalized_ttc = min(1.0, lane_ttc / 5.0) if math.isfinite(lane_ttc) else 1.0

        left_distance = self.front_distance[lane - 1] if lane > 0 else math.inf
        right_distance = self.front_distance[lane + 1] if lane < 2 else math.inf
        left_feature = min(1.0, left_distance / 500.0) if math.isfinite(left_distance) else 1.0
        right_feature = min(1.0, right_distance / 500.0) if math.isfinite(right_distance) else 1.0
        left_rear = self.left_rear_present or (lane > 0 and self.rear_distance[lane - 1] < 160.0)
        right_rear = self.right_rear_present or (lane < 2 and self.rear_distance[lane + 1] < 160.0)

        return np.asarray(
            [
                speed_feature,
                *lane_onehot,
                normalized_distance,
                normalized_relative_speed,
                normalized_ttc,
                left_feature,
                right_feature,
                1.0 if left_rear else 0.0,
                1.0 if right_rear else 0.0,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class Transition:
    """One environment step, suitable for evaluation and replay."""

    observation: Observation
    action: Action
    reward: float
    next_observation: Observation
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
