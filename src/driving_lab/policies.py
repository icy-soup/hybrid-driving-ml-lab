"""Policy interfaces and the physics/safety-aware expert policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .physics import emergency_braking_distance, safe_follow_distance
from .types import Action, ActionType, Observation


class Policy(Protocol):
    def act(self, observation: Observation) -> Action:
        """Map one ego-centric observation to a discrete action."""


@dataclass
class RulePolicy:
    """Deterministic expert policy used as the behavior-cloning teacher."""

    target_speed: float = 12.0
    emergency_ttc: float = 1.5
    lane_change_ttc: float = 2.5
    lane_change_distance: float = 320.0
    safe_rear_distance: float = 160.0

    def act(self, observation: Observation) -> Action:
        lane = max(0, min(2, observation.lane))
        front_distance = observation.front_distance[lane]
        front_speed = observation.front_speed[lane]
        current_ttc = observation.ttc[lane]

        if observation.crashed_front[lane]:
            return Action(ActionType.BRAKE)

        closing_speed = 0.0
        if front_speed is not None:
            closing_speed = abs(observation.speed) - abs(front_speed)
            if current_ttc <= self.emergency_ttc:
                return Action(ActionType.BRAKE)
            if closing_speed > 0.0 and front_distance <= emergency_braking_distance(closing_speed):
                return Action(ActionType.BRAKE)

        lane_threat = (
            front_speed is not None
            and closing_speed > 0.0
            and (front_distance < self.lane_change_distance or current_ttc < self.lane_change_ttc)
        )
        if lane_threat:
            candidates = self._safe_lanes(observation, lane)
            if candidates:
                target = max(candidates, key=lambda target_lane: observation.front_distance[target_lane])
                return Action(
                    ActionType.LANE_LEFT if target < lane else ActionType.LANE_RIGHT
                )

        if (
            front_speed is not None
            and closing_speed > 0.0
            and front_distance < safe_follow_distance(abs(observation.speed))
        ):
            return Action(ActionType.BRAKE)

        if abs(observation.speed) < self.target_speed:
            return Action(ActionType.ACCELERATE)
        return Action(ActionType.CRUISE)

    def _safe_lanes(self, observation: Observation, current_lane: int) -> list[int]:
        safe: list[int] = []
        for target_lane in (current_lane - 1, current_lane + 1):
            if target_lane < 0 or target_lane > 2:
                continue
            front_clear = (
                math.isinf(observation.front_distance[target_lane])
                or observation.front_distance[target_lane] >= self.lane_change_distance
            )
            rear_clear = (
                math.isinf(observation.rear_distance[target_lane])
                or observation.rear_distance[target_lane] >= self.safe_rear_distance
            )
            if front_clear and rear_clear:
                safe.append(target_lane)
        return safe


class NeuralPolicy:
    """Adapter that turns a trained NumPy classifier into a policy."""

    def __init__(self, model) -> None:
        self.model = model

    def act(self, observation: Observation) -> Action:
        prediction = self.model.predict(observation.to_features())
        return action_from_index(int(prediction[0]))


@dataclass
class SafetyShieldPolicy:
    """Allow a learned policy to act while preserving hard emergency rules."""

    learned_policy: Policy
    fallback: RulePolicy | None = None

    def __post_init__(self) -> None:
        if self.fallback is None:
            self.fallback = RulePolicy()
        self.last_override = False

    def act(self, observation: Observation) -> Action:
        self.last_override = False
        proposed = self.learned_policy.act(observation)
        lane = max(0, min(2, observation.lane))
        front_speed = observation.front_speed[lane]
        front_distance = observation.front_distance[lane]
        closing_speed = 0.0
        if front_speed is not None:
            closing_speed = abs(observation.speed) - abs(front_speed)

        imminent = observation.crashed_front[lane] or observation.ttc[lane] <= self.fallback.emergency_ttc
        imminent = imminent or (
            closing_speed > 0.0
            and front_distance <= emergency_braking_distance(closing_speed)
        )
        if imminent and proposed.kind is not ActionType.BRAKE:
            self.last_override = True
            return Action(ActionType.BRAKE)
        return proposed


def action_from_index(index: int) -> Action:
    """Decode a persisted legacy action index with validation."""

    try:
        return Action(ActionType(int(index)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown action index: {index}") from exc
