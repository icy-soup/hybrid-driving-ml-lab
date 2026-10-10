"""Policy interfaces and the physics/safety-aware expert policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .physics import emergency_braking_distance, lane_change_front_gap, lane_change_rear_gap, safe_follow_distance
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
    # A neighbouring lane does not need to be empty for 320 px to be a safe
    # escape route. That threshold is for detecting a threat in the current
    # lane; requiring it again for the target lane made the ego brake in a
    # narrowing corridor until impact was unavoidable.
    safe_lane_front_distance: float = 180.0
    safe_rear_distance: float = 160.0
    prediction_horizon: int = 8

    def __post_init__(self) -> None:
        self.last_reason = "init"
        self.last_plan: tuple[str, ...] = ()

    def act(self, observation: Observation) -> Action:
        lane = max(0, min(2, observation.lane))
        front_distance = observation.front_distance[lane]
        front_speed = observation.front_speed[lane]
        current_ttc = observation.ttc[lane]
        self.last_reason = "cruise"
        self.last_plan = ()

        if observation.crashed_front[lane]:
            # A stopped/crashed obstacle is not a following target.  If an
            # adjacent lane has a speed-dependent safe window, leave the lane
            # first; braking alone cannot avoid a stationary obstacle once the
            # ego reaches the minimum simulation speed.
            candidates = self._safe_lanes(observation, lane)
            if candidates:
                target = self._best_lane(observation, candidates, lane)
                action = Action(ActionType.LANE_LEFT if target < lane else ActionType.LANE_RIGHT)
                self.last_reason = f"lane_change obstacle target={target}"
                return action
            self.last_reason = "brake obstacle; adjacent lanes fail front/rear prediction"
            return Action(ActionType.BRAKE)

        closing_speed = 0.0
        if front_speed is not None:
            closing_speed = abs(observation.speed) - abs(front_speed)
            if current_ttc <= self.emergency_ttc:
                self.last_reason = "brake emergency_ttc"
                return Action(ActionType.BRAKE)
            if closing_speed > 0.0 and front_distance <= emergency_braking_distance(
                closing_speed,
                current_speed=observation.speed,
                target_speed=front_speed,
            ):
                self.last_reason = "brake stopping_distance"
                return Action(ActionType.BRAKE)

        lane_threat = (
            front_speed is not None
            and closing_speed > 0.0
            and (front_distance < self.lane_change_distance or current_ttc < self.lane_change_ttc)
        )
        if lane_threat:
            candidates = self._safe_lanes(observation, lane)
            if candidates:
                target = self._best_lane(observation, candidates, lane)
                self.last_reason = f"lane_change predicted_gap target={target}"
                return Action(ActionType.LANE_LEFT if target < lane else ActionType.LANE_RIGHT)

        if (
            front_speed is not None
            and closing_speed > 0.0
            and front_distance < safe_follow_distance(abs(observation.speed))
        ):
            self.last_reason = "brake follow_distance"
            return Action(ActionType.BRAKE)

        # Once the ego is inside the following envelope, do not accelerate
        # back toward the target speed merely because the lead vehicle is a
        # little faster.  Holding speed here prevents the brake/accelerate
        # oscillation that otherwise closes a small gap one frame at a time.
        if (
            front_speed is not None
            and front_distance < safe_follow_distance(abs(observation.speed))
        ):
            self.last_reason = "cruise match_follow_speed"
            return Action(ActionType.CRUISE)

        if abs(observation.speed) < self.target_speed:
            self.last_reason = "accelerate target_speed"
            return Action(ActionType.ACCELERATE)
        self.last_reason = "cruise target_speed"
        return Action(ActionType.CRUISE)

    def _safe_lanes(self, observation: Observation, current_lane: int) -> list[int]:
        safe: list[int] = []
        for target_lane in (current_lane - 1, current_lane + 1):
            if target_lane < 0 or target_lane > 2:
                continue
            target_front_speed = observation.front_speed[target_lane]
            target_speed = abs(float(target_front_speed)) if target_front_speed is not None else abs(observation.speed)
            required_front = max(
                self.safe_lane_front_distance,
                lane_change_front_gap(abs(observation.speed), target_speed),
            )
            predicted_front_gap = self._predicted_front_gap(
                observation.front_distance[target_lane],
                abs(observation.speed),
                target_speed,
            )
            front_clear = math.isinf(observation.front_distance[target_lane]) or predicted_front_gap >= required_front
            target_rear_speed = observation.rear_speed[target_lane]
            rear_speed = abs(float(target_rear_speed)) if target_rear_speed is not None else abs(observation.speed)
            required_rear = max(
                self.safe_rear_distance,
                lane_change_rear_gap(abs(observation.speed), rear_speed),
            )
            predicted_rear_gap = self._predicted_rear_gap(
                observation.rear_distance[target_lane],
                abs(observation.speed),
                rear_speed,
            )
            rear_clear = math.isinf(observation.rear_distance[target_lane]) or predicted_rear_gap >= required_rear
            if front_clear and rear_clear:
                safe.append(target_lane)
        return safe

    def _predicted_front_gap(self, gap: float, ego_speed: float, front_speed: float) -> float:
        if not math.isfinite(gap):
            return gap
        return gap - max(0.0, ego_speed - front_speed) * self.prediction_horizon

    def _predicted_rear_gap(self, gap: float, ego_speed: float, rear_speed: float) -> float:
        if not math.isfinite(gap):
            return gap
        return gap - max(0.0, rear_speed - ego_speed) * self.prediction_horizon

    def _best_lane(self, observation: Observation, candidates: list[int], current_lane: int) -> int:
        """Choose the lane with the largest predicted front/rear safety margin."""
        def margin(target_lane: int) -> float:
            front_speed = observation.front_speed[target_lane]
            rear_speed = observation.rear_speed[target_lane]
            fs = abs(float(front_speed)) if front_speed is not None else abs(observation.speed)
            rs = abs(float(rear_speed)) if rear_speed is not None else abs(observation.speed)
            front_need = max(self.safe_lane_front_distance, lane_change_front_gap(abs(observation.speed), fs))
            rear_need = max(self.safe_rear_distance, lane_change_rear_gap(abs(observation.speed), rs))
            return min(
                self._predicted_front_gap(observation.front_distance[target_lane], abs(observation.speed), fs) - front_need,
                self._predicted_rear_gap(observation.rear_distance[target_lane], abs(observation.speed), rs) - rear_need,
            )
        return max(candidates, key=margin)


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
        # The learned controller is allowed to choose the nominal action, but
        # it must not suppress the expert's avoidance manoeuvre.  Previously
        # the shield only intervened at the emergency-braking threshold.  A
        # model that kept returning CRUISE therefore drove into a slower lead
        # vehicle until the last instant and never issued a lane change.
        fallback_action = self.fallback.act(observation)
        if fallback_action.kind in (ActionType.LANE_LEFT, ActionType.LANE_RIGHT):
            if proposed.kind is not fallback_action.kind:
                self.last_override = True
                return fallback_action
            return proposed

        lane = max(0, min(2, observation.lane))
        front_speed = observation.front_speed[lane]
        front_distance = observation.front_distance[lane]
        closing_speed = 0.0
        if front_speed is not None:
            closing_speed = abs(observation.speed) - abs(front_speed)

        imminent = observation.crashed_front[lane] or observation.ttc[lane] <= self.fallback.emergency_ttc
        imminent = imminent or (
            closing_speed > 0.0
            and front_distance <= emergency_braking_distance(
                closing_speed,
                current_speed=observation.speed,
                target_speed=front_speed,
            )
        )
        if imminent and proposed.kind is not ActionType.BRAKE:
            self.last_override = True
            return Action(ActionType.BRAKE)
        # Follow the rule controller's earlier braking decision as well.  This
        # keeps a learned CRUISE/ACCELERATE output from turning a recoverable
        # closing situation into a collision.
        if fallback_action.kind is ActionType.BRAKE and proposed.kind is not ActionType.BRAKE:
            self.last_override = True
            return fallback_action
        return proposed


def action_from_index(index: int) -> Action:
    """Decode a persisted legacy action index with validation."""

    try:
        return Action(ActionType(int(index)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown action index: {index}") from exc
