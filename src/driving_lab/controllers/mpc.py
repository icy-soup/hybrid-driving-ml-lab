"""Discrete short-horizon model predictive controller."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

from ..physics import emergency_braking_distance, lane_change_front_gap, lane_change_rear_gap, safe_follow_distance
from ..types import Action, ActionType, Observation


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 5
    candidate_actions: tuple[int, ...] = (0, 1, 2, 3, 4)
    collision_cost: float = 100.0
    progress_cost: float = 1.0
    speed_cost: float = 0.1
    lane_change_cost: float = 0.2
    # Safety thresholds used before the short-horizon search.  The discrete
    # search is deliberately small, so it needs a physically meaningful
    # guard to avoid accelerating into a lead vehicle between replans.
    lane_change_front_margin: float = 90.0
    lane_change_rear_margin: float = 70.0
    lane_change_ttc: float = 3.0
    # Exposed as a policy parameter so the later RL agent can learn it.
    target_speed: float = 27.7777777778  # 100 km/h in the internal m/s scale


class MPCPolicy:
    def __init__(self, env_model=None, config: MPCConfig | None = None) -> None:
        self.env_model = env_model
        self.config = config or MPCConfig()
        if self.config.horizon < 1:
            raise ValueError("horizon must be positive")
        self.last_plan: list[Action] = []
        self.last_reason: str = "search"

    def act(self, observation: Observation) -> Action:
        self.last_reason = "search"
        safety_action = self._safety_action(observation)
        if safety_action is not None:
            self.last_plan = [safety_action]
            return safety_action
        actions = [Action(ActionType(index)) for index in self.config.candidate_actions]
        best_sequence = None
        best_score = float("inf")
        # Branching stays deliberately small for the discrete baseline.
        for sequence in product(actions, repeat=self.config.horizon):
            score = self._score(sequence, observation)
            if score < best_score:
                best_score, best_sequence = score, sequence
        self.last_plan = list(best_sequence or (actions[-1],))
        self.last_reason = "horizon_search"
        return self.last_plan[0]

    def _safety_action(self, observation: Observation) -> Action | None:
        """Return an immediate safety action when the horizon is too optimistic.

        The old score model only propagated the current lane's gap and did not
        model a lane change at all.  Consequently every candidate was allowed
        to accelerate until the collision was already inside the next few
        frames.  This guard uses the same distance and TTC quantities exposed
        by the environment, then lets the short horizon optimize only when no
        immediate conflict is present.
        """
        lane = max(0, min(2, int(observation.lane)))
        speed = abs(float(observation.speed))
        gap = float(observation.front_distance[lane])
        front_speed_raw = observation.front_speed[lane]
        front_speed = abs(float(front_speed_raw)) if front_speed_raw is not None else speed
        closing = max(0.0, speed - front_speed)
        ttc = float(observation.ttc[lane])

        # Apply speed regulation even when the current lane has no detected
        # lead vehicle.  This branch used to return immediately and allowed
        # the progress term to accelerate the ego to the 140 km/h road cap.
        if math.isinf(gap) and front_speed_raw is None:
            if speed > self.config.target_speed + 1.0:
                self.last_reason = "cruise_speed_control"
                return Action(ActionType.BRAKE)
            if speed < self.config.target_speed - 1.0:
                self.last_reason = "cruise_speed_control"
                return Action(ActionType.ACCELERATE)
            self.last_reason = "cruise_speed_hold"
            return Action(ActionType.CRUISE)

        threat = (
            bool(observation.crashed_front[lane])
            or gap <= emergency_braking_distance(closing, max_deceleration=0.5)
            or ttc <= self.config.lane_change_ttc
            or (closing > 0.0 and gap <= max(220.0, safe_follow_distance(speed)))
        )
        if not threat:
            # Keep the baseline at its selected cruising target.  The old
            # progress-heavy score drove the ego to the 36 m/s ceiling even
            # on an empty road, which made every newly appearing vehicle an
            # emergency.  This target remains a tunable RL parameter.
            if speed > self.config.target_speed + 1.0:
                self.last_reason = "cruise_speed_control"
                return Action(ActionType.BRAKE)
            if speed < self.config.target_speed - 1.0:
                self.last_reason = "cruise_speed_control"
                return Action(ActionType.ACCELERATE)
            self.last_reason = "cruise_speed_hold"
            return Action(ActionType.CRUISE)

        target_lane = self._safe_adjacent_lane(observation, lane, speed)
        if target_lane is not None:
            self.last_reason = "safe_lane_change"
            return Action(ActionType.LANE_LEFT if target_lane < lane else ActionType.LANE_RIGHT)

        # No lane is available.  Match the lead speed when possible; braking
        # below the lead speed is unnecessary and makes the controller stop in
        # an otherwise recoverable following situation.
        if front_speed_raw is not None and not observation.crashed_front[lane]:
            if speed > front_speed + 0.25:
                self.last_reason = "rear_blocked_brake" if self._adjacent_rear_blocked(observation, lane, speed) else "braking_required"
                return Action(ActionType.BRAKE)
            self.last_reason = "match_lead_speed"
            return Action(ActionType.CRUISE)
        self.last_reason = "rear_blocked_brake" if self._adjacent_rear_blocked(observation, lane, speed) else "obstacle_brake"
        return Action(ActionType.BRAKE)

    def _adjacent_rear_blocked(self, observation: Observation, current_lane: int, speed: float) -> bool:
        for target_lane in (current_lane - 1, current_lane + 1):
            if target_lane < 0 or target_lane > 2:
                continue
            rear_gap = observation.rear_distance[target_lane]
            rear_speed_raw = observation.rear_speed[target_lane]
            rear_speed = abs(float(rear_speed_raw)) if rear_speed_raw is not None else speed
            if not math.isinf(rear_gap) and rear_gap < max(
                self.config.lane_change_rear_margin + 2.0 * speed,
                lane_change_rear_gap(speed, rear_speed),
            ):
                return True
        return False

    def _safe_adjacent_lane(
        self, observation: Observation, current_lane: int, speed: float
    ) -> int | None:
        candidates: list[tuple[float, int]] = []
        for target_lane in (current_lane - 1, current_lane + 1):
            if target_lane < 0 or target_lane > 2:
                continue
            front_gap = observation.front_distance[target_lane]
            rear_gap = observation.rear_distance[target_lane]
            target_front_speed = observation.front_speed[target_lane]
            target_speed = abs(float(target_front_speed)) if target_front_speed is not None else speed
            front_required = max(
                self.config.lane_change_front_margin + 2.5 * speed,
                lane_change_front_gap(speed, target_speed),
            )
            # A finite rear gap is a bumper-to-bumper distance.  Add a
            # closing-speed term so a fast approaching rear car cannot be
            # treated as safe merely because it is outside a fixed pixel box.
            target_rear_speed = observation.rear_speed[target_lane]
            rear_speed = abs(float(target_rear_speed)) if target_rear_speed is not None else speed
            rear_required = max(
                self.config.lane_change_rear_margin + 2.0 * speed,
                lane_change_rear_gap(speed, rear_speed),
            )
            rear_present = not math.isinf(rear_gap)
            if rear_present and rear_gap < rear_required:
                continue
            if not math.isinf(front_gap) and front_gap < front_required:
                continue
            candidates.append((front_gap, target_lane))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _score(self, sequence: tuple[Action, ...], observation: Observation) -> float:
        speed = abs(observation.speed)
        gap = observation.front_distance[observation.lane]
        ttc = observation.ttc[observation.lane]
        score = 0.0
        for step, action in enumerate(sequence):
            if step == 0 and ttc < 1.0 and action.kind is not ActionType.BRAKE:
                score += self.config.collision_cost * 10.0
            if action.kind is ActionType.ACCELERATE:
                speed += 0.5
            elif action.kind is ActionType.BRAKE:
                speed = max(0.0, speed - 0.5)
            elif action.kind in (ActionType.LANE_LEFT, ActionType.LANE_RIGHT):
                score += self.config.lane_change_cost
            gap -= speed
            if gap <= 0 or ttc < 1.0:
                score += self.config.collision_cost
            score -= self.config.progress_cost * max(speed, 0.0)
            score += self.config.speed_cost * abs(speed - self.config.target_speed)
            if ttc < 3.0:
                ttc = max(0.0, ttc - 0.5)
        return score
