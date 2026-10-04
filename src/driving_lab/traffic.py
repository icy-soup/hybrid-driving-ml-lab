"""Independent, deterministic traffic-driver behavior."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Sequence

from .config import EnvironmentConfig
from .perception import VehicleGeometry, perceive

from .perception import PerceptionSnapshot
from .types import BehaviorMode, VehicleState


@dataclass(frozen=True)
class TrafficProfile:
    name: str = "balanced"
    desired_speed: float = 12.0
    headway: float = 1.5
    max_acceleration: float = 2.0
    comfortable_deceleration: float = 3.0
    lane_change_threshold: float = 35.0
    lane_change_cooldown: float = 3.0


@dataclass(frozen=True)
class TrafficDecision:
    target_speed: float
    target_lane: int
    behavior_mode: BehaviorMode
    reason: str


@dataclass(frozen=True)
class TrafficStepResult:
    vehicles: tuple[VehicleState, ...]
    decisions: dict[str, TrafficDecision]


class TrafficDriver:
    """A small explainable driver model; it never mutates the vehicle."""

    def __init__(self, profile: TrafficProfile | None = None) -> None:
        self.profile = profile or TrafficProfile()

    def decide(self, vehicle: VehicleState | object, perception: PerceptionSnapshot, time_step: float) -> TrafficDecision:
        speed = float(vehicle.speed)
        lane = int(vehicle.lane)
        front = perception.front_center
        desired = self.profile.desired_speed
        if front.vehicle_id is None:
            return TrafficDecision(desired, lane, BehaviorMode.CRUISE, "clear_lane")

        safe_gap = max(12.0, speed * self.profile.headway)
        if front.ttc < 1.5 or front.gap < max(8.0, safe_gap * 0.55):
            return TrafficDecision(max(0.0, front.gap / max(self.profile.headway, 0.5)), lane, BehaviorMode.EMERGENCY_BRAKE, "low_ttc")
        if front.gap < self.profile.lane_change_threshold:
            candidate = self._safe_lane_change(lane, perception)
            if candidate is not None:
                return TrafficDecision(desired, candidate, BehaviorMode.PREPARE_LANE_CHANGE, "slow_front_clear_adjacent")
        if front.gap < safe_gap:
            target_speed = min(desired, max(0.0, front.gap / max(self.profile.headway, 0.5)), speed)
            return TrafficDecision(target_speed, lane, BehaviorMode.FOLLOW, "maintain_headway")
        return TrafficDecision(min(desired, speed + self.profile.max_acceleration * max(time_step, 0.0)), lane, BehaviorMode.CRUISE, "recover_speed")

    @staticmethod
    def _safe_lane_change(lane: int, perception: PerceptionSnapshot) -> int | None:
        candidates = []
        if lane > 0:
            candidates.append((lane - 1, perception.front_left, perception.rear_left))
        if lane < len(perception.lane_occupancy) - 1:
            candidates.append((lane + 1, perception.front_right, perception.rear_right))
        for target, front, rear in candidates:
            if front.vehicle_id is None or front.gap > 25.0:
                if rear.vehicle_id is None or rear.gap > 20.0:
                    return target
        return None


class TrafficWorld:
    """Integrate traffic vehicles using independent drivers and safe gaps."""

    def __init__(self, config: EnvironmentConfig | None = None, profiles: dict[str, TrafficProfile] | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self._drivers = {key: TrafficDriver(value) for key, value in (profiles or {}).items()}

    def step_traffic(
        self,
        vehicles: Sequence[VehicleState],
        ego: VehicleState,
        scenario: object | None = None,
        time_step: float = 0.1,
    ) -> TrafficStepResult:
        geometries = [self._geometry(vehicle) for vehicle in vehicles]
        ego_geometry = self._geometry(ego)
        updated: list[VehicleState] = []
        decisions: dict[str, TrafficDecision] = {}
        for vehicle, geometry in zip(vehicles, geometries):
            snapshot = perceive(geometry, [ego_geometry, *[item for item in geometries if item.vehicle_id != geometry.vehicle_id]], self.config)
            driver = self._drivers.get(vehicle.vehicle_id, TrafficDriver())
            decision = driver.decide(vehicle, snapshot, time_step)
            decisions[vehicle.vehicle_id] = decision
            acceleration = max(-driver.profile.comfortable_deceleration, min(driver.profile.max_acceleration, (decision.target_speed - vehicle.speed) / max(time_step, 1e-6)))
            speed = max(0.0, vehicle.speed + acceleration * time_step)
            x = vehicle.x + speed * time_step
            y = decision.target_lane * self.config.lane_width
            updated.append(replace(vehicle, x=x, y=y, target_lane=decision.target_lane, desired_speed=decision.target_speed, acceleration=acceleration, mode=decision.behavior_mode))
        updated.sort(key=lambda item: (item.lane, -item.x))
        # Resolve same-lane penetration deterministically after integration.
        for index in range(1, len(updated)):
            lead, follower = updated[index - 1], updated[index]
            if lead.lane == follower.lane:
                minimum = (lead.length + follower.length) / 2.0 + self.config.traffic_min_gap
                if lead.x - follower.x < minimum:
                    updated[index] = replace(follower, x=lead.x - minimum, speed=min(follower.speed, lead.speed))
        return TrafficStepResult(tuple(updated), decisions)

    def _geometry(self, vehicle: VehicleState) -> VehicleGeometry:
        lane = int(vehicle.lane)
        return VehicleGeometry(vehicle.vehicle_id, vehicle.x, vehicle.y if vehicle.y is not None else lane * self.config.lane_width, lane, vehicle.length, vehicle.width, vehicle.speed, vehicle.mode)
