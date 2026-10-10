"""Independent, deterministic traffic-driver behavior."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Sequence

from .config import EnvironmentConfig
from .perception import VehicleGeometry, perceive

from .perception import PerceptionSnapshot
from .physics import lane_change_front_gap, lane_change_rear_gap
from .types import BehaviorMode, VehicleState


@dataclass(frozen=True)
class TrafficProfile:
    name: str = "balanced"
    desired_speed: float = 12.0
    headway: float = 1.5
    max_acceleration: float = 2.0
    comfortable_deceleration: float = 3.0
    # Legacy EnemyAI begins checking adjacent lanes well before the
    # collision corridor; 35 px caused traffic to remain in FOLLOW forever.
    lane_change_threshold: float = 300.0
    lane_change_cooldown: float = 3.0
    lane_change_speed: float = 3.0


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

        # Keep a geometry-aware minimum gap in addition to time headway. This
        # makes the behavior state change before the integrator has to clamp
        # an already-too-close vehicle after the fact.
        safe_gap = max(72.0, 12.0, speed * self.profile.headway)
        emergency_gap = max(8.0, speed * self.profile.headway * 0.55)
        # Decide a safe escape lane before applying emergency braking. The
        # old order returned EMERGENCY_BRAKE as soon as TTC became small, so a
        # vehicle trapped behind a slow ego car never reached its lane-change
        # policy and eventually entered the collision corridor.
        if front.gap >= 20.0 and front.gap < self.profile.lane_change_threshold and front.relative_speed > 0.5:
            candidate = self._safe_lane_change(lane, perception, speed)
            if candidate is not None:
                return TrafficDecision(desired, candidate, BehaviorMode.PREPARE_LANE_CHANGE, "slow_front_clear_adjacent")
        if front.ttc < 1.5 or front.gap < emergency_gap:
            return TrafficDecision(max(0.0, front.gap / max(self.profile.headway, 0.5)), lane, BehaviorMode.EMERGENCY_BRAKE, "low_ttc")
        if front.relative_speed > 0.0 and front.gap < 160.0:
            front_speed = max(0.0, speed - front.relative_speed)
            target_speed = min(desired, front_speed + max(0.0, front.gap - safe_gap) * 0.05)
            return TrafficDecision(target_speed, lane, BehaviorMode.FOLLOW, "closing_on_front_vehicle")
        if front.gap <= safe_gap:
            target_speed = min(desired, max(0.0, front.gap / max(self.profile.headway, 0.5)), speed)
            return TrafficDecision(target_speed, lane, BehaviorMode.FOLLOW, "maintain_headway")
        return TrafficDecision(min(desired, speed + self.profile.max_acceleration * max(time_step, 0.0)), lane, BehaviorMode.CRUISE, "recover_speed")

    def _safe_lane_change(self, lane: int, perception: PerceptionSnapshot, speed: float) -> int | None:
        # Convert the lane-change window to a time-headway rule.  A fixed
        # pixel gap is safe at low speed but unsafe when the vehicle is fast.
        candidates = []
        if lane > 0:
            candidates.append((lane - 1, perception.front_left, perception.rear_left))
        if lane < len(perception.lane_occupancy) - 1:
            candidates.append((lane + 1, perception.front_right, perception.rear_right))
        for target, front, rear in candidates:
            front_speed = speed - front.relative_speed if front.vehicle_id is not None else speed
            front_required = max(
                25.0,
                speed * self.profile.headway + 0.5 * speed,
                lane_change_front_gap(speed, front_speed),
            )
            front_ok = front.vehicle_id is None or front.gap >= front_required
            if not front_ok:
                continue
            if rear.vehicle_id is None:
                return target
            rear_closing = max(0.0, rear.relative_speed)
            rear_speed = speed + rear_closing
            rear_required = max(
                20.0,
                rear_speed * self.profile.headway + 0.5 * rear_closing,
                lane_change_rear_gap(speed, rear_speed),
            )
            ttc_ok = rear_closing <= 0.0 or rear.gap / rear_closing >= self.profile.headway
            if rear.gap >= rear_required and ttc_ok:
                return target
        return None


class TrafficWorld:
    """Integrate traffic vehicles using independent drivers and safe gaps."""

    def __init__(self, config: EnvironmentConfig | None = None, profiles: dict[str, TrafficProfile] | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self._drivers = {key: TrafficDriver(value) for key, value in (profiles or {}).items()}
        self._default_driver = TrafficDriver(
            TrafficProfile(desired_speed=self.config.traffic_cruise_speed)
        )
        self._scenario_driver = TrafficDriver(TrafficProfile(desired_speed=12.0))
        self._lane_change_cooldowns: dict[str, int] = {}

    def step_traffic(
        self,
        vehicles: Sequence[VehicleState],
        ego: VehicleState,
        scenario: object | None = None,
        time_step: float = 0.1,
        relative_to_ego: bool = False,
    ) -> TrafficStepResult:
        geometries = [self._geometry(vehicle) for vehicle in vehicles]
        ego_geometry = self._geometry(ego)
        updated: list[VehicleState] = []
        decisions: dict[str, TrafficDecision] = {}
        for vehicle, geometry in zip(vehicles, geometries):
            if vehicle.crashed:
                # Match the legacy Enemy.move() branch: a crashed vehicle is
                # no longer part of traffic control, but it keeps following
                # the road scroll until it leaves the removal boundary. This
                # prevents a crashed sprite from freezing in front of, or
                # being pushed by, another vehicle forever.
                crashed_x = vehicle.x
                if relative_to_ego:
                    crashed_x -= ego.speed * time_step
                updated.append(replace(
                    vehicle,
                    x=crashed_x,
                    speed=0.0,
                    desired_speed=0.0,
                    acceleration=0.0,
                    mode=BehaviorMode.CRASHED,
                    explosion_frames=vehicle.explosion_frames,
                ))
                decisions[vehicle.vehicle_id] = TrafficDecision(0.0, vehicle.lane, BehaviorMode.CRASHED, "vehicle_already_crashed")
                continue
            snapshot = perceive(geometry, [ego_geometry, *[item for item in geometries if item.vehicle_id != geometry.vehicle_id]], self.config)
            driver = self._drivers.get(vehicle.vehicle_id)
            if driver is None:
                # The hand-authored regression scenes intentionally contain
                # slow leads and sudden obstacles.  Keep their calibrated
                # behavior, while random highway traffic uses the highway
                # cruise profile above.
                driver = (
                    self._scenario_driver
                    if scenario not in (None, "random", "mixed")
                    else self._default_driver
                )
            cooldown = self._lane_change_cooldowns.get(vehicle.vehicle_id, 0)
            if cooldown > 0:
                self._lane_change_cooldowns[vehicle.vehicle_id] = cooldown - 1
            if vehicle.target_lane is not None and vehicle.target_lane != vehicle.lane:
                # A lane change is an active maneuver. Do not re-plan its
                # target on every frame, otherwise traffic oscillates, but
                # still recompute longitudinal speed so a new front vehicle
                # can trigger braking during the maneuver.
                speed_decision = driver.decide(vehicle, snapshot, time_step)
                decision = TrafficDecision(
                    target_speed=speed_decision.target_speed,
                    target_lane=vehicle.target_lane,
                    behavior_mode=BehaviorMode.LANE_CHANGE,
                    reason="continue_lane_change" if speed_decision.behavior_mode is BehaviorMode.CRUISE else speed_decision.reason,
                )
            elif cooldown > 0:
                speed_decision = driver.decide(vehicle, snapshot, time_step)
                decision = TrafficDecision(
                    target_speed=speed_decision.target_speed,
                    target_lane=vehicle.lane,
                    behavior_mode=(
                        speed_decision.behavior_mode
                        if speed_decision.behavior_mode is not BehaviorMode.PREPARE_LANE_CHANGE
                        else BehaviorMode.FOLLOW
                    ),
                    reason="lane_change_cooldown" if speed_decision.behavior_mode is BehaviorMode.CRUISE else speed_decision.reason,
                )
            else:
                decision = driver.decide(vehicle, snapshot, time_step)
                if decision.target_lane != vehicle.lane:
                    self._lane_change_cooldowns[vehicle.vehicle_id] = int(driver.profile.lane_change_cooldown / max(time_step, 1e-6))
            # In the relative frame the position update already subtracts the
            # ego speed.  Keep the driver's target speed here so an NPC on a
            # clear lane can accelerate toward the highway profile instead of
            # being frozen forever at its spawn speed.
            # Re-check the target lane on every frame during the maneuver.
            # A gap that was safe at initiation can close before the 3 px/frame
            # lateral motion finishes, especially when the ego approaches.
            current_y = vehicle.y if vehicle.y is not None else vehicle.lane * self.config.lane_width
            hold_lateral = False

            def lane_blocked(target_lane: int) -> bool:
                target_y = target_lane * self.config.lane_width
                return any(
                    other.vehicle_id != vehicle.vehicle_id
                    and abs(
                        (other.y if other.y is not None else other.lane * self.config.lane_width)
                        - target_y
                    ) < (other.width + vehicle.width) / 2.0 + 8.0
                    and abs(other.x - vehicle.x) < max(90.0, (other.length + vehicle.length) / 2.0 + 24.0)
                    for other in (ego, *vehicles)
                )

            def path_blocked(target_lane: int) -> bool:
                target_y = target_lane * self.config.lane_width
                low_y, high_y = sorted((current_y, target_y))
                for other in (ego, *vehicles):
                    if other.vehicle_id == vehicle.vehicle_id:
                        continue
                    other_current_y = other.y if other.y is not None else other.lane * self.config.lane_width
                    other_target_y = (
                        other.target_lane * self.config.lane_width
                        if other.target_lane is not None
                        else other_current_y
                    )
                    other_low, other_high = sorted((other_current_y, other_target_y))
                    margin = (other.width + vehicle.width) / 2.0 + 8.0
                    vertical_overlap = not (high_y + margin < other_low or other_high + margin < low_y)
                    if vertical_overlap and abs(other.x - vehicle.x) < max(90.0, (other.length + vehicle.length) / 2.0 + 24.0):
                        return True
                return False

            # A maneuver can be returning to its original lane after an abort,
            # so checking only ``target_lane != vehicle.lane`` is insufficient.
            # Evaluate the actual lateral target every frame and choose the
            # closest open lane when the requested one has closed.
            if abs(current_y - decision.target_lane * self.config.lane_width) > 1e-6 and lane_blocked(decision.target_lane):
                open_lanes = [
                    lane
                    for lane in range(self.config.lanes)
                    if not lane_blocked(lane) and not path_blocked(lane)
                ]
                if open_lanes:
                    target_lane = min(
                        open_lanes,
                        key=lambda lane: abs(lane * self.config.lane_width - current_y),
                    )
                else:
                    target_lane = min(
                        range(self.config.lanes),
                        key=lambda lane: abs(lane * self.config.lane_width - current_y),
                    )
                    # Every lane is currently occupied along the swept path.
                    # Hold the current lateral position until the blocker has
                    # passed; steering into the nearest lane would otherwise
                    # move a rear vehicle through the ego car.
                    hold_lateral = True
                decision = TrafficDecision(
                    target_speed=vehicle.speed,
                    target_lane=target_lane,
                    behavior_mode=BehaviorMode.LANE_CHANGE_ABORT,
                    reason="target_lane_closed_during_maneuver",
                )
            # Bound the next relative position so a one-step integration
            # cannot pass through the detected lead vehicle.
            if snapshot.front_center.vehicle_id is not None and snapshot.front_center.gap >= 0.0:
                lead_speed = vehicle.speed - snapshot.front_center.relative_speed
                kinematic_cap = max(
                    0.0,
                    lead_speed + max(0.0, snapshot.front_center.gap - 1.0) / max(time_step, 1e-6),
                )
                if decision.target_speed > kinematic_cap:
                    decision = TrafficDecision(
                        target_speed=kinematic_cap,
                        target_lane=decision.target_lane,
                        behavior_mode=decision.behavior_mode,
                        reason="kinematic_gap_cap",
                    )
            decisions[vehicle.vehicle_id] = decision
            acceleration = max(-driver.profile.comfortable_deceleration, min(driver.profile.max_acceleration, (decision.target_speed - vehicle.speed) / max(time_step, 1e-6)))
            speed = max(0.0, vehicle.speed + acceleration * time_step)
            if snapshot.front_center.vehicle_id is not None and snapshot.front_center.gap < self.config.traffic_min_gap:
                # Spacing repair must not merely move the rear vehicle back;
                # it also needs to cap its speed at the lead speed for this
                # frame.  Otherwise a lane-change decision can accelerate
                # through the same-lane clearance corridor.
                lead_speed = max(0.0, vehicle.speed - snapshot.front_center.relative_speed)
                speed = min(speed, lead_speed)
            if relative_to_ego:
                # HighwayEnv stores traffic x in the ego-relative frame.
                x = vehicle.x - (ego.speed - speed) * time_step
            else:
                x = vehicle.x + speed * time_step
            current_y = vehicle.y if vehicle.y is not None else vehicle.lane * self.config.lane_width
            target_y = decision.target_lane * self.config.lane_width
            delta_y = target_y - current_y
            if hold_lateral:
                y = current_y
                # If the hold point is already inside another vehicle's
                # lateral envelope, move just outside that envelope before
                # waiting. This handles a rear car that has reached the edge
                # of the ego lane during an aborted lane change.
                for other in (ego, *vehicles):
                    if other.vehicle_id == vehicle.vehicle_id:
                        continue
                    other_y = other.y if other.y is not None else other.lane * self.config.lane_width
                    if abs(other.x - vehicle.x) >= max(90.0, (other.length + vehicle.length) / 2.0 + 24.0):
                        continue
                    required = (other.width + vehicle.width) / 2.0 + 1.0
                    if abs(y - other_y) < required:
                        y = other_y + required if y >= other_y else other_y - required
                lane = vehicle.lane
                mode = BehaviorMode.LANE_CHANGE_ABORT
                sprite_state = 1
            elif abs(delta_y) > 1e-6:
                step_y = min(abs(delta_y), driver.profile.lane_change_speed * max(time_step, 0.0))
                y = current_y + (step_y if delta_y > 0 else -step_y)
                lane = vehicle.lane
                mode = decision.behavior_mode if decision.behavior_mode is BehaviorMode.LANE_CHANGE_ABORT else BehaviorMode.LANE_CHANGE
                sprite_state = 0 if delta_y < 0 else 2
                if abs(target_y - y) <= 1e-6:
                    y = target_y
                    lane = decision.target_lane
            else:
                y = target_y
                lane = decision.target_lane
                mode = decision.behavior_mode
                sprite_state = 1
            # Do not push a lead vehicle away from the ego bumper here.  The
            # environment owns ego collision detection and must see the true
            # integrated position; clamping x to a clearance value made a
            # direct impact look like the lead car was being pushed forever
            # while no collision event was emitted.
            updated.append(replace(vehicle, x=x, y=y, lane=lane, speed=speed, target_lane=decision.target_lane, desired_speed=decision.target_speed, acceleration=acceleration, mode=mode, sprite_state=sprite_state))
        updated.sort(key=lambda item: (item.lane, -item.x))
        if scenario in {"random", "mixed"}:
            # Random traffic is a safety-controlled stream. Two independent
            # lane changes can otherwise cross between substeps even though
            # both endpoint gaps were clear. Revert the active lateral
            # maneuver to its last valid lateral position and let it retry on
            # a later frame; this prevents a visual side impact without
            # teleporting either vehicle longitudinally.
            original_by_id = {item.vehicle_id: item for item in vehicles}
            for _ in range(2):
                repaired = False
                for first_index, first in enumerate(updated):
                    for second_index in range(first_index + 1, len(updated)):
                        second = updated[second_index]
                        longitudinal = abs(first.x - second.x) < (first.length + second.length) / 2.0
                        first_y = first.y if first.y is not None else first.lane * self.config.lane_width
                        second_y = second.y if second.y is not None else second.lane * self.config.lane_width
                        lateral = abs(first_y - second_y) < (first.width + second.width) / 2.0
                        if not (longitudinal and lateral):
                            continue
                        active = [
                            (first_index, first),
                            (second_index, second),
                        ]
                        moving = [
                            item for item in active
                            if abs(
                                (item[1].y if item[1].y is not None else item[1].lane * self.config.lane_width)
                                - (
                                    original_by_id[item[1].vehicle_id].y
                                    if original_by_id[item[1].vehicle_id].y is not None
                                    else original_by_id[item[1].vehicle_id].lane * self.config.lane_width
                                )
                            ) > 1e-6
                        ]
                        if not moving:
                            moving = [
                                item for item in active
                                if abs(
                                    (item[1].y if item[1].y is not None else item[1].lane * self.config.lane_width)
                                    - item[1].lane * self.config.lane_width
                                ) > 1e-6
                            ]
                        if not moving:
                            continue
                        restore_index, restore = max(
                            moving,
                            key=lambda item: abs(
                                (item[1].y if item[1].y is not None else item[1].lane * self.config.lane_width)
                                - item[1].lane * self.config.lane_width
                            ),
                        )
                        original = original_by_id[restore.vehicle_id]
                        updated[restore_index] = replace(
                            restore,
                            y=original.lane * self.config.lane_width,
                            lane=original.lane,
                            target_lane=original.lane,
                            mode=BehaviorMode.LANE_CHANGE_ABORT,
                            sprite_state=1,
                        )
                        repaired = True
                if not repaired:
                    break
        # Resolve same-lane penetration deterministically after integration.
        for index in range(1, len(updated)):
            lead, follower = updated[index - 1], updated[index]
            if lead.lane == follower.lane:
                # The minimum longitudinal gap must include the actual sprite
                # extents. A fixed 72 px gap is smaller than a bus plus a car,
                # so it allowed their collision rectangles to overlap while
                # the spacing pass still considered them safe.
                if relative_to_ego:
                    minimum = max(
                        self.config.traffic_min_gap,
                        (lead.length + follower.length) / 2.0 + 1.0,
                    )
                else:
                    minimum = self.config.traffic_min_gap + (lead.length + follower.length) / 2.0
                if lead.x - follower.x < minimum:
                    if not relative_to_ego:
                        updated[index] = replace(
                            follower,
                            x=lead.x - minimum,
                            speed=min(follower.speed, lead.speed),
                        )
                        continue
                    geometric_gap = lead.x - (lead.length + follower.length) / 2.0 - follower.x
                    if geometric_gap < 0.0:
                        if scenario in {"random", "mixed"}:
                            # Random traffic is safety-controlled: resolve a
                            # discrete substep that would penetrate the lead
                            # vehicle before it becomes a collision event.
                            updated[index] = replace(
                                follower,
                                x=lead.x - (lead.length + follower.length) / 2.0,
                                speed=min(follower.speed, lead.speed),
                            )
                            continue
                        updated[index - 1] = replace(
                            lead, crashed=True, speed=0.0, desired_speed=0.0,
                            acceleration=0.0, mode=BehaviorMode.CRASHED,
                        )
                        updated[index] = replace(
                            follower, crashed=True, speed=0.0, desired_speed=0.0,
                            acceleration=0.0, mode=BehaviorMode.CRASHED,
                        )
                    else:
                        # Brake the follower, but never teleport it away from
                        # the impact point. The environment records any actual
                        # overlap as a collision event.
                        updated[index] = replace(follower, speed=min(follower.speed, lead.speed))
        return TrafficStepResult(tuple(updated), decisions)

    def _geometry(self, vehicle: VehicleState) -> VehicleGeometry:
        lane = int(vehicle.lane)
        return VehicleGeometry(vehicle.vehicle_id, vehicle.x, vehicle.y if vehicle.y is not None else lane * self.config.lane_width, lane, vehicle.length, vehicle.width, vehicle.speed, vehicle.mode)
