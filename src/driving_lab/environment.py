"""Deterministic, Pygame-free highway simulation used by all policies."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

import numpy as np

from .config import EnvironmentConfig
from .events import EpisodeDiagnostics, classify_collision
from .physics import (
    emergency_braking_distance,
    lane_change_front_gap,
    lane_change_rear_gap,
    time_to_collision,
)
from .perception import VehicleGeometry
from .traffic import TrafficWorld
from .rewards import EpisodeEvents, compute_reward
from .scenarios import ScenarioSampler
from .types import Action, ActionType, BehaviorMode, Observation, Transition, VehicleState


class HighwayEnv:
    """A small ego-centric highway environment for controlled experiments."""

    BASE_SCENARIOS = ("empty", "slow_lead", "sudden_brake", "obstacle", "rear_approach", "random")
    SCENARIOS = BASE_SCENARIOS + ("mixed",)

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng()
        self.scenario = "empty"
        self.step_count = 0
        self.distance_travelled = 0.0
        initial_lane = self.config.lanes // 2
        self.ego = VehicleState(
            "ego", initial_lane, 0.0, self.config.ego_start_speed,
            y=initial_lane * self.config.lane_width,
            vehicle_type="跑车", color="银色",
        )
        self.vehicles: list[VehicleState] = []
        self.traffic_world = TrafficWorld(self.config)
        self.episode_events = EpisodeEvents()
        self.diagnostics = EpisodeDiagnostics(self.scenario)
        self._previous_vehicle_x: dict[str, float] = {}
        self._overtaken_vehicle_ids: set[str] = set()
        self._legacy_last_forward_step = 0
        self._legacy_last_rear_step = 0
        self._legacy_vehicle_serial = 0
        self._last_traffic_decisions = {}
        self._recorded_collision_pairs: set[tuple[str, str]] = set()
        self.last_impact_speed: float | None = None
        self.last_impact_ttc: float | None = None
        self.last_action_name: str | None = None
        self._lane_change_spawn_lock = 0

    def reset(self, seed: int | None = None, scenario: str = "random") -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if scenario not in self.SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if scenario == "mixed":
            scenario = ScenarioSampler().sample(self.rng)
        self.scenario = scenario
        self.step_count = 0
        self.distance_travelled = 0.0
        initial_lane = self.config.lanes // 2
        self.ego = VehicleState(
            "ego", initial_lane, 0.0, self.config.ego_start_speed,
            y=initial_lane * self.config.lane_width,
            vehicle_type="跑车", color="银色",
        )
        self.vehicles = self._make_scenario(scenario)
        self.traffic_world = TrafficWorld(self.config)
        self.episode_events = EpisodeEvents()
        self.diagnostics = EpisodeDiagnostics(scenario, seed)
        self._previous_vehicle_x = {vehicle.vehicle_id: vehicle.x for vehicle in self.vehicles}
        self._overtaken_vehicle_ids = set()
        self._legacy_last_forward_step = 0
        self._legacy_last_rear_step = 0
        self._legacy_vehicle_serial = len(self.vehicles)
        self._last_traffic_decisions = {}
        self._recorded_collision_pairs = set()
        self.last_impact_speed = None
        self.last_impact_ttc = None
        self.last_action_name = None
        self._lane_change_spawn_lock = 0
        return self.observe()

    def observe(self) -> Observation:
        front_distance = [math.inf] * 3
        front_speed: list[float | None] = [None] * 3
        ttc = [math.inf] * 3
        rear_distance = [math.inf] * 3
        rear_speed: list[float | None] = [None] * 3
        crashed_front = [False] * 3

        for vehicle in self.vehicles:
            if vehicle.lane < 0 or vehicle.lane >= 3:
                continue
            if vehicle.x >= 0:
                if vehicle.x < front_distance[vehicle.lane]:
                    front_distance[vehicle.lane] = vehicle.x
                    front_speed[vehicle.lane] = -vehicle.speed
                    crashed_front[vehicle.lane] = vehicle.crashed
                    ttc[vehicle.lane] = time_to_collision(
                        vehicle.x,
                        self.ego.speed - vehicle.speed,
                    )
            elif abs(vehicle.x) < rear_distance[vehicle.lane]:
                rear_distance[vehicle.lane] = abs(vehicle.x)
                rear_speed[vehicle.lane] = vehicle.speed

        lane = self.ego.lane
        return Observation(
            speed=-self.ego.speed,
            lane=lane,
            front_distance=tuple(front_distance),
            front_speed=tuple(front_speed),
            ttc=tuple(ttc),
            rear_distance=tuple(rear_distance),
            rear_speed=tuple(rear_speed),
            crashed_front=tuple(crashed_front),
            left_rear_present=lane > 0 and rear_distance[lane - 1] < 160.0,
            right_rear_present=lane < 2 and rear_distance[lane + 1] < 160.0,
        )

    def step(self, action: Action, *, shield_override: bool = False) -> Transition:
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        if self.ego.crashed:
            observation = self.observe()
            return Transition(
                observation,
                action,
                0.0,
                observation,
                True,
                {
                    "collision": True,
                    "scenario": self.scenario,
                    "step": self.step_count,
                    "speed": 0.0,
                    "distance": self.distance_travelled,
                    "shield_override": False,
                    "events": self.episode_events.as_dict(),
                    "reward_components": {},
                    "collision_events": [],
                    "vehicle_snapshot": [_vehicle_log(vehicle) for vehicle in self.vehicles],
                    "traffic_decisions": {},
                    "diagnostics": self.diagnostics.summary(),
                },
            )
        observation = self.observe()
        self.last_action_name = action.kind.name
        self.last_impact_speed = None
        self.last_impact_ttc = None
        previous_lane = self.ego.lane
        previous_vehicle_x = dict(self._previous_vehicle_x)
        for vehicle in self.vehicles:
            previous_vehicle_x.setdefault(vehicle.vehicle_id, vehicle.x)
        self._apply_action(action)
        self._advance_ego_lateral()
        self._advance_traffic()
        self.step_count += 1

        traffic_collision_events = self._resolve_traffic_collisions()
        # Record a real overlap before applying the random-road spacing repair.
        # Repairing first hid traffic-traffic impacts from diagnostics and
        # made the surviving vehicle appear to be pushing the crashed one.
        self._resolve_random_penetrations()

        ego_y = self.ego.y if self.ego.y is not None else self.ego.lane * self.config.lane_width

        def ego_longitudinal_contact(vehicle: VehicleState) -> bool:
            limit = min(
                self.config.collision_distance,
                (vehicle.length + self.ego.length) / 2.0,
            )
            previous_x = previous_vehicle_x.get(vehicle.vehicle_id, vehicle.x)
            # Test the complete swept segment. Checking only the final x let a
            # fast vehicle jump from one side of the ego to the other between
            # substeps and visually pass through the ego car.
            return min(previous_x, vehicle.x) <= limit and max(previous_x, vehicle.x) >= -limit

        def ego_lateral_contact(vehicle: VehicleState) -> bool:
            return abs(
                (vehicle.y if vehicle.y is not None else vehicle.lane * self.config.lane_width)
                - ego_y
            ) < (vehicle.width + self.ego.width) / 2.0

        collision = any(
            ego_longitudinal_contact(vehicle) and ego_lateral_contact(vehicle)
            for vehicle in self.vehicles
        )
        if collision:
            self.last_impact_speed = float(self.ego.speed)
            self.last_impact_ttc = float(observation.ttc[observation.lane])
        collision_events = list(traffic_collision_events)
        for vehicle in self.vehicles:
            if ego_longitudinal_contact(vehicle) and ego_lateral_contact(vehicle):
                pair = tuple(sorted(("ego", vehicle.vehicle_id)))
                if pair in self._recorded_collision_pairs:
                    continue
                ego_geometry = VehicleGeometry("ego", 0.0, self.ego.lane * self.config.lane_width, self.ego.lane, self.ego.length, self.ego.width, self.ego.speed)
                limit = min(self.config.collision_distance, (vehicle.length + self.ego.length) / 2.0)
                impact_x = max(-limit, min(limit, vehicle.x))
                traffic_geometry = VehicleGeometry(vehicle.vehicle_id, impact_x, vehicle.lane * self.config.lane_width, vehicle.lane, vehicle.length, vehicle.width, vehicle.speed, vehicle.mode)
                collision_events.append(classify_collision(ego_geometry, traffic_geometry, step=self.step_count, scenario=self.scenario))
                self._recorded_collision_pairs.add(pair)
        if collision:
            self.ego = replace(
                self.ego,
                crashed=True,
                speed=0.0,
                desired_speed=0.0,
                acceleration=0.0,
                target_lane=None,
                mode=BehaviorMode.CRASHED,
                explosion_frames=self.config.explosion_frames,
            )
            self.vehicles = [
                replace(
                    vehicle,
                    x=self._previous_vehicle_x.get(vehicle.vehicle_id, vehicle.x),
                    crashed=True,
                    speed=0.0,
                    desired_speed=0.0,
                    acceleration=0.0,
                    mode=BehaviorMode.CRASHED,
                    explosion_frames=self.config.explosion_frames,
                )
                if ego_longitudinal_contact(vehicle) and ego_lateral_contact(vehicle) else vehicle
                for vehicle in self.vehicles
            ]
        for event in collision_events:
            self.diagnostics.record_collision(event)
        traffic_pair_collision = any(
            event.vehicle_a != "ego" and event.vehicle_b != "ego"
            for event in collision_events
        )
        done = collision or self.step_count >= self.config.max_steps
        next_observation = self.observe()
        lane_changed = self.ego.lane != previous_lane
        overtakes = self._count_overtakes(previous_vehicle_x)
        current_ttc = observation.ttc[observation.lane]
        ttc_warning = current_ttc < self.config.reward.ttc_threshold
        self.episode_events.overtakes += overtakes
        self.episode_events.lane_changes += int(lane_changed)
        self.episode_events.ttc_warnings += int(ttc_warning)
        self.episode_events.shield_overrides += int(shield_override)
        self.episode_events.collisions += int(collision)
        breakdown = compute_reward(
            self.config.reward,
            speed=-self.ego.speed,
            ttc=current_ttc,
            overtake=overtakes > 0,
            lane_change=lane_changed,
            collision=collision,
            shield_override=shield_override,
        )
        self.distance_travelled += abs(self.ego.speed)
        self.diagnostics.record_step(
            self.step_count,
            {
                "speed": self.ego.speed,
                "ttc": current_ttc,
                "collision": collision,
                "reward": breakdown.total,
                "distance": self.distance_travelled,
                "reward_components": breakdown.components,
                "vehicles": [_vehicle_log(vehicle) for vehicle in self.vehicles],
            },
        )
        info = {
            "collision": collision,
            "scenario": self.scenario,
            "step": self.step_count,
            "speed": self.ego.speed,
            "impact_speed": self.last_impact_speed,
            "impact_ttc": self.last_impact_ttc,
            "action_before_collision": self.last_action_name,
            "distance": self.distance_travelled,
            "shield_override": shield_override,
            "events": self.episode_events.as_dict(),
            "reward_components": breakdown.components,
            "collision_events": [event.__dict__ for event in collision_events],
            "vehicle_snapshot": [_vehicle_log(vehicle) for vehicle in self.vehicles],
            "traffic_decisions": {
                vehicle_id: {
                    "target_speed": decision.target_speed,
                    "target_lane": decision.target_lane,
                    "behavior_mode": decision.behavior_mode.name,
                    "reason": decision.reason,
                }
                for vehicle_id, decision in self._last_traffic_decisions.items()
            },
            "diagnostics": self.diagnostics.summary(),
        }
        return Transition(observation, action, breakdown.total, next_observation, done, info)

    def _resolve_traffic_collisions(self):
        """Freeze vehicles at impact instead of integrating them through one another."""
        events = []
        crashed_ids: set[str] = set()
        for index, first in enumerate(self.vehicles):
            for second in self.vehicles[index + 1 :]:
                longitudinal = abs(first.x - second.x)
                lateral = abs((first.y if first.y is not None else first.lane * self.config.lane_width) - (second.y if second.y is not None else second.lane * self.config.lane_width))
                if longitudinal < (first.length + second.length) / 2.0 and lateral < (first.width + second.width) / 2.0:
                    pair = tuple(sorted((first.vehicle_id, second.vehicle_id)))
                    if pair in self._recorded_collision_pairs:
                        continue
                    first_geometry = VehicleGeometry(first.vehicle_id, first.x, first.y or first.lane * self.config.lane_width, first.lane, first.length, first.width, first.speed, first.mode)
                    second_geometry = VehicleGeometry(second.vehicle_id, second.x, second.y or second.lane * self.config.lane_width, second.lane, second.length, second.width, second.speed, second.mode)
                    events.append(classify_collision(first_geometry, second_geometry, step=self.step_count, scenario=self.scenario))
                    self._recorded_collision_pairs.add(pair)
                    crashed_ids.update((first.vehicle_id, second.vehicle_id))
        if crashed_ids:
            self.vehicles = [
                replace(vehicle, crashed=True, speed=0.0, desired_speed=0.0, acceleration=0.0, mode=BehaviorMode.CRASHED, explosion_frames=self.config.explosion_frames)
                if vehicle.vehicle_id in crashed_ids else vehicle
                for vehicle in self.vehicles
            ]
            self.vehicles = [
                replace(vehicle, x=self._previous_vehicle_x.get(vehicle.vehicle_id, vehicle.x))
                if vehicle.vehicle_id in crashed_ids else vehicle
                for vehicle in self.vehicles
            ]
        return events

    def _resolve_random_penetrations(self) -> None:
        """Keep random traffic safety controlled before event classification."""
        if self.scenario not in {"random", "mixed"}:
            return
        resolved = list(self.vehicles)
        ego_clearance = max(self.config.collision_distance + 8.0, (self.config.traffic_min_gap + 10.0) / 2.0)
        for _ in range(max(4, len(resolved) * 2)):
            changed = False
            for index, first in enumerate(resolved):
                for other_index in range(index + 1, len(resolved)):
                    second = resolved[other_index]
                    if first.crashed or second.crashed:
                        continue
                    lateral = abs(
                        (first.y if first.y is not None else first.lane * self.config.lane_width)
                        - (second.y if second.y is not None else second.lane * self.config.lane_width)
                    ) < (first.width + second.width) / 2.0
                    if not lateral or abs(first.x - second.x) >= (first.length + second.length) / 2.0:
                        continue
                    front_index, rear_index = (index, other_index) if first.x >= second.x else (other_index, index)
                    front, rear = resolved[front_index], resolved[rear_index]
                    half_extent = (front.length + rear.length) / 2.0 + 1.0
                    if 0.0 <= rear.x < ego_clearance:
                        # Preserve the ego bumper clearance and move the
                        # leader forward by the minimum amount instead.
                        resolved[front_index] = replace(front, x=rear.x + half_extent)
                    else:
                        resolved[rear_index] = replace(
                            rear,
                            x=front.x - half_extent,
                            speed=min(rear.speed, front.speed),
                        )
                    changed = True
            if not changed:
                break
        # Finish with a deterministic lane-wise pass. Pairwise repair can
        # otherwise oscillate when several traffic vehicles overlap. The ego
        # is deliberately not used as a spacing anchor: an overlap with ego
        # must remain visible to the collision detector.
        for lane in range(self.config.lanes):
            lane_indices = [i for i, vehicle in enumerate(resolved) if vehicle.lane == lane]
            forward = sorted((i for i in lane_indices if resolved[i].x >= 0.0), key=lambda i: resolved[i].x)
            previous_x = None
            previous_length = None
            for i in forward:
                vehicle = resolved[i]
                if previous_x is not None and previous_length is not None:
                    minimum_x = previous_x + (previous_length + vehicle.length) / 2.0 + 1.0
                    new_x = max(vehicle.x, minimum_x)
                    resolved[i] = replace(vehicle, x=new_x)
                    previous_x = new_x
                    previous_length = vehicle.length
                else:
                    previous_x = vehicle.x
                    previous_length = vehicle.length
            rear = sorted((i for i in lane_indices if resolved[i].x < 0.0), key=lambda i: resolved[i].x, reverse=True)
            previous_x = None
            previous_length = None
            for i in rear:
                vehicle = resolved[i]
                if previous_x is not None and previous_length is not None:
                    maximum_x = previous_x - (previous_length + vehicle.length) / 2.0 - 1.0
                    new_x = min(vehicle.x, maximum_x)
                    resolved[i] = replace(vehicle, x=new_x)
                    previous_x = new_x
                    previous_length = vehicle.length
                else:
                    previous_x = vehicle.x
                    previous_length = vehicle.length
        self.vehicles = resolved

    def _count_overtakes(self, previous_vehicle_x: dict[str, float]) -> int:
        count = 0
        for vehicle in self.vehicles:
            previous_x = previous_vehicle_x.get(vehicle.vehicle_id)
            if (
                previous_x is not None
                and previous_x > 0.0
                and vehicle.x <= 5.0
                and vehicle.vehicle_id not in self._overtaken_vehicle_ids
            ):
                self._overtaken_vehicle_ids.add(vehicle.vehicle_id)
                count += 1
        self._previous_vehicle_x = {vehicle.vehicle_id: vehicle.x for vehicle in self.vehicles}
        return count

    def _apply_action(self, action: Action) -> None:
        if action.kind is ActionType.LANE_LEFT:
            if self.ego.target_lane is None:
                target_lane = max(0, self.ego.lane - 1)
                if target_lane != self.ego.lane:
                    self.ego = replace(
                        self.ego,
                        target_lane=target_lane,
                        mode=BehaviorMode.LANE_CHANGE,
                        sprite_state=0,
                    )
        elif action.kind is ActionType.LANE_RIGHT:
            if self.ego.target_lane is None:
                target_lane = min(self.config.lanes - 1, self.ego.lane + 1)
                if target_lane != self.ego.lane:
                    self.ego = replace(
                        self.ego,
                        target_lane=target_lane,
                        mode=BehaviorMode.LANE_CHANGE,
                        sprite_state=2,
                    )
        elif action.kind is ActionType.ACCELERATE:
            self.ego = self._replace_ego_speed(
                min(self.config.max_forward_speed, self.ego.speed + self.config.acceleration_step)
            )
        elif action.kind is ActionType.BRAKE:
            self.ego = self._replace_ego_speed(
                # Braking is a control action, not normal cruising.  It must
                # be able to reach the lead vehicle's speed or stop before a
                # stationary obstacle; the minimum forward speed only applies
                # to ordinary traffic generation/cruising.
                max(0.0, self.ego.speed - self.config.braking_step)
            )

    def _replace_ego_speed(self, speed: float) -> VehicleState:
        return replace(self.ego, speed=speed)

    def _advance_ego_lateral(self) -> None:
        """Animate an ego lane change using the legacy 6 px/frame motion."""

        target_lane = self.ego.target_lane
        current_y = self.ego.y if self.ego.y is not None else self.ego.lane * self.config.lane_width
        if target_lane is None or target_lane == self.ego.lane:
            self.ego = replace(
                self.ego,
                y=float(self.ego.lane * self.config.lane_width),
                target_lane=None,
                mode=BehaviorMode.CRUISE if not self.ego.crashed else BehaviorMode.CRASHED,
                sprite_state=1,
            )
            return
        # The paper's lane-change pause condition is needed here: a target
        # lane can become occupied after the maneuver starts. Re-check the
        # predicted front and rear windows before moving another lateral step.
        # Cancelling early keeps the ego in its original lane so the policy can
        # choose a different escape lane on the next decision.
        if not self._target_lane_remains_safe(target_lane):
            self.ego = replace(
                self.ego,
                y=float(self.ego.lane * self.config.lane_width),
                target_lane=None,
                mode=BehaviorMode.LANE_CHANGE_ABORT,
                sprite_state=1,
            )
            return
        target_y = float(target_lane * self.config.lane_width)
        delta = target_y - current_y
        step = min(abs(delta), 6.0)
        next_y = current_y + (step if delta > 0.0 else -step)
        if abs(target_y - next_y) <= 1e-6:
            self.ego = replace(
                self.ego,
                lane=target_lane,
                y=target_y,
                target_lane=None,
                mode=BehaviorMode.CRUISE,
                sprite_state=1,
            )
            self._lane_change_spawn_lock = 12
        else:
            self.ego = replace(
                self.ego,
                y=next_y,
                mode=BehaviorMode.LANE_CHANGE,
                sprite_state=0 if delta < 0.0 else 2,
            )

    def _target_lane_remains_safe(self, target_lane: int) -> bool:
        observation = self.observe()
        if target_lane < 0 or target_lane >= self.config.lanes:
            return False
        if observation.crashed_front[target_lane]:
            return False
        front_speed = observation.front_speed[target_lane]
        target_speed = abs(float(front_speed)) if front_speed is not None else abs(self.ego.speed)
        front_gap = observation.front_distance[target_lane]
        if front_gap < float("inf") and front_gap < lane_change_front_gap(self.ego.speed, target_speed):
            return False
        rear_gap = observation.rear_distance[target_lane]
        rear_speed = observation.rear_speed[target_lane]
        if rear_gap < float("inf"):
            rear_speed_value = abs(float(rear_speed)) if rear_speed is not None else abs(self.ego.speed)
            if rear_gap < lane_change_rear_gap(self.ego.speed, rear_speed_value):
                return False
        return True

    def _advance_traffic(self) -> None:
        if self._lane_change_spawn_lock > 0:
            self._lane_change_spawn_lock -= 1
        self._spawn_legacy_traffic()
        updated = list(self.vehicles)
        self._last_traffic_decisions = {}
        # Legacy Main.py advances at 60 FPS. Keep one environment step as one
        # logical frame while integrating traffic in smaller substeps so a
        # vehicle cannot jump through a collision corridor.
        for _ in range(4):
            result = self.traffic_world.step_traffic(
                updated,
                self.ego,
                scenario=self.scenario,
                time_step=0.25,
                relative_to_ego=True,
            )
            updated = list(result.vehicles)
            self._last_traffic_decisions = result.decisions
        if self.scenario == "sudden_brake" and self.step_count >= 3:
            updated = [
                replace(
                    vehicle,
                    speed=max(self.config.min_forward_speed / 2.0, vehicle.speed - 2.0)
                    if vehicle.vehicle_id == "lead" else vehicle.speed,
                )
                for vehicle in updated
            ]
        updated = self._resolve_traffic_spacing(updated)
        updated = [
            replace(vehicle, explosion_frames=max(0, vehicle.explosion_frames - 1))
            if vehicle.explosion_frames > 0 else vehicle
            for vehicle in updated
        ]
        slow = abs(self.ego.speed) < 10.0
        # Legacy boundaries are absolute screen x values. The canonical
        # environment stores x relative to the ego anchor at 455 px.
        anchor_x = 455.0
        left = (self.config.legacy_slow_remove_left if slow else self.config.legacy_remove_left) - anchor_x
        right = (self.config.legacy_slow_remove_right if slow else self.config.legacy_remove_right) - anchor_x
        self.vehicles = [vehicle for vehicle in updated if left <= vehicle.x <= right]

    def _spawn_legacy_traffic(self) -> None:
        """Spawn traffic at the legacy off-screen positions with safe gaps."""
        if self.scenario not in {"random", "mixed"}:
            return
        if len(self.vehicles) >= self.config.legacy_max_traffic:
            return

        # Legacy 800 ms at 60 FPS, with a score-like decay toward 150 ms.
        # Convert legacy absolute screen positions to the ego-relative frame.
        anchor_x = 455.0
        forward_spawn_x = self.config.legacy_forward_spawn_x - anchor_x
        rear_spawn_x = self.config.legacy_rear_spawn_x - anchor_x
        forward_interval = (
            48
            if self._legacy_last_forward_step == 0
            else max(9, int(48 / (1.0 + self.step_count * 0.04)))
        )
        if self.step_count - self._legacy_last_forward_step >= forward_interval:
            lane = self._choose_spawn_lane(forward_spawn_x, rear=False)
            if lane is not None:
                self.vehicles.append(self._new_legacy_vehicle(lane, forward_spawn_x, rear=False))
                self._legacy_last_forward_step = self.step_count

        if (
            len(self.vehicles) < self.config.legacy_max_traffic
            and self.step_count - self._legacy_last_rear_step >= 180
            and float(self.rng.random()) < 0.25
        ):
            lane = self._choose_spawn_lane(rear_spawn_x, rear=True)
            if lane is not None:
                self.vehicles.append(self._new_legacy_vehicle(lane, rear_spawn_x, rear=True))
                self._legacy_last_rear_step = self.step_count

    def _choose_spawn_lane(self, x: float, *, rear: bool = False) -> int | None:
        candidates = []
        reserved_lane = self.ego.target_lane
        for lane in range(self.config.lanes):
            # Do not inject a newly spawned vehicle into the lane currently
            # being entered.  The lane-change prediction is made before the
            # traffic spawn phase; allowing a spawn here creates a vehicle
            # that was invisible to the decision and can cut across the ego
            # during the final lateral frames.
            if reserved_lane is not None and lane == reserved_lane:
                continue
            if self._lane_change_spawn_lock > 0 and lane == self.ego.lane:
                continue
            # Do not insert a very slow forward vehicle into the ego lane when
            # the current gap is already shorter than the physical stopping
            # distance.  In relative coordinates the screen-edge spawn point
            # is still a real obstacle; allowing it here creates a collision
            # that no policy can prevent after the vehicle appears.
            if not rear and lane == self.ego.lane:
                ego_speed = max(0.0, float(self.ego.speed))
                lead_speed = max(10.0, float(self.config.min_forward_speed))
                closing = max(0.0, ego_speed - lead_speed)
                required_gap = emergency_braking_distance(
                    closing,
                    current_speed=ego_speed,
                    target_speed=lead_speed,
                    reaction_frames=2.0,
                    max_deceleration=1.0,
                    margin=self.config.collision_distance + 12.0,
                )
                if x < required_gap:
                    continue
            # Admission uses the largest legacy sprite body because the
            # vehicle type is selected only after the lane is chosen.  This
            # prevents the spacing repair from teleporting an existing NPC
            # backwards when a bus or truck enters a crowded lane.
            spawn_length = 120.0
            lane_clear = all(
                vehicle.lane != lane
                or abs(vehicle.x - x)
                >= self.config.traffic_min_gap + (spawn_length + vehicle.length) / 2.0
                for vehicle in self.vehicles
            )
            if lane_clear:
                if abs(x) >= self.config.traffic_min_gap or lane != self.ego.lane:
                    candidates.append(lane)
        return int(self.rng.choice(candidates)) if candidates else None

    def _new_legacy_vehicle(self, lane: int, x: float, *, rear: bool) -> VehicleState:
        types = (
            ("电动四轮车", ("黄色", "绿色", "黑色", "红色")),
            ("小轿车", ("蓝色", "灰色", "绿色", "黄色")),
            ("大卡车", ("红色", "蓝色", "银色")),
            ("大巴车", ("红色", "蓝色", "绿色")),
            ("面包车", ("黑色",)),
            ("卫星车", ("白色",)),
            ("跑车", ("红色", "黄色", "银色")),
        )
        vehicle_type, colors = types[self._legacy_vehicle_serial % len(types)]
        # Keep the collision body close to the sprite footprint.  The earlier
        # legacy vehicles all inherited length=10,width=4, so a 120 px bus
        # could visually overlap another vehicle without producing a contact.
        dimensions = {
            "电动四轮车": (60.0, 28.0),
            "小轿车": (70.0, 26.0),
            "大卡车": (80.0, 32.0),
            "大巴车": (120.0, 34.0),
            "面包车": (75.0, 30.0),
            "卫星车": (70.0, 30.0),
            "跑车": (80.0, 24.0),
        }
        length, width = dimensions[vehicle_type]
        color = colors[int(self.rng.integers(0, len(colors)))]
        self._legacy_vehicle_serial += 1
        if rear:
            # Rear traffic enters from the left and catches the ego vehicle.
            speed = float(self.rng.uniform(max(10.0, self.ego.speed + 1.0), max(12.0, self.ego.speed + 8.0)))
        else:
            # Forward traffic uses highway-like speeds.  It remains slower
            # than the ego when possible, but is no longer an unrealistically
            # slow 4--7 m/s obstacle that forces the ego to the speed ceiling.
            lower = max(10.0, min(self.ego.speed - 8.0, 20.0))
            upper = max(lower + 0.5, min(32.0, self.ego.speed - 0.5))
            speed = float(self.rng.uniform(lower, upper))
        return VehicleState(
            f"legacy-car-{self._legacy_vehicle_serial}", lane, x, speed,
            y=lane * self.config.lane_width,
            length=length,
            width=width,
            vehicle_type=vehicle_type, color=color, sprite_state=1,
        )

    def _resolve_traffic_spacing(self, vehicles: list[VehicleState]) -> list[VehicleState]:
        """Keep traffic vehicles from occupying the same longitudinal space."""

        resolved = list(vehicles)
        for lane in range(self.config.lanes):
            lane_indices = sorted(
                (index for index, vehicle in enumerate(resolved) if vehicle.lane == lane),
                key=lambda index: resolved[index].x,
                reverse=True,
            )
            for front_index, rear_index in zip(lane_indices, lane_indices[1:]):
                front = resolved[front_index]
                rear = resolved[rear_index]
                if front.vehicle_type is not None or rear.vehicle_type is not None:
                    # For rendered legacy vehicles, traffic_min_gap is the
                    # free space beyond the two physical half lengths.
                    minimum_rear_x = front.x - (
                        self.config.traffic_min_gap + (front.length + rear.length) / 2.0
                    )
                else:
                    # Preserve the compact analytical fixture contract used
                    # by the headless unit tests.
                    minimum_rear_x = front.x - self.config.traffic_min_gap
                rear_y = rear.y if rear.y is not None else rear.lane * self.config.lane_width
                front_y = front.y if front.y is not None else front.lane * self.config.lane_width
                lateral_overlap = abs(front_y - rear_y) <= (front.width + rear.width) / 2.0
                if rear.x > minimum_rear_x and lateral_overlap:
                    geometric_gap = front.x - (front.length + rear.length) / 2.0 - rear.x
                    if geometric_gap < 0.0:
                        if self.scenario in {"random", "mixed"}:
                            # Random traffic is safety controlled: stop the
                            # follower at the bumper instead of creating an
                            # overlap that would push both vehicles.
                            resolved[rear_index] = replace(
                                rear,
                                x=front.x - (front.length + rear.length) / 2.0,
                                speed=min(rear.speed, front.speed),
                            )
                        else:
                            # Fixed scenarios intentionally preserve impacts
                            # so their collision contract stays observable.
                            resolved[front_index] = replace(
                                front, crashed=True, speed=0.0, desired_speed=0.0,
                                acceleration=0.0, mode=BehaviorMode.CRASHED,
                            )
                            resolved[rear_index] = replace(
                                rear, crashed=True, speed=0.0, desired_speed=0.0,
                                acceleration=0.0, mode=BehaviorMode.CRASHED,
                            )
                        continue
                    resolved[rear_index] = replace(
                        rear,
                        x=minimum_rear_x,
                        speed=min(rear.speed, front.speed),
                    )
        return resolved

    def _resolve_ego_lane_spacing(self, vehicles: list[VehicleState]) -> list[VehicleState]:
        """Prevent random traffic from tunneling through the ego corridor."""
        # Keep the ego collision corridor clear without moving a vehicle by a
        # full traffic headway. The old 72 px clamp could push a car into its
        # leader; 40 px is the bumper clearance used by the runtime guard.
        safe = max(self.config.collision_distance + 8.0, 40.0)
        resolved = list(vehicles)
        same_lane = [(index, vehicle) for index, vehicle in enumerate(resolved) if vehicle.lane == self.ego.lane]
        front = min((item for item in same_lane if item[1].x >= 0.0), key=lambda item: item[1].x, default=None)
        rear = max((item for item in same_lane if item[1].x < 0.0), key=lambda item: item[1].x, default=None)
        for item, sign in ((front, 1.0), (rear, -1.0)):
            if item is None:
                continue
            index, vehicle = item
            if -safe < vehicle.x < safe:
                resolved[index] = replace(
                    vehicle,
                    x=sign * safe,
                    speed=min(vehicle.speed, self.ego.speed),
                )
        return resolved

    def _make_scenario(self, scenario: str) -> list[VehicleState]:
        lane = self.config.lanes // 2
        if scenario == "empty":
            return []
        if scenario == "slow_lead":
            return [
                VehicleState("lead", lane, 280.0, 4.0, vehicle_type="小轿车", color="蓝色"),
                VehicleState("traffic-follower", lane, -80.0, 14.0, vehicle_type="大卡车", color="红色"),
            ]
        if scenario == "sudden_brake":
            return [VehicleState("lead", lane, 260.0, self.config.ego_start_speed, vehicle_type="大巴车", color="红色")]
        if scenario == "obstacle":
            return [VehicleState("obstacle", lane, 220.0, 0.0, crashed=True, vehicle_type="大卡车", color="银色")]
        if scenario == "rear_approach":
            return [VehicleState("rear", lane, -120.0, 14.0, vehicle_type="跑车", color="红色")]
        return self._make_random_traffic()

    def _make_random_traffic(self) -> list[VehicleState]:
        # The legacy game starts with an empty road. Traffic enters through
        # the lifecycle spawner above, from outside the left or right edge.
        return []


def _vehicle_log(vehicle: VehicleState) -> dict[str, object]:
    """Stable human-readable vehicle record for JSONL diagnostics/UI."""
    record = dict(vehicle.__dict__)
    record["mode"] = vehicle.mode.name
    return record
