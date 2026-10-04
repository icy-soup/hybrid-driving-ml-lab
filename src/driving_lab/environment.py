"""Deterministic, Pygame-free highway simulation used by all policies."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import EnvironmentConfig
from .events import EpisodeDiagnostics, classify_collision
from .physics import time_to_collision
from .perception import VehicleGeometry
from .rewards import EpisodeEvents, compute_reward
from .scenarios import ScenarioSampler
from .types import Action, ActionType, Observation, Transition, VehicleState


class HighwayEnv:
    """A small ego-centric highway environment for controlled experiments."""

    BASE_SCENARIOS = ("empty", "slow_lead", "sudden_brake", "obstacle", "rear_approach", "random")
    SCENARIOS = BASE_SCENARIOS + ("mixed",)

    def __init__(self, config: EnvironmentConfig | None = None):
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng()
        self.scenario = "empty"
        self.step_count = 0
        self.ego = VehicleState("ego", self.config.lanes // 2, 0.0, self.config.ego_start_speed)
        self.vehicles: list[VehicleState] = []
        self.episode_events = EpisodeEvents()
        self.diagnostics = EpisodeDiagnostics(self.scenario)
        self._previous_vehicle_x: dict[str, float] = {}
        self._overtaken_vehicle_ids: set[str] = set()

    def reset(self, seed: int | None = None, scenario: str = "random") -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if scenario not in self.SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if scenario == "mixed":
            scenario = ScenarioSampler().sample(self.rng)
        self.scenario = scenario
        self.step_count = 0
        self.ego = VehicleState("ego", self.config.lanes // 2, 0.0, self.config.ego_start_speed)
        self.vehicles = self._make_scenario(scenario)
        self.episode_events = EpisodeEvents()
        self.diagnostics = EpisodeDiagnostics(scenario, seed)
        self._previous_vehicle_x = {vehicle.vehicle_id: vehicle.x for vehicle in self.vehicles}
        self._overtaken_vehicle_ids = set()
        return self.observe()

    def observe(self) -> Observation:
        front_distance = [math.inf] * 3
        front_speed: list[float | None] = [None] * 3
        ttc = [math.inf] * 3
        rear_distance = [math.inf] * 3
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

        lane = self.ego.lane
        return Observation(
            speed=-self.ego.speed,
            lane=lane,
            front_distance=tuple(front_distance),
            front_speed=tuple(front_speed),
            ttc=tuple(ttc),
            rear_distance=tuple(rear_distance),
            crashed_front=tuple(crashed_front),
            left_rear_present=lane > 0 and rear_distance[lane - 1] < 160.0,
            right_rear_present=lane < 2 and rear_distance[lane + 1] < 160.0,
        )

    def step(self, action: Action, *, shield_override: bool = False) -> Transition:
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        observation = self.observe()
        previous_lane = self.ego.lane
        previous_vehicle_x = dict(self._previous_vehicle_x)
        for vehicle in self.vehicles:
            previous_vehicle_x.setdefault(vehicle.vehicle_id, vehicle.x)
        self._apply_action(action)
        self._advance_traffic()
        self.step_count += 1

        collision = any(
            vehicle.lane == self.ego.lane and abs(vehicle.x) <= self.config.collision_distance
            for vehicle in self.vehicles
        )
        collision_events = []
        for vehicle in self.vehicles:
            if vehicle.lane == self.ego.lane and abs(vehicle.x) <= self.config.collision_distance:
                ego_geometry = VehicleGeometry("ego", 0.0, self.ego.lane * self.config.lane_width, self.ego.lane, self.ego.length, self.ego.width, self.ego.speed)
                traffic_geometry = VehicleGeometry(vehicle.vehicle_id, vehicle.x, vehicle.lane * self.config.lane_width, vehicle.lane, vehicle.length, vehicle.width, vehicle.speed, vehicle.mode)
                collision_events.append(classify_collision(ego_geometry, traffic_geometry, step=self.step_count, scenario=self.scenario))
        for event in collision_events:
            self.diagnostics.record_collision(event)
        self.diagnostics.record_step(self.step_count, {"speed": self.ego.speed, "ttc": observation.ttc[observation.lane], "collision": collision})
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
        info = {
            "collision": collision,
            "scenario": self.scenario,
            "step": self.step_count,
            "speed": self.ego.speed,
            "shield_override": shield_override,
            "events": self.episode_events.as_dict(),
            "reward_components": breakdown.components,
            "collision_events": [event.__dict__ for event in collision_events],
            "vehicle_snapshot": [vehicle.__dict__ for vehicle in self.vehicles],
            "diagnostics": self.diagnostics.summary(),
        }
        return Transition(observation, action, breakdown.total, next_observation, done, info)

    def _count_overtakes(self, previous_vehicle_x: dict[str, float]) -> int:
        count = 0
        for vehicle in self.vehicles:
            previous_x = previous_vehicle_x.get(vehicle.vehicle_id)
            if (
                previous_x is not None
                and previous_x > 0.0
                and vehicle.x <= 0.0
                and vehicle.vehicle_id not in self._overtaken_vehicle_ids
            ):
                self._overtaken_vehicle_ids.add(vehicle.vehicle_id)
                count += 1
        self._previous_vehicle_x = {vehicle.vehicle_id: vehicle.x for vehicle in self.vehicles}
        return count

    def _apply_action(self, action: Action) -> None:
        if action.kind is ActionType.LANE_LEFT:
            self.ego = VehicleState(
                self.ego.vehicle_id,
                max(0, self.ego.lane - 1),
                self.ego.x,
                self.ego.speed,
                self.ego.crashed,
            )
        elif action.kind is ActionType.LANE_RIGHT:
            self.ego = VehicleState(
                self.ego.vehicle_id,
                min(self.config.lanes - 1, self.ego.lane + 1),
                self.ego.x,
                self.ego.speed,
                self.ego.crashed,
            )
        elif action.kind is ActionType.ACCELERATE:
            self.ego = self._replace_ego_speed(
                min(self.config.max_forward_speed, self.ego.speed + self.config.acceleration_step)
            )
        elif action.kind is ActionType.BRAKE:
            self.ego = self._replace_ego_speed(
                max(self.config.min_forward_speed, self.ego.speed - self.config.braking_step)
            )

    def _replace_ego_speed(self, speed: float) -> VehicleState:
        return VehicleState(self.ego.vehicle_id, self.ego.lane, self.ego.x, speed, self.ego.crashed)

    def _advance_traffic(self) -> None:
        updated: list[VehicleState] = []
        for vehicle in self.vehicles:
            speed = vehicle.speed
            if self.scenario == "sudden_brake" and vehicle.vehicle_id == "lead" and self.step_count >= 3:
                speed = max(self.config.min_forward_speed / 2.0, speed - 2.0)
            updated.append(
                VehicleState(
                    vehicle.vehicle_id,
                    vehicle.lane,
                    vehicle.x - (self.ego.speed - speed),
                    speed,
                    vehicle.crashed,
                )
            )
        updated = self._resolve_traffic_spacing(updated)
        self.vehicles = [
            vehicle
            for vehicle in updated
            if -self.config.detection_distance <= vehicle.x <= self.config.detection_distance
        ]

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
                minimum_rear_x = front.x - self.config.traffic_min_gap
                if rear.x > minimum_rear_x:
                    resolved[rear_index] = VehicleState(
                        rear.vehicle_id,
                        rear.lane,
                        minimum_rear_x,
                        min(rear.speed, front.speed),
                        rear.crashed,
                    )
        return resolved

    def _make_scenario(self, scenario: str) -> list[VehicleState]:
        lane = self.config.lanes // 2
        if scenario == "empty":
            return []
        if scenario == "slow_lead":
            return [VehicleState("lead", lane, 280.0, 4.0)]
        if scenario == "sudden_brake":
            return [VehicleState("lead", lane, 260.0, self.config.ego_start_speed)]
        if scenario == "obstacle":
            return [VehicleState("obstacle", lane, 220.0, 0.0, crashed=True)]
        if scenario == "rear_approach":
            return [VehicleState("rear", lane, -120.0, 14.0)]
        return self._make_random_traffic()

    def _make_random_traffic(self) -> list[VehicleState]:
        vehicles: list[VehicleState] = []
        count = int(self.rng.integers(4, 9))
        for index in range(count):
            for _ in range(100):
                lane = int(self.rng.integers(0, self.config.lanes))
                x = float(self.rng.integers(-250, 520))
                if abs(x) < 80:
                    x += 180.0
                if all(
                    vehicle.lane != lane
                    or abs(vehicle.x - x) >= self.config.traffic_min_gap
                    for vehicle in vehicles
                ):
                    break
            speed = float(self.rng.uniform(4.0, 16.0))
            vehicles.append(VehicleState(f"car-{index}", lane, x, speed))
        return vehicles
