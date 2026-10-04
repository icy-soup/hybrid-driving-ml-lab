"""Discrete short-horizon model predictive controller."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..types import Action, ActionType, Observation


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 5
    candidate_actions: tuple[int, ...] = (0, 1, 2, 3, 4)
    collision_cost: float = 100.0
    progress_cost: float = 1.0
    speed_cost: float = 0.1
    lane_change_cost: float = 0.2


class MPCPolicy:
    def __init__(self, env_model=None, config: MPCConfig | None = None) -> None:
        self.env_model = env_model
        self.config = config or MPCConfig()
        if self.config.horizon < 1:
            raise ValueError("horizon must be positive")
        self.last_plan: list[Action] = []

    def act(self, observation: Observation) -> Action:
        actions = [Action(ActionType(index)) for index in self.config.candidate_actions]
        best_sequence = None
        best_score = float("inf")
        # Branching stays deliberately small for the discrete baseline.
        for sequence in product(actions, repeat=self.config.horizon):
            score = self._score(sequence, observation)
            if score < best_score:
                best_score, best_sequence = score, sequence
        self.last_plan = list(best_sequence or (actions[-1],))
        return self.last_plan[0]

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
            score += self.config.speed_cost * abs(speed - 12.0)
            if ttc < 3.0:
                ttc = max(0.0, ttc - 0.5)
        return score
