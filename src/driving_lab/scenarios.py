"""Deterministic scenario sampling for baseline and future RL experiments."""

from __future__ import annotations

import numpy as np


class ScenarioSampler:
    def __init__(self, scenarios: tuple[str, ...] | None = None):
        self.scenarios = scenarios or (
            "empty",
            "slow_lead",
            "sudden_brake",
            "obstacle",
            "rear_approach",
            "random",
        )

    def sample(self, rng: np.random.Generator) -> str:
        return str(rng.choice(self.scenarios))

    def schedule(self, *, seed: int, episodes: int) -> list[str]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        rng = np.random.default_rng(seed)
        return [self.sample(rng) for _ in range(episodes)]
