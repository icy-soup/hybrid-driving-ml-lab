"""Lightweight CPU vectorization for batched policy evaluation and sampling."""

from __future__ import annotations

import numpy as np

from .config import EnvironmentConfig
from .environment import HighwayEnv
from .types import Action, Transition


class VectorHighwayEnv:
    def __init__(self, num_envs: int, config: EnvironmentConfig | None = None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.envs = [HighwayEnv(config) for _ in range(num_envs)]

    def reset(
        self,
        *,
        seeds: list[int] | None = None,
        scenarios: list[str] | None = None,
    ) -> np.ndarray:
        count = len(self.envs)
        seeds = seeds if seeds is not None else list(range(count))
        scenarios = scenarios if scenarios is not None else ["mixed"] * count
        if len(seeds) != count or len(scenarios) != count:
            raise ValueError("seeds and scenarios must match num_envs")
        observations = [
            env.reset(seed=seed, scenario=scenario)
            for env, seed, scenario in zip(self.envs, seeds, scenarios)
        ]
        return np.stack([observation.to_features() for observation in observations])

    def step(self, actions: list[Action]) -> tuple[np.ndarray, list[Transition]]:
        if len(actions) != len(self.envs):
            raise ValueError("actions must match num_envs")
        transitions = [env.step(action) for env, action in zip(self.envs, actions)]
        observations = np.stack(
            [transition.next_observation.to_features() for transition in transitions]
        )
        return observations, transitions
