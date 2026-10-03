"""Episode-aware storage for state/action demonstrations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .types import Action, Observation


@dataclass
class Dataset:
    states: np.ndarray
    actions: np.ndarray
    episodes: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.episodes = np.asarray(self.episodes, dtype=np.int64)
        if self.states.ndim != 2 or len(self.states) != len(self.actions):
            raise ValueError("states must be a 2D array aligned with actions")
        if len(self.episodes) != len(self.states):
            raise ValueError("episodes must be aligned with states")

    def action_counts(self, action_count: int = 5) -> dict[int, int]:
        """Return counts for every discrete action, including unseen actions."""

        if action_count < 1:
            raise ValueError("action_count must be positive")
        counts = np.bincount(self.actions, minlength=action_count)
        return {index: int(counts[index]) for index in range(action_count)}

    def split_by_episode(
        self,
        *,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> dict[str, "Dataset"]:
        if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
            raise ValueError("split fractions must be in [0, 1)")
        episode_ids = np.unique(self.episodes)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(episode_ids)
        if len(shuffled) < 3:
            train_ids, validation_ids, test_ids = shuffled, np.array([], dtype=int), np.array([], dtype=int)
        else:
            n_test = max(1, int(round(len(shuffled) * test_fraction))) if test_fraction else 0
            n_validation = max(1, int(round(len(shuffled) * validation_fraction))) if validation_fraction else 0
            n_test = min(n_test, len(shuffled) - 1)
            n_validation = min(n_validation, len(shuffled) - n_test - 1)
            test_ids = shuffled[:n_test]
            validation_ids = shuffled[n_test : n_test + n_validation]
            train_ids = shuffled[n_test + n_validation :]

        return {
            "train": self._select(train_ids),
            "validation": self._select(validation_ids),
            "test": self._select(test_ids),
        }

    def _select(self, episode_ids: np.ndarray) -> "Dataset":
        mask = np.isin(self.episodes, episode_ids)
        return Dataset(self.states[mask], self.actions[mask], self.episodes[mask], dict(self.metadata))


class DatasetCollector:
    def __init__(self) -> None:
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._episodes: list[int] = []

    def add(self, observation: Observation | np.ndarray, action: Action | int, *, episode: int) -> None:
        features = observation.to_features() if isinstance(observation, Observation) else np.asarray(observation)
        if features.shape != (11,):
            raise ValueError("each state must contain exactly 11 features")
        index = action.index if isinstance(action, Action) else int(action)
        if index not in range(5):
            raise ValueError("action index must be in [0, 4]")
        self._states.append(np.asarray(features, dtype=np.float32))
        self._actions.append(index)
        self._episodes.append(int(episode))

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        if not self._states:
            raise ValueError("cannot save an empty dataset")
        np.savez_compressed(
            path,
            states=np.asarray(self._states, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.int64),
            episodes=np.asarray(self._episodes, dtype=np.int64),
            metadata=json.dumps(metadata or {}, ensure_ascii=False),
        )

    def get_dataset(self) -> Dataset:
        if not self._states:
            raise ValueError("dataset is empty")
        return Dataset(
            np.asarray(self._states, dtype=np.float32),
            np.asarray(self._actions, dtype=np.int64),
            np.asarray(self._episodes, dtype=np.int64),
        )


def load_dataset(path: str | Path) -> Dataset:
    with np.load(path, allow_pickle=False) as data:
        states = data["states"]
        actions = data["actions"]
        legacy = "episodes" not in data.files
        episodes = data["episodes"] if not legacy else np.zeros(len(states), dtype=np.int64)
        metadata: dict[str, Any] = {}
        if "metadata" in data.files:
            metadata = json.loads(str(data["metadata"].item()))
        metadata["legacy_format"] = legacy
    return Dataset(states, actions, episodes, metadata)
