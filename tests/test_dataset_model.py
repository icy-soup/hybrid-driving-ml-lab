import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.dataset import DatasetCollector, load_dataset  # noqa: E402
from driving_lab.model import MLPClassifier, class_balanced_weights  # noqa: E402
from driving_lab.types import Action, ActionType, Observation  # noqa: E402


def make_observation(value: float = 0.0) -> Observation:
    return Observation(
        speed=-8.0 + value,
        lane=1,
        front_distance=(500.0, 500.0, 500.0),
        front_speed=(None, None, None),
    )


def test_dataset_save_load_preserves_episode_boundaries(tmp_path):
    collector = DatasetCollector()
    for episode in range(3):
        collector.add(make_observation(float(episode)), Action(ActionType.ACCELERATE), episode=episode)

    path = tmp_path / "dataset.npz"
    collector.save(path)
    dataset = load_dataset(path)
    splits = dataset.split_by_episode(seed=4)

    assert dataset.states.shape == (3, 11)
    assert dataset.actions.tolist() == [0, 0, 0]
    assert set(splits["train"].episodes).isdisjoint(splits["validation"].episodes)
    assert set(splits["train"].episodes).isdisjoint(splits["test"].episodes)


def test_legacy_dataset_without_episode_array_is_marked(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, states=np.zeros((2, 11), dtype=np.float32), actions=np.array([0, 1]))

    dataset = load_dataset(path)

    assert dataset.metadata["legacy_format"] is True
    assert dataset.episodes.tolist() == [0, 0]


def test_dataset_reports_all_action_counts():
    collector = DatasetCollector()
    for index, action in enumerate(
        (ActionType.ACCELERATE, ActionType.ACCELERATE, ActionType.BRAKE)
    ):
        collector.add(make_observation(float(index)), Action(action), episode=0)

    assert collector.get_dataset().action_counts() == {0: 2, 1: 1, 2: 0, 3: 0, 4: 0}


def test_empty_dataset_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        DatasetCollector().save(tmp_path / "empty.npz")


def test_mlp_trains_and_round_trips(tmp_path):
    rng = np.random.default_rng(5)
    states = rng.normal(size=(12, 11)).astype(np.float32)
    actions = np.array([0, 1] * 6)
    model = MLPClassifier(seed=5)
    history = model.fit(states, actions, epochs=3, validation_fraction=0.25)
    path = tmp_path / "model.pkl"
    model.save(path)
    restored = MLPClassifier.load(path)

    assert len(history["train_loss"]) == 3
    assert restored.predict(states[:2]).shape == (2,)


def test_mlp_learns_a_simple_separable_action_rule():
    rng = np.random.default_rng(3)
    states = rng.normal(size=(100, 11)).astype(np.float32)
    actions = (states[:, 0] > 0.0).astype(np.int64)

    model = MLPClassifier(seed=3)
    history = model.fit(states, actions, epochs=30, validation_fraction=0.0, seed=3)

    assert history["train_accuracy"][-1] > 0.8


def test_mlp_rejects_single_class_training_data():
    model = MLPClassifier(seed=5)
    with pytest.raises(ValueError, match="at least two action classes"):
        model.fit(np.zeros((4, 11), dtype=np.float32), np.zeros(4, dtype=np.int64), epochs=1)


def test_class_balanced_weights_upweight_rare_actions():
    weights = class_balanced_weights(np.array([0, 0, 0, 1, 1, 2]))
    assert weights[2] > weights[0]
