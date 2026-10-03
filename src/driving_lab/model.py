"""Small NumPy MLP for behavior-cloning experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def class_balanced_weights(actions: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Return inverse-frequency weights without assigning mass to absent classes."""

    labels = np.asarray(actions, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    if np.any(present):
        weights[present] = len(labels) / (np.sum(present) * counts[present])
    return weights


class MLPClassifier:
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 32,
        output_size: int = 5,
        *,
        seed: int | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.1, size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = rng.normal(0.0, 0.1, size=(hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def _forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = states @ self.w1 + self.b1
        hidden = np.maximum(z1, 0.0)
        logits = hidden @ self.w2 + self.b2
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return z1, hidden, probabilities

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        array = np.asarray(states, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.shape[1] != self.input_size:
            raise ValueError(f"expected {self.input_size} input features")
        return self._forward(array)[2]

    def predict(self, states: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(states), axis=1)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 0.05,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> dict[str, list[float]]:
        x = np.asarray(states, dtype=np.float32)
        y = np.asarray(actions, dtype=np.int64)
        if len(x) == 0:
            raise ValueError("cannot train on empty data")
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(f"expected states with shape (n, {self.input_size})")
        if len(np.unique(y)) < 2:
            raise ValueError("at least two action classes are required")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(x))
        n_validation = int(round(len(x) * validation_fraction))
        n_validation = min(max(n_validation, 0), len(x) - 1)
        validation_idx = permutation[:n_validation]
        train_idx = permutation[n_validation:]
        train_x, train_y = x[train_idx], y[train_idx]
        validation_x, validation_y = x[validation_idx], y[validation_idx]
        class_weights = class_balanced_weights(train_y, self.output_size)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "validation_loss": [],
            "train_accuracy": [],
            "validation_accuracy": [],
        }

        for _ in range(epochs):
            order = rng.permutation(len(train_x))
            for start in range(0, len(train_x), batch_size):
                batch_idx = order[start : start + batch_size]
                batch_x = train_x[batch_idx]
                batch_y = train_y[batch_idx]
                z1, hidden, probabilities = self._forward(batch_x)
                one_hot = np.eye(self.output_size)[batch_y]
                sample_weights = class_weights[batch_y]
                dz2 = (probabilities - one_hot) * sample_weights[:, None]
                dz2 /= max(float(np.sum(sample_weights)), 1.0)
                dw2 = hidden.T @ dz2
                db2 = dz2.mean(axis=0)
                dz1 = (dz2 @ self.w2.T) * (z1 > 0.0)
                dw1 = batch_x.T @ dz1
                db1 = dz1.mean(axis=0)
                self.w2 -= learning_rate * dw2
                self.b2 -= learning_rate * db2
                self.w1 -= learning_rate * dw1
                self.b1 -= learning_rate * db1

            train_loss, train_accuracy = self._metrics(train_x, train_y)
            if len(validation_x):
                validation_loss, validation_accuracy = self._metrics(validation_x, validation_y)
            else:
                validation_loss, validation_accuracy = train_loss, train_accuracy
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            history["train_accuracy"].append(train_accuracy)
            history["validation_accuracy"].append(validation_accuracy)
        return history

    def _metrics(self, states: np.ndarray, actions: np.ndarray) -> tuple[float, float]:
        probabilities = self.predict_proba(states)
        loss = -np.mean(np.log(probabilities[np.arange(len(actions)), actions] + 1e-8))
        accuracy = float(np.mean(np.argmax(probabilities, axis=1) == actions))
        return float(loss), accuracy

    def save(self, path: str | Path) -> None:
        payload = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifier":
        with open(path, "rb") as handle:
            payload: dict[str, Any] = pickle.load(handle)
        model = cls(
            input_size=payload["input_size"],
            hidden_size=payload["hidden_size"],
            output_size=payload["output_size"],
        )
        model.w1 = payload["w1"]
        model.b1 = payload["b1"]
        model.w2 = payload["w2"]
        model.b2 = payload["b2"]
        return model
