from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import numpy as np


@dataclass(frozen=True)
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.data.append(Experience(np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        return batch

    def __len__(self):
        return len(self.data)
