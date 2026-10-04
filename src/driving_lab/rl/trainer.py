from __future__ import annotations

import numpy as np


class DQNTrainer:
    def __init__(self, agent, seed: int = 0):
        self.agent = agent
        self.rng = np.random.default_rng(seed)

    def train_steps(self, steps: int = 100) -> dict[str, float]:
        losses = []
        state = np.zeros(self.agent.config.input_dim, dtype=np.float32)
        for index in range(steps):
            next_state = self.rng.normal(0.0, 0.1, size=state.shape).astype(np.float32)
            action = self.agent.act(state, epsilon=0.2)
            self.agent.replay.add(state, action, 0.01, next_state, False)
            loss = self.agent.train_step()
            if loss is not None: losses.append(loss)
            state = next_state
        return {"steps": steps, "mean_loss": float(np.mean(losses)) if losses else 0.0}
