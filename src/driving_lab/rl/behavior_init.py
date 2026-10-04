from __future__ import annotations

import torch


def initialize_from_behavior_cloning(agent, model_path) -> None:
    state = torch.load(model_path, map_location="cpu") if isinstance(model_path, (str, bytes)) else model_path
    if not isinstance(state, dict):
        raise ValueError("behavior-cloning checkpoint must be a state dict")
    try:
        agent.network.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("behavior-cloning checkpoint shape does not match DQN network") from exc
