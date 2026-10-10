"""Batch policy evaluation independent of rendering."""

from __future__ import annotations

from typing import Any

from .config import EnvironmentConfig
from .environment import HighwayEnv
from .policies import Policy


def evaluate_policy(
    policy: Policy,
    *,
    episodes: int = 20,
    seed: int = 0,
    scenario: str = "random",
    config: EnvironmentConfig | None = None,
    reference_policy: Policy | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    collision_count = 0
    survival_steps: list[int] = []
    speeds: list[float] = []
    returns: list[float] = []
    overtakes: list[int] = []
    lane_changes: list[int] = []
    ttc_warnings: list[int] = []
    distances: list[float] = []
    scores: list[float] = []
    shield_overrides = 0
    total_steps = 0
    action_matches = 0
    action_total = 0

    for episode in range(episodes):
        if reference_policy is not None:
            reference_env = HighwayEnv(config)
            reference_observation = reference_env.reset(seed=seed + episode, scenario=scenario)
            while True:
                reference_action = reference_policy.act(reference_observation)
                action_matches += int(policy.act(reference_observation).index == reference_action.index)
                action_total += 1
                reference_transition = reference_env.step(reference_action)
                reference_observation = reference_transition.next_observation
                if reference_transition.done:
                    break

        env = HighwayEnv(config)
        observation = env.reset(seed=seed + episode, scenario=scenario)
        total_speed = 0.0
        total_return = 0.0
        total_distance = 0.0
        steps = 0
        while True:
            action = policy.act(observation)
            shield_overrides += int(getattr(policy, "last_override", False))
            transition = env.step(action)
            total_speed += abs(env.ego.speed)
            total_distance += abs(env.ego.speed)
            total_return += transition.reward
            steps += 1
            observation = transition.next_observation
            if transition.done:
                collision_count += int(transition.info["collision"])
                survival_steps.append(steps)
                speeds.append(total_speed / steps)
                returns.append(total_return)
                events = transition.info["events"]
                overtakes.append(events["overtakes"])
                lane_changes.append(events["lane_changes"])
                ttc_warnings.append(events["ttc_warnings"])
                distances.append(total_distance)
                # Primary score is progress; safety terms are bounded episode
                # penalties and overtakes are deliberately excluded.
                scores.append(
                    total_distance
                    - 100.0 * int(transition.info["collision"])
                    - 0.5 * events["ttc_warnings"]
                    - 0.05 * events["lane_changes"]
                )
                total_steps += steps
                break

    metrics: dict[str, Any] = {
        "episodes": episodes,
        "collisions": collision_count,
        "collision_rate": collision_count / episodes,
        "average_survival_steps": sum(survival_steps) / len(survival_steps),
        "average_speed": sum(speeds) / len(speeds),
        "average_speed_kmh": (sum(speeds) / len(speeds)) * 3.6,
        "average_return": sum(returns) / len(returns),
        "average_overtakes": sum(overtakes) / len(overtakes),
        "average_lane_changes": sum(lane_changes) / len(lane_changes),
        "average_ttc_warnings": sum(ttc_warnings) / len(ttc_warnings),
        "average_distance": sum(distances) / len(distances),
        "average_score": sum(scores) / len(scores),
        "shield_override_rate": shield_overrides / max(total_steps, 1),
    }
    if reference_policy is not None:
        metrics["action_accuracy_vs_reference"] = action_matches / max(action_total, 1)
    return metrics
