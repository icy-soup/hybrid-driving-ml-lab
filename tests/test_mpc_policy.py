from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.controllers.mpc import MPCConfig, MPCPolicy
from driving_lab.environment import HighwayEnv
from driving_lab.types import Action, ActionType, Observation


def _observation(front=100.0, ttc=10.0, speed=12.0):
    return Observation(speed=-speed, lane=1, front_distance=(100.0, front, 100.0), front_speed=(None, 6.0, None), ttc=(10.0, ttc, 10.0))


def test_mpc_brakes_for_immediate_collision_risk():
    policy = MPCPolicy(config=MPCConfig(horizon=3))
    action = policy.act(_observation(front=8.0, ttc=0.5, speed=14.0))
    assert action.kind is ActionType.BRAKE
    assert policy.last_plan


def test_mpc_action_is_deterministic_and_horizon_one_has_plan():
    policy = MPCPolicy(config=MPCConfig(horizon=1))
    first = policy.act(_observation())
    second = policy.act(_observation())
    assert first == second
    assert len(policy.last_plan) == 1


def test_mpc_survives_fixed_lead_and_obstacle_scenarios_for_100_steps():
    """The controller must replan a lane escape before braking becomes futile."""
    for scenario in ("slow_lead", "sudden_brake", "obstacle"):
        env = HighwayEnv()
        observation = env.reset(seed=7, scenario=scenario)
        policy = MPCPolicy()
        for _ in range(100):
            transition = env.step(policy.act(observation))
            observation = transition.next_observation
            assert not transition.done
        assert env.episode_events.collisions == 0
        assert env.episode_events.lane_changes >= 1


def test_mpc_rejects_a_lane_when_the_predicted_rear_gap_is_unsafe():
    observation = Observation(
        speed=-12.0,
        lane=1,
        front_distance=(150.0, 135.0, 240.0),
        front_speed=(-10.0, -2.0, -11.0),
        ttc=(75.0, 13.5, 240.0),
        rear_distance=(190.0, 45.0, 35.0),
        rear_speed=(-10.0, -18.0, -18.0),
    )
    policy = MPCPolicy()
    action = policy.act(observation)
    assert action.kind is ActionType.BRAKE
    assert "rear" in policy.last_reason


def test_mpc_mixed_seed7_runs_to_episode_end_without_injected_rear_end():
    """Regression: a late forward spawn must not create an unavoidable crash."""
    env = HighwayEnv()
    observation = env.reset(seed=7, scenario="mixed")
    policy = MPCPolicy()
    for _ in range(env.config.max_steps):
        transition = env.step(policy.act(observation))
        observation = transition.next_observation
        if transition.done:
            break

    assert transition.info["collision"] is False
    assert transition.info["step"] == env.config.max_steps
    assert transition.info["impact_speed"] is None
