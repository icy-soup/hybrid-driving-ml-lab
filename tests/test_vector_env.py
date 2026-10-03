import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.environment import HighwayEnv  # noqa: E402
from driving_lab.scenarios import ScenarioSampler  # noqa: E402
from driving_lab.types import Action, ActionType  # noqa: E402
from driving_lab.vector_env import VectorHighwayEnv  # noqa: E402


def test_scenario_sampler_is_deterministic_for_a_seed():
    first = ScenarioSampler().schedule(seed=7, episodes=10)
    second = ScenarioSampler().schedule(seed=7, episodes=10)

    assert first == second
    assert set(first).issubset(set(HighwayEnv.BASE_SCENARIOS))
    assert len(set(first)) > 1


def test_mixed_environment_resolves_to_a_concrete_scenario():
    env = HighwayEnv()
    observation = env.reset(seed=7, scenario="mixed")

    assert observation.lane == env.ego.lane
    assert env.scenario in HighwayEnv.BASE_SCENARIOS


def test_vector_environment_returns_batched_features_without_shared_state():
    env = VectorHighwayEnv(num_envs=2)
    observations = env.reset(seeds=[1, 2], scenarios=["empty", "slow_lead"])
    next_observations, transitions = env.step(
        [Action(ActionType.CRUISE), Action(ActionType.BRAKE)]
    )

    assert observations.shape == (2, 11)
    assert next_observations.shape == (2, 11)
    assert len(transitions) == 2
    assert env.envs[0] is not env.envs[1]
    assert env.envs[0].ego is not env.envs[1].ego
