from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.environment import HighwayEnv
from driving_lab.policies import RulePolicy


def test_step_diagnostics_include_reward_distance_and_vehicle_state():
    env = HighwayEnv()
    observation = env.reset(seed=3, scenario="rear_approach")
    transition = env.step(RulePolicy().act(observation))
    record = env.diagnostics.steps[-1]
    assert {"reward", "distance", "reward_components", "vehicles"} <= set(record)
    json.dumps(record, allow_nan=False)
