import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.physics import (  # noqa: E402
    emergency_braking_distance,
    time_to_collision,
)
from driving_lab.types import Action, ActionType, Observation  # noqa: E402


def test_action_type_keeps_legacy_training_indices():
    assert [member.value for member in ActionType] == [0, 1, 2, 3, 4]
    assert Action(ActionType.LANE_LEFT).index == 2
    assert Action(ActionType.CRUISE).index == 4


def test_observation_exports_the_legacy_11_feature_vector():
    observation = Observation(
        speed=-12.0,
        lane=1,
        front_distance=(300.0, 180.0, math.inf),
        front_speed=(-10.0, -8.0, None),
        ttc=(math.inf, 9.0, math.inf),
        rear_distance=(math.inf, 100.0, math.inf),
    )

    features = observation.to_features()

    assert isinstance(features, np.ndarray)
    assert features.shape == (11,)
    assert features[1:4].tolist() == [0.0, 1.0, 0.0]
    assert features[-1] == 0.0


def test_time_to_collision_is_infinite_when_not_closing():
    assert time_to_collision(100.0, 0.0) == math.inf
    assert time_to_collision(100.0, -2.0) == math.inf
    assert time_to_collision(math.inf, 10.0) == math.inf
    assert time_to_collision(100.0, 10.0) == 10.0


def test_emergency_braking_distance_grows_with_closing_speed():
    slow = emergency_braking_distance(4.0)
    fast = emergency_braking_distance(8.0)
    assert slow > 0
    assert fast > slow


def test_emergency_braking_distance_accounts_for_moving_lead_speed():
    # Ego 8 -> lead 4 must shed 4 m/s, but its absolute speed still determines
    # how much relative distance is spent during the braking manoeuvre.
    moving_lead = emergency_braking_distance(
        4.0, current_speed=8.0, target_speed=4.0
    )
    stationary_lead = emergency_braking_distance(
        8.0, current_speed=8.0, target_speed=0.0
    )
    assert moving_lead > emergency_braking_distance(4.0)
    assert stationary_lead > moving_lead
