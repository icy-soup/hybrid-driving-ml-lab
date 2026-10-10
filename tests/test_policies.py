import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.policies import RulePolicy, SafetyShieldPolicy  # noqa: E402
from driving_lab.types import ActionType, Observation  # noqa: E402


def make_observation(**overrides):
    values = dict(
        speed=-8.0,
        lane=1,
        front_distance=(math.inf, math.inf, math.inf),
        front_speed=(None, None, None),
        ttc=(math.inf, math.inf, math.inf),
        rear_distance=(math.inf, math.inf, math.inf),
    )
    values.update(overrides)
    return Observation(**values)


def test_rule_policy_changes_lane_for_a_crashed_obstacle_when_clear():
    observation = make_observation(
        front_distance=(math.inf, 120.0, math.inf),
        front_speed=(None, 0.0, None),
        ttc=(math.inf, 2.0, math.inf),
        crashed_front=(False, True, False),
    )

    assert RulePolicy().act(observation).kind is ActionType.LANE_LEFT


def test_rule_policy_accelerates_when_road_is_clear_and_below_target_speed():
    observation = make_observation(speed=-8.0)
    assert RulePolicy(target_speed=12.0).act(observation).kind is ActionType.ACCELERATE


def test_rule_policy_changes_to_a_safe_adjacent_lane():
    observation = make_observation(
        front_distance=(500.0, 180.0, 420.0),
        front_speed=(None, -4.0, None),
        ttc=(math.inf, 2.0, math.inf),
    )

    assert RulePolicy().act(observation).kind is ActionType.LANE_LEFT


def test_rule_policy_does_not_change_into_a_close_rear_vehicle():
    observation = make_observation(
        front_distance=(500.0, 180.0, 420.0),
        front_speed=(None, -4.0, None),
        ttc=(math.inf, 2.0, math.inf),
        rear_distance=(100.0, math.inf, math.inf),
    )

    assert RulePolicy().act(observation).kind is ActionType.LANE_RIGHT


def test_safety_shield_overrides_a_neural_cruise_for_imminent_collision():
    class AlwaysCruise:
        def act(self, observation):
            return type("Decision", (), {"kind": ActionType.CRUISE})()

    observation = make_observation(
        front_distance=(math.inf, 80.0, math.inf),
        front_speed=(None, 0.0, None),
        ttc=(math.inf, 1.0, math.inf),
    )

    action = SafetyShieldPolicy(AlwaysCruise()).act(observation)

    assert action.kind is ActionType.BRAKE


def test_rule_policy_brakes_early_for_a_moving_lead_using_ego_speed():
    # Relative speed is only 4, but stopping distance must be based on the
    # ego's 8 speed while the lead keeps moving at 4.
    observation = make_observation(
        speed=-8.0,
        front_distance=(math.inf, 180.0, math.inf),
        front_speed=(None, -4.0, None),
        ttc=(math.inf, 45.0, math.inf),
        rear_distance=(100.0, math.inf, 100.0),
    )

    assert RulePolicy().act(observation).kind is ActionType.BRAKE


def test_rule_policy_looks_at_both_adjacent_lanes_before_choosing_escape():
    """A close lead with two neighbouring vehicles must be planned jointly."""
    observation = make_observation(
        speed=-12.0,
        front_distance=(150.0, 135.0, 240.0),
        front_speed=(-10.0, -2.0, -11.0),
        ttc=(75.0, 13.5, 240.0),
        rear_distance=(190.0, 45.0, 35.0),
        rear_speed=(-10.0, -18.0, -18.0),
    )
    policy = RulePolicy()
    action = policy.act(observation)
    assert action.kind is ActionType.BRAKE
    assert policy.last_reason.startswith("brake")
