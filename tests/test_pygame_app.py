import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.pygame_app import ACTION_LABELS, POLICY_LABELS, world_to_screen


def test_world_to_screen_keeps_ego_anchor_and_maps_lanes():
    assert world_to_screen(0.0, 1, ego_x=220.0, scale=1.5) == (220.0, 300.0)
    assert world_to_screen(100.0, 0, ego_x=220.0, scale=1.5) == (370.0, 180.0)


def test_ui_labels_cover_all_current_actions_and_policies():
    assert set(ACTION_LABELS) == {0, 1, 2, 3, 4}
    assert ACTION_LABELS[0] == "ACCELERATE"
    assert set(POLICY_LABELS) == {"rule", "mpc", "neural", "hybrid"}
