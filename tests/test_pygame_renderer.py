from __future__ import annotations

import copy
import os

import pytest

pygame = pytest.importorskip("pygame")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from driving_lab.environment import HighwayEnv
from driving_lab.pygame_app import build_snapshot, render_snapshot


def test_renderer_consumes_snapshot_without_mutating_it():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.Surface((640, 420))
    snapshot = {"ego": {"x": 0.0, "lane": 1, "speed": 8.0}, "vehicles": [{"x": 40.0, "lane": 1, "crashed": False}], "step": 1}
    original = copy.deepcopy(snapshot)
    render_snapshot(screen, snapshot, metrics={"collision_count": 0})
    assert snapshot == original
    pygame.quit()


def test_snapshot_contains_audit_fields_from_environment():
    env = HighwayEnv()
    env.reset(seed=4, scenario="rear_approach")
    snapshot = build_snapshot(env, last_action=4, total_reward=1.25)
    assert {"ego", "vehicles", "step", "score", "metrics", "recent_events"} <= set(snapshot)
    assert snapshot["ego"]["distance"] >= 0.0
    assert snapshot["vehicles"][0]["vehicle_id"] == "rear"
    assert "mode" in snapshot["vehicles"][0]


def test_snapshot_is_detached_from_environment_objects():
    env = HighwayEnv()
    env.reset(seed=2, scenario="empty")
    snapshot = build_snapshot(env, last_action=4, total_reward=0.0)
    snapshot["ego"]["speed"] = -999.0
    assert env.ego.speed != -999.0
