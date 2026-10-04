from __future__ import annotations

import copy
import os

import pytest

pygame = pytest.importorskip("pygame")

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from driving_lab.pygame_app import render_snapshot


def test_renderer_consumes_snapshot_without_mutating_it():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    screen = pygame.Surface((640, 420))
    snapshot = {"ego": {"x": 0.0, "lane": 1, "speed": 8.0}, "vehicles": [{"x": 40.0, "lane": 1, "crashed": False}], "step": 1}
    original = copy.deepcopy(snapshot)
    render_snapshot(screen, snapshot, metrics={"collision_count": 0})
    assert snapshot == original
    pygame.quit()
