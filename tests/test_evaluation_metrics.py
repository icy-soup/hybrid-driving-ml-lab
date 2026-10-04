from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.evaluation import evaluate_policy
from driving_lab.policies import RulePolicy


def test_evaluation_reports_distance_primary_score_without_overtake_bonus():
    metrics = evaluate_policy(RulePolicy(), episodes=2, seed=4, scenario="empty")
    assert metrics["average_distance"] > 0
    assert metrics["average_score"] == metrics["average_distance"]
