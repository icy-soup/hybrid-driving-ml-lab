import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_lab.cli import main  # noqa: E402
from driving_lab.evaluation import evaluate_policy  # noqa: E402
from driving_lab.policies import RulePolicy, SafetyShieldPolicy  # noqa: E402
from driving_lab.types import Action, ActionType  # noqa: E402


class SpeedGatePolicy:
    def act(self, observation):
        return Action(ActionType.BRAKE if observation.speed > -8.5 else ActionType.CRUISE)


def test_evaluation_reports_safety_and_performance_metrics():
    metrics = evaluate_policy(RulePolicy(), episodes=3, seed=7, scenario="empty")

    assert metrics["episodes"] == 3
    assert 0.0 <= metrics["collision_rate"] <= 1.0
    assert metrics["average_survival_steps"] > 0
    assert "average_speed" in metrics


def test_evaluation_reports_baseline_event_metrics():
    metrics = evaluate_policy(RulePolicy(), episodes=2, seed=7, scenario="mixed")

    assert "average_return" in metrics
    assert "average_overtakes" in metrics
    assert "average_lane_changes" in metrics
    assert "average_ttc_warnings" in metrics
    assert "shield_override_rate" in metrics


def test_reference_action_accuracy_uses_reference_trajectory():
    metrics = evaluate_policy(
        SpeedGatePolicy(),
        episodes=1,
        seed=2,
        scenario="empty",
        reference_policy=RulePolicy(),
    )

    assert metrics["action_accuracy_vs_reference"] > 0.9


def test_cli_collect_train_and_evaluate(tmp_path):
    dataset_path = tmp_path / "demo.npz"
    model_path = tmp_path / "demo.pkl"
    result_path = tmp_path / "metrics.json"

    assert main(["collect", "--episodes", "5", "--seed", "4", "--output", str(dataset_path)]) == 0
    assert main(
        [
            "train",
            "--data",
            str(dataset_path),
            "--model",
            str(model_path),
            "--epochs",
            "2",
        ]
    ) == 0
    assert main(
        [
            "evaluate",
            "--policy",
            "neural",
            "--model",
            str(model_path),
            "--episodes",
            "2",
            "--seed",
            "4",
            "--output",
            str(result_path),
        ]
    ) == 0
    baseline_path = tmp_path / "baseline.json"
    assert main(
        [
            "baseline",
            "--model",
            str(model_path),
            "--episodes",
            "2",
            "--seed",
            "4",
            "--scenario",
            "mixed",
            "--output",
            str(baseline_path),
        ]
    ) == 0

    assert dataset_path.exists()
    assert model_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8"))["episodes"] == 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert set(baseline) == {"rule", "neural", "hybrid"}
