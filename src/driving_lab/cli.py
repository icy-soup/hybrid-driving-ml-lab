"""Command-line experiments for the autonomous-driving study project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import DatasetCollector, load_dataset
from .environment import HighwayEnv
from .evaluation import evaluate_policy
from .model import MLPClassifier
from .policies import NeuralPolicy, RulePolicy, SafetyShieldPolicy


def collect_dataset(episodes: int, seed: int, scenario: str) -> DatasetCollector:
    collector = DatasetCollector()
    expert = RulePolicy()
    for episode in range(episodes):
        env = HighwayEnv()
        observation = env.reset(seed=seed + episode, scenario=scenario)
        while True:
            action = expert.act(observation)
            collector.add(observation, action, episode=episode)
            transition = env.step(action)
            observation = transition.next_observation
            if transition.done:
                break
    return collector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="物理信息增强的数据驱动型自动驾驶算法研究项目"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect rule-policy demonstrations")
    collect.add_argument("--episodes", type=int, default=20)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--scenario", choices=HighwayEnv.SCENARIOS, default="random")
    collect.add_argument("--output", type=Path, default=Path("training_data.npz"))

    train = subparsers.add_parser("train", help="train the NumPy behavior-cloning model")
    train.add_argument("--data", type=Path, default=Path("training_data.npz"))
    train.add_argument("--model", type=Path, default=Path("neural_model.pkl"))
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=0)

    evaluate = subparsers.add_parser("evaluate", help="evaluate rule or neural policy")
    evaluate.add_argument("--policy", choices=("rule", "neural", "hybrid"), default="rule")
    evaluate.add_argument("--model", type=Path, default=Path("neural_model.pkl"))
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--scenario", choices=HighwayEnv.SCENARIOS, default="random")
    evaluate.add_argument("--output", type=Path, default=Path("evaluation.json"))

    baseline = subparsers.add_parser(
        "baseline", help="run the reproducible rule/neural/hybrid baseline benchmark"
    )
    baseline.add_argument("--model", type=Path, default=Path("neural_model.pkl"))
    baseline.add_argument("--episodes", type=int, default=20)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--scenario", choices=HighwayEnv.SCENARIOS, default="mixed")
    baseline.add_argument("--output", type=Path, default=Path("baseline.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        if args.episodes < 1:
            raise ValueError("episodes must be positive")
        collector = collect_dataset(args.episodes, args.seed, args.scenario)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        collector.save(args.output, metadata={"scenario": args.scenario, "source": "RulePolicy"})
        print(f"saved demonstrations to {args.output}")
        return 0

    if args.command == "train":
        dataset = load_dataset(args.data)
        splits = dataset.split_by_episode(seed=args.seed)
        train_data = splits["train"]
        validation_data = splits["validation"]
        model = MLPClassifier(seed=args.seed)
        history = model.fit(
            train_data.states,
            train_data.actions,
            epochs=args.epochs,
            validation_fraction=0.0,
            seed=args.seed,
        )
        if len(validation_data.states):
            validation_accuracy = float(
                (model.predict(validation_data.states) == validation_data.actions).mean()
            )
        else:
            validation_accuracy = history["validation_accuracy"][-1]
        args.model.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model)
        print(
            f"saved model to {args.model}; "
            f"train_accuracy={history['train_accuracy'][-1]:.3f}; "
            f"episode_validation_accuracy={validation_accuracy:.3f}; "
            f"train_samples={len(train_data.states)}; "
            f"validation_samples={len(validation_data.states)}; "
            f"action_counts={json.dumps(dataset.action_counts(), sort_keys=True)}"
        )
        return 0

    if args.command == "baseline":
        if args.episodes < 1:
            raise ValueError("episodes must be positive")
        learned_policy = NeuralPolicy(MLPClassifier.load(args.model))
        policies = {
            "rule": RulePolicy(),
            "neural": learned_policy,
            "hybrid": SafetyShieldPolicy(learned_policy),
        }
        metrics = {
            name: evaluate_policy(
                policy,
                episodes=args.episodes,
                seed=args.seed,
                scenario=args.scenario,
                reference_policy=RulePolicy() if name != "rule" else None,
            )
            for name, policy in policies.items()
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    if args.policy in ("neural", "hybrid"):
        learned_policy = NeuralPolicy(MLPClassifier.load(args.model))
        policy = SafetyShieldPolicy(learned_policy) if args.policy == "hybrid" else learned_policy
    else:
        policy = RulePolicy()
    metrics = evaluate_policy(
        policy,
        episodes=args.episodes,
        seed=args.seed,
        scenario=args.scenario,
        reference_policy=RulePolicy() if args.policy in ("neural", "hybrid") else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0
