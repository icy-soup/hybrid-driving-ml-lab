"""Small reproducible DQN training entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driving_lab.rl import DQNAgent, DQNConfig, DQNTrainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=Path("experiments/models/dqn.pt"))
    args = parser.parse_args()
    agent = DQNAgent(DQNConfig(device=args.device))
    metrics = DQNTrainer(agent).train_steps(args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.output)
    print(json.dumps({**metrics, "device": str(agent.device), "checkpoint": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
