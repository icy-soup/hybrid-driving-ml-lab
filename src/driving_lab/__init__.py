"""Headless, reproducible components for the autonomous-driving study project."""

from .types import Action, ActionType, Observation, Transition, VehicleState
from .config import EnvironmentConfig
from .environment import HighwayEnv
from .policies import NeuralPolicy, RulePolicy, SafetyShieldPolicy
from .rewards import EpisodeEvents, RewardBreakdown, RewardConfig
from .scenarios import ScenarioSampler
from .vector_env import VectorHighwayEnv

__all__ = [
    "Action",
    "ActionType",
    "Observation",
    "Transition",
    "VehicleState",
    "EnvironmentConfig",
    "HighwayEnv",
    "NeuralPolicy",
    "RulePolicy",
    "SafetyShieldPolicy",
    "RewardConfig",
    "RewardBreakdown",
    "EpisodeEvents",
    "ScenarioSampler",
    "VectorHighwayEnv",
]
