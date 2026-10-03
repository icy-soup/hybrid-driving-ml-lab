"""Configuration for the headless highway environment."""

from dataclasses import dataclass, field

from .rewards import RewardConfig


@dataclass(frozen=True)
class EnvironmentConfig:
    lanes: int = 3
    lane_width: float = 50.0
    ego_start_speed: float = 8.0
    min_forward_speed: float = 4.0
    max_forward_speed: float = 36.0
    acceleration_step: float = 0.5
    braking_step: float = 0.5
    collision_distance: float = 32.0
    detection_distance: float = 600.0
    max_steps: int = 600
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.lanes < 1:
            raise ValueError("lanes must be positive")
        if self.min_forward_speed <= 0 or self.max_forward_speed < self.min_forward_speed:
            raise ValueError("invalid forward speed bounds")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
