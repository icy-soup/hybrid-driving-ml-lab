"""Configuration for the headless highway environment."""

from dataclasses import dataclass, field

from .rewards import RewardConfig


@dataclass(frozen=True)
class EnvironmentConfig:
    lanes: int = 3
    lane_width: float = 50.0
    # Internal kinematics use m/s.  The UI converts these values to km/h.
    # The road is allowed to reach 140 km/h, while the baseline controller
    # chooses a lower cruising target and can later learn this parameter.
    ego_start_speed: float = 8.0
    min_forward_speed: float = 4.0
    max_forward_speed: float = 38.8888888889  # 140 km/h
    cruise_speed: float = 27.7777777778  # 100 km/h baseline target
    traffic_cruise_speed: float = 24.0  # 86.4 km/h default NPC target
    acceleration_step: float = 0.5
    braking_step: float = 1.0
    collision_distance: float = 32.0
    traffic_min_gap: float = 72.0
    detection_distance: float = 600.0
    max_steps: int = 600
    legacy_forward_spawn_x: float = 960.0
    legacy_rear_spawn_x: float = -150.0
    legacy_max_traffic: int = 24
    legacy_remove_left: float = -400.0
    legacy_remove_right: float = 1310.0
    legacy_slow_remove_left: float = -350.0
    legacy_slow_remove_right: float = 1260.0
    explosion_frames: int = 30
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.lanes < 1:
            raise ValueError("lanes must be positive")
        if self.min_forward_speed <= 0 or self.max_forward_speed < self.min_forward_speed:
            raise ValueError("invalid forward speed bounds")
        if self.traffic_min_gap <= 0:
            raise ValueError("traffic_min_gap must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.legacy_max_traffic < 1:
            raise ValueError("legacy_max_traffic must be positive")
