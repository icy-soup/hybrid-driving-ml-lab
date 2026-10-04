"""Configurable reward shaping and episode event accounting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RewardConfig:
    alive: float = 0.01
    speed_weight: float = 0.05
    progress_weight: float = 0.05
    target_speed: float = 12.0
    # Overtakes are logged for analysis, but intentionally low-weight because
    # random traffic layouts should not dominate the learning signal.
    overtake: float = 0.05
    ttc_warning: float = -0.2
    ttc_threshold: float = 2.5
    lane_change: float = -0.01
    shield_override: float = -0.1
    collision: float = -20.0


@dataclass
class EpisodeEvents:
    overtakes: int = 0
    lane_changes: int = 0
    ttc_warnings: int = 0
    collisions: int = 0
    shield_overrides: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "overtakes": self.overtakes,
            "lane_changes": self.lane_changes,
            "ttc_warnings": self.ttc_warnings,
            "collisions": self.collisions,
            "shield_overrides": self.shield_overrides,
        }


@dataclass(frozen=True)
class RewardBreakdown:
    components: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(sum(self.components.values()))


def compute_reward(
    config: RewardConfig,
    *,
    speed: float,
    ttc: float,
    overtake: bool = False,
    lane_change: bool = False,
    collision: bool = False,
    shield_override: bool = False,
) -> RewardBreakdown:
    speed_error = abs(abs(speed) - config.target_speed) / max(config.target_speed, 1e-6)
    speed_component = config.speed_weight * max(0.0, 1.0 - speed_error)
    progress_component = config.progress_weight * min(
        1.0, abs(speed) / max(config.target_speed, 1e-6)
    )
    components = {
        "alive": config.alive if not collision else 0.0,
        "speed": speed_component if not collision else 0.0,
        "progress": progress_component if not collision else 0.0,
        "overtake": config.overtake if overtake and not collision else 0.0,
        "ttc_warning": config.ttc_warning if ttc < config.ttc_threshold else 0.0,
        "lane_change": config.lane_change if lane_change else 0.0,
        "shield_override": config.shield_override if shield_override else 0.0,
        "collision": config.collision if collision else 0.0,
    }
    return RewardBreakdown(components)
