"""Small, unit-tested physical calculations used by the driving policies."""

from __future__ import annotations

import math


def time_to_collision(distance: float, closing_speed: float) -> float:
    """Return TTC in simulation frames, or infinity when collision is not closing."""

    if not math.isfinite(distance) or distance <= 0.0 or closing_speed <= 0.0:
        return math.inf
    return distance / closing_speed


def emergency_braking_distance(
    closing_speed: float,
    *,
    reaction_frames: float = 5.0,
    max_deceleration: float = 0.25,
    margin: float = 40.0,
) -> float:
    """Reaction distance plus braking distance and a conservative margin."""

    if closing_speed <= 0.0:
        return 50.0
    if max_deceleration <= 0.0:
        raise ValueError("max_deceleration must be positive")
    reaction_distance = closing_speed * reaction_frames
    braking_distance = (closing_speed**2) / (2.0 * max_deceleration)
    return reaction_distance + braking_distance + margin


def safe_follow_distance(speed_abs: float, *, minimum: float = 250.0, time_gap: float = 2.5) -> float:
    """Return a speed-dependent following distance in pixels."""

    return max(minimum, abs(speed_abs) * time_gap)
