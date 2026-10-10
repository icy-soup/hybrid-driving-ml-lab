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
    current_speed: float | None = None,
    target_speed: float = 0.0,
    reaction_frames: float = 5.0,
    max_deceleration: float = 0.25,
    margin: float = 40.0,
) -> float:
    """Distance needed to reduce the closing speed safely.

    ``closing_speed`` is retained as the first argument for compatibility with
    the old policy code.  When the ego and lead speeds are available, braking
    is computed from the actual speed that must be shed.  That matters for a
    moving lead: reducing 8 m/s to a 4 m/s lead takes more room than treating
    the relative speed (4 m/s) as if the ego were stopping from rest.
    """

    if closing_speed <= 0.0:
        return 50.0
    if max_deceleration <= 0.0:
        raise ValueError("max_deceleration must be positive")
    ego_speed = abs(closing_speed) if current_speed is None else max(0.0, abs(current_speed))
    lead_speed = 0.0 if current_speed is None else max(0.0, abs(target_speed))
    reaction_distance = closing_speed * reaction_frames
    speed_reduction_distance = max(0.0, ego_speed**2 - lead_speed**2)
    braking_distance = speed_reduction_distance / (2.0 * max_deceleration)
    return reaction_distance + braking_distance + margin


def safe_follow_distance(speed_abs: float, *, minimum: float = 250.0, time_gap: float = 2.5) -> float:
    """Return a speed-dependent following distance in pixels."""

    return max(minimum, abs(speed_abs) * time_gap)


def lane_change_front_gap(
    current_speed: float,
    target_speed: float,
    *,
    lane_change_time: float = 8.0,
    max_deceleration: float = 0.5,
    margin: float = 40.0,
) -> float:
    """Minimum front gap needed to complete a lane change safely.

    The gap covers longitudinal travel during the lateral maneuver and the
    distance needed to settle at the target-lane vehicle's speed.
    """
    speed = max(0.0, abs(current_speed))
    target = max(0.0, abs(target_speed))
    closing = max(0.0, speed - target)
    maneuver_distance = speed * max(0.0, lane_change_time)
    braking_distance = emergency_braking_distance(
        closing,
        current_speed=speed,
        target_speed=target,
        max_deceleration=max_deceleration,
        reaction_frames=0.0,
        margin=0.0,
    ) if closing > 0.0 else 0.0
    return max(120.0, maneuver_distance + braking_distance + margin)


def lane_change_rear_gap(
    current_speed: float,
    rear_speed: float,
    *,
    lane_change_time: float = 8.0,
    margin: float = 40.0,
) -> float:
    """Minimum rear gap after predicting the rear vehicle through a maneuver."""
    closing = max(0.0, abs(rear_speed) - abs(current_speed))
    predicted_closing = closing * max(0.0, lane_change_time)
    return max(100.0, predicted_closing + margin)
