"""Interactive Pygame view backed by the research environment and policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .environment import HighwayEnv
from .controllers import MPCPolicy
from .model import MLPClassifier
from .policies import NeuralPolicy, RulePolicy, SafetyShieldPolicy


ACTION_LABELS = {
    0: "ACCELERATE",
    1: "BRAKE",
    2: "LANE LEFT",
    3: "LANE RIGHT",
    4: "CRUISE",
}
POLICY_LABELS = {"rule": "RULE", "mpc": "MPC", "neural": "NEURAL", "hybrid": "HYBRID"}


def build_snapshot(env: HighwayEnv, *, last_action: int, total_reward: float) -> dict[str, Any]:
    """Copy the canonical environment state into a renderer-safe dictionary."""
    events = env.episode_events.as_dict()
    diagnostics = env.diagnostics.summary()
    vehicles = []
    for vehicle in env.vehicles:
        vehicles.append(
            {
                "vehicle_id": vehicle.vehicle_id,
                "x": float(vehicle.x),
                "y": float(vehicle.y if vehicle.y is not None else vehicle.lane * env.config.lane_width),
                "lane": int(vehicle.lane),
                "speed": float(vehicle.speed),
                "mode": vehicle.mode.name,
                "crashed": bool(vehicle.crashed),
                "target_lane": vehicle.target_lane,
            }
        )
    recent_events = []
    for event in env.diagnostics.collisions[-3:]:
        recent_events.append(
            {
                "type": event.collision_type,
                "vehicles": f"{event.vehicle_a} / {event.vehicle_b}",
                "reason": event.reason,
            }
        )
    distance = float(env.step_count * abs(env.ego.speed))
    return {
        "step": int(env.step_count),
        "scenario": str(env.scenario),
        "last_action": int(last_action),
        "score": float(total_reward),
        "ego": {
            "x": float(env.ego.x),
            "lane": int(env.ego.lane),
            "speed": float(env.ego.speed),
            "distance": distance,
            "mode": env.ego.mode.name,
        },
        "vehicles": vehicles,
        "metrics": {**events, **diagnostics, "distance": distance},
        "recent_events": recent_events,
    }


def render_snapshot(screen: Any, snapshot: dict[str, Any], assets: Any = None, metrics: dict[str, Any] | None = None) -> None:
    """Render an immutable environment snapshot; this function never steps it."""
    import pygame

    metrics = {**snapshot.get("metrics", {}), **(metrics or {})}
    width, height = screen.get_size()
    panel_x = int(width * 0.68)
    screen.fill((14, 21, 32))
    pygame.draw.rect(screen, (31, 41, 56), (0, 0, panel_x, height))
    pygame.draw.rect(screen, (247, 249, 252), (panel_x, 0, width - panel_x, height))
    title = pygame.font.SysFont("segoeui", 24, bold=True)
    body = pygame.font.SysFont("segoeui", 16)
    small = pygame.font.SysFont("segoeui", 13)
    _text(screen, title, "HYBRID DRIVING LAB", (28, 22), (240, 244, 250))
    _text(screen, small, f"SCENARIO  {str(snapshot.get('scenario', '')) .upper()}", (30, 54), (145, 163, 186))
    road = pygame.Rect(24, 86, panel_x - 48, height - 132)
    pygame.draw.rect(screen, (52, 61, 72), road, border_radius=16)
    lane_height = road.height // 3
    for lane in range(4):
        y = road.top + lane * lane_height
        pygame.draw.line(screen, (176, 185, 194), (road.left + 12, y), (road.right - 12, y), 2 if lane in (0, 3) else 1)
    ego = snapshot.get("ego", {})
    ego_x = road.left + road.width * 0.32
    ego_y = road.top + (int(ego.get("lane", 1)) + 0.5) * lane_height
    pygame.draw.rect(screen, (45, 151, 255), (ego_x - 25, ego_y - 18, 50, 36), border_radius=8)
    _text(screen, small, "EGO", (ego_x - 17, ego_y - 8), (235, 247, 255))
    scale = 1.35
    for vehicle in snapshot.get("vehicles", []):
        x = ego_x + (float(vehicle.get("x", 0.0)) * scale)
        y = road.top + (int(vehicle.get("lane", 1)) + 0.5) * lane_height
        if road.left + 16 <= x <= road.right - 16:
            color = (226, 75, 76) if vehicle.get("crashed") else (240, 166, 65)
            pygame.draw.rect(screen, color, (x - 22, y - 16, 44, 32), border_radius=7)
            _text(screen, small, str(vehicle.get("mode", "CRUISE")), (x - 28, y + 20), (222, 230, 237))
    _panel_heading(screen, title, body, panel_x + 24, 28, "LIVE AUDIT")
    _metric(screen, body, "Policy action", ACTION_LABELS.get(snapshot.get("last_action", 4), "UNKNOWN"), (panel_x + 24, 82))
    _metric(screen, body, "Step", str(snapshot.get("step", 0)), (panel_x + 24, 112))
    _metric(screen, body, "Speed", f"{float(ego.get('speed', 0.0)):.1f} m/s", (panel_x + 24, 142))
    _metric(screen, body, "Distance", f"{float(ego.get('distance', 0.0)):.1f}", (panel_x + 24, 172))
    _metric(screen, body, "Score", f"{float(snapshot.get('score', 0.0)):.2f}", (panel_x + 24, 202))
    _panel_heading(screen, title, body, panel_x + 24, 252, "SAFETY")
    _metric(screen, body, "Collisions", str(metrics.get("collision_count", 0)), (panel_x + 24, 306))
    _metric(screen, body, "TTC warnings", str(metrics.get("ttc_warnings", 0)), (panel_x + 24, 336))
    _metric(screen, body, "Lane changes", str(metrics.get("lane_changes", 0)), (panel_x + 24, 366))
    _panel_heading(screen, title, body, panel_x + 24, 416, "RECENT EVENTS")
    recent = snapshot.get("recent_events", [])
    if not recent:
        _text(screen, small, "No collision events", (panel_x + 24, 470), (100, 113, 128))
    for index, event in enumerate(recent[-3:]):
        y = 470 + index * 42
        _text(screen, small, str(event.get("type", "event")).upper(), (panel_x + 24, y), (194, 71, 75))
        _text(screen, small, str(event.get("vehicles", "")), (panel_x + 24, y + 16), (73, 84, 98))
    _text(screen, small, "SPACE pause   R reset   ESC exit", (28, height - 24), (178, 190, 204))


def world_to_screen(
    world_x: float,
    lane: int,
    *,
    ego_x: float = 220.0,
    scale: float = 1.5,
) -> tuple[float, float]:
    """Project ego-centric highway coordinates into the 2D road view."""

    return ego_x + world_x * scale, 180.0 + lane * 120.0


def build_policy(name: str, model_path: Path | None = None):
    """Build one of the policies exposed by the interactive viewer."""

    if name == "rule":
        return RulePolicy()
    if name == "mpc":
        return MPCPolicy()
    if name not in {"neural", "hybrid"}:
        raise ValueError(f"unknown policy: {name}")
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            f"model file is required for {name} policy: {model_path}"
        )
    learned = NeuralPolicy(MLPClassifier.load(model_path))
    return SafetyShieldPolicy(learned) if name == "hybrid" else learned


def run(
    *,
    policy_name: str = "hybrid",
    model_path: Path | None = None,
    scenario: str = "mixed",
    seed: int = 7,
    fps: int = 10,
) -> int:
    """Run the interactive viewer. Pygame is imported only when this is called."""

    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - depends on local GUI setup
        raise RuntimeError(
            "Pygame is not installed. Install it with: python -m pip install pygame"
        ) from exc

    pygame.init()
    screen = pygame.display.set_mode((1280, 760))
    pygame.display.set_caption("Hybrid Driving ML Lab")
    clock = pygame.time.Clock()
    title_font = pygame.font.SysFont("segoeui", 26, bold=True)
    body_font = pygame.font.SysFont("segoeui", 17)
    small_font = pygame.font.SysFont("segoeui", 14)

    env = HighwayEnv()
    observation = env.reset(seed=seed, scenario=scenario)
    policy = build_policy(policy_name, model_path)
    active_name = policy_name
    total_reward = 0.0
    last_action = 4
    done = False
    paused = False
    episode_seed = seed
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    episode_seed += 1
                    observation = env.reset(seed=episode_seed, scenario=scenario)
                    total_reward = 0.0
                    last_action = 4
                    done = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    selected = {pygame.K_1: "rule", pygame.K_2: "mpc", pygame.K_3: "hybrid"}[event.key]
                    try:
                        policy = build_policy(selected, model_path)
                    except FileNotFoundError:
                        selected = active_name
                    else:
                        active_name = selected

        if not paused and not done:
            action = policy.act(observation)
            shield_override = bool(getattr(policy, "last_override", False))
            transition = env.step(action, shield_override=shield_override)
            observation = transition.next_observation
            total_reward += transition.reward
            last_action = int(action.kind)
            done = transition.done

        snapshot = build_snapshot(env, last_action=last_action, total_reward=total_reward)
        snapshot["paused"] = paused
        snapshot["done"] = done
        render_snapshot(screen, snapshot, metrics={"policy": active_name})
        pygame.display.flip()
        clock.tick(max(1, fps))

    pygame.quit()
    return 0


def _draw_scene(
    pygame: Any,
    screen: Any,
    env: HighwayEnv,
    observation: Any,
    title_font: Any,
    body_font: Any,
    small_font: Any,
    policy_name: str,
    last_action: int,
    total_reward: float,
    done: bool,
    paused: bool,
) -> None:
    """Render a readable road view and a grouped metrics panel."""

    screen.fill((19, 27, 38))
    road = pygame.Rect(0, 0, 820, 620)
    panel = pygame.Rect(820, 0, 300, 620)
    pygame.draw.rect(screen, (49, 57, 67), road)
    pygame.draw.rect(screen, (237, 241, 245), panel)
    pygame.draw.rect(screen, (27, 34, 44), (24, 26, 772, 568), border_radius=16)

    for lane in range(3):
        y = 180 + lane * 120
        pygame.draw.line(screen, (220, 223, 227), (34, y - 60), (786, y - 60), 2)
        pygame.draw.line(screen, (220, 223, 227), (34, y + 60), (786, y + 60), 2)

    ego_x = 220.0
    ego_y = 180 + env.ego.lane * 120
    pygame.draw.rect(screen, (53, 156, 255), (ego_x - 24, ego_y - 18, 48, 36), border_radius=8)
    pygame.draw.rect(screen, (197, 232, 255), (ego_x + 4, ego_y - 12, 13, 24), border_radius=3)

    for vehicle in env.vehicles:
        x, y = world_to_screen(vehicle.x, vehicle.lane, ego_x=ego_x)
        if 45 <= x <= 790:
            color = (235, 91, 74) if vehicle.crashed else (240, 169, 67)
            pygame.draw.rect(screen, color, (x - 20, y - 16, 40, 32), border_radius=7)

    _text(screen, title_font, "HYBRID DRIVING LAB", (42, 42), (237, 241, 245))
    _text(screen, small_font, "LIVE SIMULATION", (44, 75), (139, 161, 181))
    _panel_card(pygame, screen, (840, 28, 260, 116), "POLICY", POLICY_LABELS[policy_name], body_font, title_font)
    _panel_card(pygame, screen, (840, 158, 260, 150), "VEHICLE STATE", "", body_font, title_font)
    _metric(screen, body_font, "Speed", f"{env.ego.speed:.1f} m/s", (858, 204))
    _metric(screen, body_font, "Lane", str(env.ego.lane + 1), (858, 235))
    _metric(screen, body_font, "Step", str(env.step_count), (858, 266))
    _panel_card(pygame, screen, (840, 322, 260, 154), "EPISODE", "", body_font, title_font)
    _metric(screen, body_font, "Last action", ACTION_LABELS[last_action], (858, 368))
    _metric(screen, body_font, "Reward", f"{total_reward:.1f}", (858, 399))
    _metric(screen, body_font, "Overtakes", str(env.episode_events.overtakes), (858, 430))
    _metric(screen, body_font, "Collisions", str(env.episode_events.collisions), (858, 461))
    status = "CRASHED - press R" if done else ("PAUSED - press SPACE" if paused else "RUNNING")
    status_color = (199, 67, 67) if done else ((198, 126, 38) if paused else (45, 145, 91))
    _text(screen, body_font, status, (858, 520), status_color)
    _text(screen, small_font, "1 Rule   2 Neural   3 Hybrid   R Reset   ESC Exit", (42, 578), (166, 181, 194))


def _panel_card(pygame: Any, screen: Any, rect: tuple[int, int, int, int], heading: str, value: str, body_font: Any, title_font: Any) -> None:
    pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=12)
    _text(screen, body_font, heading, (rect[0] + 18, rect[1] + 14), (111, 126, 140))
    if value:
        _text(screen, title_font, value, (rect[0] + 18, rect[1] + 47), (31, 52, 70))


def _metric(screen: Any, font: Any, label: str, value: str, position: tuple[int, int]) -> None:
    _text(screen, font, label, position, (111, 126, 140))
    _text(screen, font, value, (position[0] + 126, position[1]), (31, 52, 70))


def _panel_heading(screen: Any, title_font: Any, body_font: Any, x: int, y: int, heading: str) -> None:
    _text(screen, title_font, heading, (x, y), (31, 52, 70))
    pygame = __import__("pygame")
    pygame.draw.line(screen, (220, 225, 231), (x, y + 32), (screen.get_width() - 24, y + 32), 1)


def _text(screen: Any, font: Any, value: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    screen.blit(font.render(value, True, color), position)
