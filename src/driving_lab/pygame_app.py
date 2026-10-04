"""Interactive Pygame view backed by the research environment and policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .environment import HighwayEnv
from .model import MLPClassifier
from .policies import NeuralPolicy, RulePolicy, SafetyShieldPolicy


ACTION_LABELS = {
    0: "ACCELERATE",
    1: "BRAKE",
    2: "LANE LEFT",
    3: "LANE RIGHT",
    4: "CRUISE",
}
POLICY_LABELS = {"rule": "RULE", "neural": "NEURAL", "hybrid": "HYBRID"}


def render_snapshot(screen: Any, snapshot: dict[str, Any], assets: Any = None, metrics: dict[str, Any] | None = None) -> None:
    """Render an immutable environment snapshot; this function never steps it."""
    import pygame

    metrics = metrics or {}
    screen.fill((22, 29, 40))
    road = pygame.Rect(24, 24, 592, 372)
    pygame.draw.rect(screen, (55, 63, 74), road, border_radius=12)
    for lane in range(3):
        y = 86 + lane * 120
        pygame.draw.line(screen, (210, 214, 220), (36, y), (604, y), 2)
    ego = snapshot.get("ego", {})
    ego_x = 190 + float(ego.get("x", 0.0)) * 1.5
    ego_y = 86 + int(ego.get("lane", 1)) * 120
    pygame.draw.rect(screen, (53, 156, 255), (ego_x - 22, ego_y - 15, 44, 30), border_radius=6)
    for vehicle in snapshot.get("vehicles", []):
        x = 190 + float(vehicle.get("x", 0.0)) * 1.5
        y = 86 + int(vehicle.get("lane", 1)) * 120
        if 36 <= x <= 604:
            color = (235, 91, 74) if vehicle.get("crashed", False) else (240, 169, 67)
            pygame.draw.rect(screen, color, (x - 18, y - 13, 36, 26), border_radius=5)
    font = pygame.font.SysFont("segoeui", 15)
    label = f"step {snapshot.get('step', 0)}   collisions {metrics.get('collision_count', 0)}"
    screen.blit(font.render(label, True, (236, 240, 245)), (38, 370))


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
    screen = pygame.display.set_mode((1120, 620))
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
                    selected = {pygame.K_1: "rule", pygame.K_2: "neural", pygame.K_3: "hybrid"}[event.key]
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

        _draw_scene(
            pygame,
            screen,
            env,
            observation,
            title_font,
            body_font,
            small_font,
            active_name,
            last_action,
            total_reward,
            done,
            paused,
        )
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


def _text(screen: Any, font: Any, value: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    screen.blit(font.render(value, True, color), position)
