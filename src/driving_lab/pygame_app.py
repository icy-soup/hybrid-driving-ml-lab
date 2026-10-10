"""Interactive Pygame view backed by the research environment and policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .environment import HighwayEnv
from .config import EnvironmentConfig
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


def _kmh(speed: float | None) -> float:
    """Convert the simulation's internal m/s value to the UI's km/h."""
    return abs(float(speed or 0.0)) * 3.6


def _asset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "pygame"


def _load_asset(name: str):
    import pygame

    matches = list((_asset_root() / "images" / "others").glob(name))
    if not matches:
        return None
    try:
        return pygame.image.load(str(matches[0])).convert_alpha()
    except pygame.error:
        return pygame.image.load(str(matches[0]))


def _load_legacy_font(size: int, bold: bool = False):
    import pygame

    fonts = list((_asset_root() / "font").glob("*.ttf"))
    if fonts:
        try:
            return pygame.font.Font(str(fonts[1 if size >= 24 and len(fonts) > 1 else 0]), size)
        except pygame.error:
            pass
    return pygame.font.SysFont("microsoftyahei", size, bold=bold)


def _load_vehicle_sprite(vehicle_key: str = "ego", crashed: bool = False, color: str | None = None, state: int = 1):
    import pygame

    vehicle_dir = _asset_root() / "images" / "vehicles"
    exact_match = False
    if vehicle_key != "ego" and color:
        exact = vehicle_dir / f"{vehicle_key}_{color}{max(0, min(2, int(state)))}.png"
        exact_match = exact.exists()
        matches = [exact] if exact_match else sorted(vehicle_dir.glob("*.png"))
    else:
        matches = sorted(vehicle_dir.glob("*.png"))
    if not matches:
        return None
    try:
        index = sum(ord(char) for char in vehicle_key) % len(matches)
        image = pygame.image.load(str(matches[index])).convert_alpha()
        if vehicle_key != "ego" and not exact_match:
            palette = ((55, 145, 235), (230, 86, 76), (239, 173, 58), (83, 190, 116), (173, 104, 219))
            tint = palette[index % len(palette)]
            overlay = pygame.Surface(image.get_size(), pygame.SRCALPHA)
            overlay.fill((*tint, 255))
            image.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        if crashed:
            image = image.copy()
            image.fill((180, 60, 60, 180), special_flags=pygame.BLEND_RGBA_MULT)
        return image
    except pygame.error:
        return None


def _load_explosion_sprite():
    """Load the same 60x60 explosion sprite used by the legacy UI."""
    return _load_asset("爆炸.png")


def _legacy_sprite_size(vehicle_type: str | None, *, ego: bool = False) -> tuple[int, int]:
    """Return the dimensions used by the legacy vehicle_details table."""
    if ego:
        return 80, 38
    return {
        "电动四轮车": (60, 48),
        "小轿车": (70, 44),
        "大卡车": (80, 60),
        "大巴车": (120, 72),
        "面包车": (75, 55),
        "卫星车": (70, 56),
        "跑车": (80, 38),
    }.get(vehicle_type or "", (70, 44))
POLICY_LABELS = {"rule": "RULE", "mpc": "MPC", "neural": "NEURAL", "hybrid": "HYBRID"}


def build_snapshot(env: HighwayEnv, *, last_action: int, total_reward: float) -> dict[str, Any]:
    """Copy the canonical environment state into a renderer-safe dictionary."""
    events = env.episode_events.as_dict()
    diagnostics = env.diagnostics.summary()
    vehicles = []
    monitor = []
    for vehicle in env.vehicles:
        vehicle_type = vehicle.vehicle_type
        sprite_width, sprite_height = _legacy_sprite_size(vehicle_type)
        screen_left = 455.0 + float(vehicle.x) - sprite_width / 2.0
        screen_bottom = 200.0 + float(vehicle.y if vehicle.y is not None else vehicle.lane * env.config.lane_width)
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
                "vehicle_type": vehicle.vehicle_type,
                "color": vehicle.color,
                "sprite_state": int(vehicle.sprite_state),
                "exploding": bool(vehicle.exploding),
                "explosion_frames": int(vehicle.explosion_frames),
                "screen_left": screen_left,
                "screen_bottom": screen_bottom,
                "sprite_width": sprite_width,
                "sprite_height": sprite_height,
            }
        )
        monitor.append(
            {
                "vehicle_id": vehicle.vehicle_id,
                "sprite": f"{vehicle_type or 'unknown'}_{vehicle.color or 'unknown'}{int(vehicle.sprite_state)}",
                "screen_left": screen_left,
                "screen_bottom": screen_bottom,
                "lane_error": float((vehicle.y if vehicle.y is not None else vehicle.lane * env.config.lane_width) - vehicle.lane * env.config.lane_width),
                "inside_road": bool(-400.0 <= screen_left <= 1310.0),
                "mode": vehicle.mode.name,
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
    distance = float(getattr(env, "distance_travelled", 0.0))
    # UI score is a bounded progress score; raw distance remains visible as a
    # separate metric. Distance is the main term, while safety events only
    # subtract bounded penalties so traffic randomness cannot dominate it.
    reference_distance = max(1.0, float(env.config.max_steps * env.config.ego_start_speed))
    progress_score = min(100.0, 100.0 * distance / reference_distance)
    safety_penalty = min(
        100.0,
        100.0 * int(diagnostics.get("collision_count", 0))
        + 0.5 * int(events.get("ttc_warnings", 0))
        + 0.05 * int(events.get("lane_changes", 0)),
    )
    primary_score = max(0.0, progress_score - safety_penalty)
    return {
        "step": int(env.step_count),
        "scenario": str(env.scenario),
        "last_action": int(last_action),
        "score": primary_score,
        "shaped_reward": float(total_reward),
        "ego": {
            "x": float(env.ego.x),
            "lane": int(env.ego.lane),
            "y": float(env.ego.y if env.ego.y is not None else env.ego.lane * env.config.lane_width),
            "target_lane": env.ego.target_lane,
            "speed": float(env.ego.speed),
            "impact_speed": getattr(env, "last_impact_speed", None),
            "impact_ttc": getattr(env, "last_impact_ttc", None),
            "action_before_collision": getattr(env, "last_action_name", None),
            "distance": distance,
            "mode": env.ego.mode.name,
            "sprite_state": int(env.ego.sprite_state),
            "vehicle_type": env.ego.vehicle_type,
            "color": env.ego.color,
            "crashed": bool(env.ego.crashed),
            "exploding": bool(env.ego.exploding),
            "explosion_frames": int(env.ego.explosion_frames),
        },
        "vehicles": vehicles,
        "monitor": monitor,
        "metrics": {**events, **diagnostics, "distance": distance},
        "recent_events": recent_events,
    }


def render_snapshot(screen: Any, snapshot: dict[str, Any], assets: Any = None, metrics: dict[str, Any] | None = None) -> None:
    """Render an immutable environment snapshot; this function never steps it."""
    import pygame

    metrics = {**snapshot.get("metrics", {}), **(metrics or {})}
    width, height = screen.get_size()
    panel_x = min(910, max(420, width - 240))
    screen.fill((191, 191, 191))
    pygame.draw.rect(screen, (60, 60, 60), (panel_x, 0, width - panel_x, height))
    title = _load_legacy_font(28, bold=True)
    body = _load_legacy_font(17)
    small = _load_legacy_font(14)
    _text(screen, title, "控制中心", (panel_x + 76, 8), (255, 255, 255))
    _text(screen, small, f"场景：{snapshot.get('scenario', 'random')}", (panel_x + 18, 42), (0, 255, 255))
    road = pygame.Rect(0, 0, panel_x, height)
    lane_asset = _load_asset("车道.png")
    rail_asset = _load_asset("白色护栏.png")
    # Scrolling assets and sprites are strictly clipped to the 910 px road;
    # they must never paint over the control panel.
    screen.set_clip(road)
    road_offset = -float(snapshot.get("ego", {}).get("distance", 0.0)) % max(1, panel_x)
    if lane_asset is not None:
        lane_scaled = pygame.transform.scale(lane_asset, (panel_x, min(165, height)))
        # Legacy Main.py scrolls the road texture by the ego's longitudinal
        # motion and repeats it across the full 910 px road.
        screen.blit(lane_scaled, (road_offset - panel_x, 150))
        screen.blit(lane_scaled, (road_offset, 150))
    else:
        pygame.draw.rect(screen, (150, 150, 150), road)
    if rail_asset is not None:
        rail_scaled = pygame.transform.scale(rail_asset, (panel_x, 60))
        for x in (road_offset - panel_x, road_offset):
            screen.blit(rail_scaled, (x, 85))
            screen.blit(rail_scaled, (x, 305))
    ego = snapshot.get("ego", {})
    # Keep the original Pygame geometry: ego anchor x≈455 and lane bottoms
    # 200/250/300.  Traffic x is already ego-relative in the environment.
    ego_x = 455.0
    ego_y = 200.0 + float(ego.get("y", int(ego.get("lane", 1)) * 50.0))
    ego_sprite = _load_vehicle_sprite(
        str(ego.get("vehicle_type") or "跑车"),
        bool(ego.get("crashed")),
        ego.get("color") or "银色",
        int(ego.get("sprite_state", 1)),
    )
    if ego_sprite is not None:
        ego_sprite = pygame.transform.smoothscale(ego_sprite, _legacy_sprite_size(None, ego=True))
        screen.blit(ego_sprite, (ego_x - ego_sprite.get_width() / 2, ego_y - ego_sprite.get_height()))
    else:
        pygame.draw.rect(screen, (45, 151, 255), (ego_x - 25, ego_y - 18, 50, 36), border_radius=8)
    _text(screen, small, "EGO", (ego_x - 17, ego_y + 25), (0, 0, 0))
    scale = 1.35
    for vehicle in snapshot.get("vehicles", []):
        # Collision participants remain in the snapshot so the legacy vehicle
        # sprite and explosion overlay can be shown at the impact position.
        x = ego_x + float(vehicle.get("x", 0.0))
        logical_y = vehicle.get("y")
        y = 200.0 + (float(logical_y) if logical_y is not None else int(vehicle.get("lane", 1)) * 50.0)
        vehicle_type = vehicle.get("vehicle_type")
        sprite_width, sprite_height = _legacy_sprite_size(vehicle_type)
        sprite_left = x - sprite_width / 2.0
        if road.left - sprite_width <= sprite_left <= road.right:
            sprite = _load_vehicle_sprite(
                str(vehicle_type or vehicle.get("vehicle_id", "traffic")),
                bool(vehicle.get("crashed")),
                vehicle.get("color"),
                int(vehicle.get("sprite_state", 1)),
            )
            if sprite is not None:
                sprite = pygame.transform.smoothscale(sprite, (sprite_width, sprite_height))
                screen.blit(sprite, (sprite_left, y - sprite.get_height()))
            else:
                color = (226, 75, 76) if vehicle.get("crashed") else (240, 166, 65)
                pygame.draw.rect(screen, color, (sprite_left, y - sprite_height, sprite_width, sprite_height), border_radius=7)
            if vehicle.get("exploding") or vehicle.get("crashed"):
                explosion = _load_explosion_sprite()
                if explosion is not None:
                    screen.blit(explosion, (x - explosion.get_width() / 2, y - sprite_height / 2 - explosion.get_height() / 2))
    if ego.get("exploding") or ego.get("crashed"):
        explosion = _load_explosion_sprite()
        if explosion is not None:
            screen.blit(explosion, (ego_x - explosion.get_width() / 2, ego_y - _legacy_sprite_size(None, ego=True)[1] / 2 - explosion.get_height() / 2))
    screen.set_clip(None)
    _metric(screen, body, "动作", ACTION_LABELS.get(snapshot.get("last_action", 4), "UNKNOWN"), (panel_x + 18, 78), color=(255, 255, 255))
    _metric(screen, body, "步数", str(snapshot.get("step", 0)), (panel_x + 18, 108), color=(255, 255, 255))
    impact_speed = ego.get("impact_speed")
    speed_text = f"{_kmh(ego.get('speed', 0.0)):.1f} km/h"
    if impact_speed is not None:
        speed_text += f"  撞击前 {_kmh(impact_speed):.1f}"
    _metric(screen, body, "速度", speed_text, (panel_x + 18, 138), color=(255, 255, 255))
    _metric(screen, body, "距离", f"{float(ego.get('distance', 0.0)):.1f}", (panel_x + 18, 168), color=(255, 255, 255))
    _metric(screen, body, "主分", f"{float(snapshot.get('score', 0.0)):.2f}", (panel_x + 18, 198), color=(0, 255, 0))
    _panel_heading(screen, title, body, panel_x + 18, 238, "安全诊断")
    _metric(screen, body, "碰撞", str(metrics.get("collision_count", 0)), (panel_x + 18, 284), color=(255, 255, 255))
    _metric(screen, body, "TTC警告", str(metrics.get("ttc_warnings", 0)), (panel_x + 18, 314), color=(255, 255, 255))
    _metric(screen, body, "变道", str(metrics.get("lane_changes", 0)), (panel_x + 18, 344), color=(255, 255, 255))
    _panel_heading(screen, title, body, panel_x + 18, 378, "最近事件")
    recent = snapshot.get("recent_events", [])
    if not recent:
        _text(screen, small, "暂无碰撞事件", (panel_x + 18, 418), (210, 210, 210))
    for index, event in enumerate(recent[-3:]):
        y = 418 + index * 32
        _text(screen, small, str(event.get("type", "event")), (panel_x + 18, y), (255, 100, 100))
        _text(screen, small, str(event.get("vehicles", "")), (panel_x + 18, y + 15), (230, 230, 230))
    _text(screen, small, "SPACE暂停  R重置  ESC退出", (12, height - 24), (0, 0, 0))


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
    scenario: str = "random",
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

    # The interactive viewer is a long-running simulation. Keep the shorter
    # benchmark horizon for evaluation, but do not end a clean UI episode at
    # 600 logical frames (about one minute at the default 10 FPS).
    env = HighwayEnv(EnvironmentConfig(max_steps=3600))
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
    ego_y = 180 + int((env.ego.y if env.ego.y is not None else env.ego.lane * env.config.lane_width) * 120 / env.config.lane_width)
    pygame.draw.rect(screen, (53, 156, 255), (ego_x - 24, ego_y - 18, 48, 36), border_radius=8)
    pygame.draw.rect(screen, (197, 232, 255), (ego_x + 4, ego_y - 12, 13, 24), border_radius=3)

    for vehicle in env.vehicles:
        if vehicle.crashed:
            continue
        x, y = world_to_screen(vehicle.x, vehicle.lane, ego_x=ego_x)
        if 45 <= x <= 790:
            color = (235, 91, 74) if vehicle.crashed else (240, 169, 67)
            pygame.draw.rect(screen, color, (x - 20, y - 16, 40, 32), border_radius=7)

    _text(screen, title_font, "HYBRID DRIVING LAB", (42, 42), (237, 241, 245))
    _text(screen, small_font, "LIVE SIMULATION", (44, 75), (139, 161, 181))
    _panel_card(pygame, screen, (840, 28, 260, 116), "POLICY", POLICY_LABELS[policy_name], body_font, title_font)
    _panel_card(pygame, screen, (840, 158, 260, 150), "VEHICLE STATE", "", body_font, title_font)
    _metric(screen, body_font, "Speed", f"{_kmh(env.ego.speed):.1f} km/h", (858, 204))
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


def _metric(screen: Any, font: Any, label: str, value: str, position: tuple[int, int], color=(31, 52, 70)) -> None:
    _text(screen, font, label, position, color)
    _text(screen, font, value, (position[0] + 110, position[1]), color)


def _panel_heading(screen: Any, title_font: Any, body_font: Any, x: int, y: int, heading: str) -> None:
    _text(screen, title_font, heading, (x, y), (31, 52, 70))
    pygame = __import__("pygame")
    pygame.draw.line(screen, (220, 225, 231), (x, y + 32), (screen.get_width() - 24, y + 32), 1)


def _text(screen: Any, font: Any, value: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    screen.blit(font.render(value, True, color), position)
