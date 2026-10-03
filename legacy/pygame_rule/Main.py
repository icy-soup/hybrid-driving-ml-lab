"""
智驾仿真主程序 - 优化版
核心：后方生成快车、完善删除逻辑、速度比例优化
"""
import pygame
import random
from pathlib import Path
from ai_module import Action
from ai_module import RuleBasedAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets" / "pygame"

# 创建带数据收集功能的规则AI
ai_controller = RuleBasedAI(enable_data_collection=True)

current_controller = ai_controller

import enemy_ai_module

pygame.init()

# ==================== 窗口配置 ====================
GAME_WIDTH = 910
PANEL_WIDTH = 240
bg_width = GAME_WIDTH + PANEL_WIDTH
bg_height = 450
bg_color = (191, 191, 191)
FPS = 60

# ==================== 速度参数 ====================
init_speed = -8
base_speed = init_speed
prev_base_speed_abs = abs(base_speed)

max_speed = -36
min_speed = -6
speed_step = 0.12
BRAKE_STEP = 0.5
EMERGENCY_BRAKE_STEP = 2.5

# ==================== 车道参数 ====================
tail_y = 150
base_position_y = 50
LANE_CENTERS_BOTTOM = [tail_y + base_position_y * p for p in [1, 2, 3]]
LANE_CHANGE_SPEED = 6
ENEMY_LANE_CHANGE_SPEED = 3
LANE_CHANGE_DISTANCE = abs(LANE_CENTERS_BOTTOM[0] - LANE_CENTERS_BOTTOM[1])
LANE_CHANGE_FRAMES = (LANE_CHANGE_DISTANCE / LANE_CHANGE_SPEED) * 1.2
LANE_CHANGE_DURATION_SEC = LANE_CHANGE_FRAMES / FPS

# 同步 AI 模块中的车道配置
import ai_module
ai_module.LANE_CENTERS_BOTTOM = LANE_CENTERS_BOTTOM
ai_module.LANE_CHANGE_DURATION_SEC = LANE_CHANGE_DURATION_SEC
ai_module.LANE_CHANGE_DISTANCE = LANE_CHANGE_DISTANCE
ai_module.LANE_CHANGE_SPEED = LANE_CHANGE_SPEED

# 同步敌车 AI 模块配置
enemy_ai_module.LANE_CENTERS_BOTTOM = LANE_CENTERS_BOTTOM
enemy_ai_module.LANE_CHANGE_SPEED = LANE_CHANGE_SPEED
enemy_ai_module.LANE_CHANGE_DISTANCE = LANE_CHANGE_DISTANCE
enemy_ai_module.FPS = FPS

# ==================== 安全参数 ====================
STATIC_MARGIN_SEC = 0.5
MIN_DIST_THRESHOLD_BASE = 180
SAFE_REAR_DIST = 180
MIN_LANE_CHANGE_FRONT_DIST = 300
SAFE_FOLLOW_DIST = 250
TTC_DANGER_THRESHOLD = 3.0
SAFE_TIME_GAP = 2.0
TTC_EMERGENCY_THRESHOLD = 1.5

# ==================== 速度比例尺与策略 ====================
KM_H_PER_PIXEL_SPEED = 6.0

STRATEGIES = {
    "conservative": {"target_speed_kmh": 100, "dist_factor": 3.0, "enemy_interval_factor": 1.2},
    "balanced":     {"target_speed_kmh": 120, "dist_factor": 2.2, "enemy_interval_factor": 1.0},
    "aggressive":   {"target_speed_kmh": 150, "dist_factor": 1.5, "enemy_interval_factor": 0.8}
}

# ==================== 敌车生成 ====================
ENEMY_START_INTERVAL = 800
ENEMY_MIN_INTERVAL = 150
DECAY_FACTOR = 0.04
MAX_ENEMIES = 24

REAR_SPAWN_INTERVAL = 3000
REAR_SPAWN_CHANCE = 0.25
REAR_SPAWN_X = -150
REAR_REL_SPEED_MIN = 10
REAR_REL_SPEED_MAX = 18
FAST_CAR_INDICES = [0, 1, 6]

# 将原参数修改为：
NORMAL_REMOVE_LEFT = -400
NORMAL_REMOVE_RIGHT = GAME_WIDTH + 400
SLOW_REMOVE_LEFT = -350
SLOW_REMOVE_RIGHT = GAME_WIDTH + 350

# ==================== 全局状态 ====================
score = 0
kilometer = 0
cruise_control = False
STRATEGY_MODE = "conservative"
current_strategy = STRATEGIES[STRATEGY_MODE]
BASE_SPEED_LIMIT = -(current_strategy["target_speed_kmh"] / KM_H_PER_PIXEL_SPEED)
game_state = {"is_ai": False, "game_over": False}

# ==================== 资源加载 ====================
screen = pygame.display.set_mode((bg_width, bg_height), pygame.DOUBLEBUF)
clock = pygame.time.Clock()

BLACK, WHITE, GRAY, DARK_GRAY = (0,0,0), (255,255,255), (150,150,150), (60,60,60)
GREEN, RED, BLUE, ORANGE, CYAN, PURPLE = (0,255,0), (255,0,0), (0,100,255), (255,140,0), (0,255,255), (128,0,128)

def load_font(name, size):
    path = ASSET_ROOT / "font" / f"{name}.ttf"
    if path.exists():
        try:
            return pygame.font.Font(str(path), size)
        except:
            pass
    try:
        return pygame.font.SysFont(name, size, bold=(size > 40))
    except:
        return pygame.font.SysFont("arial", size, bold=(size > 40))

font_big = load_font("凤凰点阵体 16px", 60)
font_small = load_font("凤凰点阵体 12px", 20)
font_ui = load_font("凤凰点阵体 12px", 16)

def load_image(path, w, h, color):
    try:
        return pygame.image.load(str(ASSET_ROOT / path))
    except:
        surf = pygame.Surface((w, h))
        surf.fill(color)
        return surf

tail = load_image("images/others/车道.png", 910, 165, (150,150,150))
rail = load_image("images/others/白色护栏.png", 910, 60, (200,200,200))
Explosive = load_image("images/others/爆炸.png", 60, 60, (255,100,0))
Brake = load_image("images/others/break.png", 70, 60, (255,0,0))
Cruise = load_image("images/others/cuise.png", 70, 60, (0,255,0))
Opps = load_image("images/others/opps.png", 30, 48, (255,255,0))

# ==================== 车辆配置 ====================
cars = {
    "电动四轮车": ["黄色", "绿色", "黑色", "红色"],
    "小轿车": ["蓝色", "灰色", "绿色", "黄色"],
    "大卡车": ["红色", "蓝色", "银色"],
    "大巴车": ["红色", "蓝色", "绿色"],
    "面包车": ["黑色"],
    "卫星车": ["白色"],
    "跑车": ["红色", "黄色", "银色"]
}

car_sprites = {}
for car_type, colors in cars.items():
    car_sprites[car_type] = {}
    for color in colors:
        car_sprites[car_type][color] = {}
        for state in [0, 1, 2]:
            try:
                image_path = ASSET_ROOT / "images" / "vehicles" / f"{car_type}_{color}{state}.png"
                car_sprites[car_type][color][state] = pygame.image.load(str(image_path))
            except:
                surf = pygame.Surface((60, 40))
                surf.fill((255, 0, 0))
                car_sprites[car_type][color][state] = surf

vehicle_details = [
    [car_sprites["电动四轮车"], 60, 48, 3.2, 4],
    [car_sprites["小轿车"], 70, 44, 4.8, 11.2],
    [car_sprites["大卡车"], 80, 60, 4.8, 8],
    [car_sprites["大巴车"], 120, 72, 4.8, 8],
    [car_sprites["面包车"], 75, 55, 4.8, 9.6],
    [car_sprites["卫星车"], 70, 56, 4.8, 12],
    [car_sprites["跑车"], 80, 38, 14.4, 28]
]

# ==================== 工具函数 ====================
def get_lane_index_from_y(y):
    return min(range(3), key=lambda i: abs(y - LANE_CENTERS_BOTTOM[i]))

def is_lane_safe(lane_idx, player_lane, dist, ttc, rear_dist, rear_rel_speed,
                 trigger_ttc, dist_threshold, safe_rear_dist, min_front_dist):
    if dist != float('inf') and dist < min_front_dist:
        return False
    front_ok = (ttc > trigger_ttc) and (dist > dist_threshold)
    if not front_ok:
        return False
    if rear_dist < safe_rear_dist:
        return False
    return True

# ==================== 玩家类 ====================
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.info = vehicle_details[6]
        self.color = "银色"
        self.image_state = 1
        self.image = self.info[0][self.color][self.image_state]
        self.width = self.info[1]
        self.height = self.info[2]
        self.left = GAME_WIDTH / 2 - self.width / 2
        self.bottom = LANE_CENTERS_BOTTOM[1]
        self.collision_rect = pygame.Rect(self.left, self.bottom - 20, self.width, 28)
        self.moving = 1
        self.exploding = False
        self.target_lane_idx = 1
        self.is_changing_lane = False

    def _get_approx_lane(self):
        return get_lane_index_from_y(self.bottom)

    def update_position(self, action, is_ai):
        if is_ai:
            if not self.is_changing_lane:
                if action.up and self.target_lane_idx > 0:
                    self.target_lane_idx -= 1
                    self.is_changing_lane = True
                    self.moving = 0
                elif action.down and self.target_lane_idx < 2:
                    self.target_lane_idx += 1
                    self.is_changing_lane = True
                    self.moving = 2

            if self.is_changing_lane:
                target = LANE_CENTERS_BOTTOM[self.target_lane_idx]
                self.moving = 2 if self.bottom < target else 0
                diff = target - self.bottom
                if abs(diff) > 1:
                    step = LANE_CHANGE_SPEED if diff > 0 else -LANE_CHANGE_SPEED
                    self.bottom += step
                    if (diff > 0 and self.bottom >= target) or (diff < 0 and self.bottom <= target):
                        self.bottom = target
                        self.is_changing_lane = False
                        self.moving = 1
                else:
                    self.bottom = target
                    self.is_changing_lane = False
                    self.moving = 1
            else:
                self.moving = 1
        else:
            move = 8
            if action.up:
                self.bottom -= move
                self.moving = 0
            elif action.down:
                self.bottom += move
                self.moving = 2
            else:
                self.moving = 1
            self.bottom = max(tail_y + base_position_y, min(tail_y + base_position_y*3, self.bottom))
            self.target_lane_idx = self._get_approx_lane()
            self.is_changing_lane = False

        self.collision_rect.bottom = self.bottom
        self.collision_rect.left = self.left

    def Animation(self):
        self.image = self.info[0][self.color][self.moving]
        screen.blit(self.image, (self.left, self.bottom - self.height))
        if self.exploding:
            screen.blit(Explosive, (self.left + self.width/2 - 32, self.bottom - self.height/2 - 30))

# ==================== 敌人类 ====================
class Enemy(pygame.sprite.Sprite):
    def __init__(self, car_type_idx, position_x, lane_idx, is_rear_spawn=False):
        super().__init__()
        self.info = vehicle_details[car_type_idx]
        self.color = random.choice(list(self.info[0].keys()))
        self.image_state = 1
        self.image = self.info[0][self.color][self.image_state]
        self.width = self.info[1]
        self.height = self.info[2]

        # 速度相关
        if is_rear_spawn:
            self.base_relative_speed = random.uniform(REAR_REL_SPEED_MIN, REAR_REL_SPEED_MAX)
        else:
            speed_factor = 2.5
            self.base_relative_speed = (self.info[3] if position_x > GAME_WIDTH / 2 else self.info[4]) * speed_factor
            self.base_relative_speed += random.uniform(-0.5, 0.5)

        self.relative_speed = self.base_relative_speed
        self.speed = 0

        # 位置
        self.left = position_x
        self.bottom = LANE_CENTERS_BOTTOM[lane_idx]
        self.collision_rect = pygame.Rect(self.left, self.bottom - 20, self.width, 28)

        # 状态
        self.exploding = False
        self.crashed = False
        self.is_changing_lane = False
        self.target_bottom = self.bottom
        self.lane_change_direction = 0
        self.last_lane_change_time = -2000
        self.warning_timer = 0
        self.decision_to_change = None
        self.stuck_time = 0
        self.is_rear_spawn = is_rear_spawn
        self.has_overtaken = False          # 是否已被玩家超过（用于计分）

        # AI
        self.ai = enemy_ai_module.EnemyAI()

    def _get_lane_idx(self):
        return get_lane_index_from_y(self.bottom)

    def _start_lane_change(self, direction):
        self.is_changing_lane = True
        self.lane_change_direction = direction
        my_lane = self._get_lane_idx()
        target_idx = my_lane + direction
        if 0 <= target_idx <= 2:
            self.target_bottom = LANE_CENTERS_BOTTOM[target_idx]
        else:
            self.is_changing_lane = False

    def _perform_lane_change(self):
        diff = self.target_bottom - self.bottom
        if abs(diff) > 1:
            step = ENEMY_LANE_CHANGE_SPEED if diff > 0 else -ENEMY_LANE_CHANGE_SPEED
            self.bottom += step
            if (diff > 0 and self.bottom >= self.target_bottom) or \
               (diff < 0 and self.bottom <= self.target_bottom):
                self.bottom = self.target_bottom
                self.is_changing_lane = False
                self.lane_change_direction = 0
        else:
            self.bottom = self.target_bottom
            self.is_changing_lane = False

    def move(self, all_vehicles, current_base_speed, player):
        global score
        remove_left = SLOW_REMOVE_LEFT if abs(current_base_speed) < 10 else NORMAL_REMOVE_LEFT
        remove_right = SLOW_REMOVE_RIGHT if abs(current_base_speed) < 10 else NORMAL_REMOVE_RIGHT

        # 坏车移动
        if self.crashed:
            self.left += current_base_speed
            self.collision_rect.left = self.left
            self.collision_rect.bottom = self.bottom
            if self.left <= remove_left or self.left >= remove_right:
                self.kill()
            return

        # AI 决策
        decision = self.ai.decide(all_vehicles, self, current_base_speed, pygame.time.get_ticks())

        # 紧急制动
        if decision["is_emergency"]:
            self.relative_speed = decision["target_rel_speed"]
            self.speed = current_base_speed + self.relative_speed
            self.left += self.speed
            self.collision_rect.left = self.left
            self.collision_rect.bottom = self.bottom
            self.warning_timer = decision["warning_timer"]
            self.ai.warning_timer = decision["warning_timer"]
            if self.left <= remove_left or self.left >= remove_right:
                self.kill()
            return

        # 速度控制
        target_rel = decision["target_rel_speed"]
        if abs(self.relative_speed - target_rel) > 0.08:
            if self.relative_speed < target_rel:
                self.relative_speed = min(target_rel, self.relative_speed + 0.08)
            else:
                self.relative_speed = max(target_rel, self.relative_speed - 0.08)
        else:
            self.relative_speed = target_rel

        # 移动
        self.speed = current_base_speed + self.relative_speed
        self.left += self.speed
        self.collision_rect.left = self.left
        self.collision_rect.bottom = self.bottom

        # 变道处理
        if decision["need_change"] and not self.is_changing_lane:
            self._start_lane_change(decision["change_dir"])
            self.last_lane_change_time = pygame.time.get_ticks()
        if self.is_changing_lane:
            self._perform_lane_change()

        # 感叹号显示
        self.warning_timer = decision["warning_timer"]
        self.ai.warning_timer = decision["warning_timer"]

        # 边界删除
        if self.left <= remove_left or self.left >= remove_right:
            self.kill()

    def Animation(self):
        screen.blit(self.image, (self.left, self.bottom - self.height))
        if self.exploding or self.crashed:
            screen.blit(Explosive, (self.left + self.width/2 - 32, self.bottom - self.height/2 - 30))
        if self.warning_timer > 0:
            opps_scaled = pygame.transform.scale(Opps, (24, 38))
            screen.blit(opps_scaled, (self.left + self.width/2 - 12, self.bottom - self.height - 38))

# ==================== 控制器 ====================
class ManualController:
    def __init__(self):
        self.keys = {pygame.K_UP: False, pygame.K_DOWN: False,
                     pygame.K_RIGHT: False, pygame.K_LEFT: False,
                     pygame.K_SPACE: False}

    def handle_event(self, event):
        if event.type in (pygame.KEYDOWN, pygame.KEYUP) and event.key in self.keys:
            self.keys[event.key] = (event.type == pygame.KEYDOWN)

    def get_action(self, player, enemies, current_speed, strategy_mode="balanced"):
        global cruise_control
        action = Action()
        action.up = self.keys[pygame.K_UP]
        action.down = self.keys[pygame.K_DOWN]
        if any(self.keys[k] for k in (pygame.K_UP, pygame.K_DOWN, pygame.K_RIGHT, pygame.K_LEFT)):
            cruise_control = False
        if self.keys[pygame.K_RIGHT]:
            action.accelerate = 1
        elif self.keys[pygame.K_LEFT]:
            action.accelerate = -1
        else:
            action.accelerate = 0
        action.braking = (action.accelerate == -1)
        action.is_emergency_brake = False
        action.debug_info = "Manual"
        return action

# ==================== 速度控制 ====================
def player_accelerate_logic(accelerate_flag, follow_override=None, dynamic_limit_override=None, is_emergency=False):
    global base_speed, cruise_control, game_state, STRATEGY_MODE, current_strategy, BASE_SPEED_LIMIT

    if game_state["is_ai"]:
        if is_emergency:
            base_speed = max(max_speed, base_speed + EMERGENCY_BRAKE_STEP)
            return

        target = None
        if follow_override is not None:
            target = follow_override
        elif dynamic_limit_override is not None:
            target = dynamic_limit_override
        else:
            target = BASE_SPEED_LIMIT

        target = max(max_speed, min(min_speed, target))

        if abs(abs(base_speed) - abs(target)) < 0.08:
            base_speed = target
            return
        if abs(base_speed) < abs(target):
            base_speed = max(max_speed, base_speed - speed_step)
        else:
            base_speed = min(min_speed, base_speed + speed_step)
            if abs(abs(base_speed) - abs(target)) < 0.08:
                base_speed = target
        return

    # 手动模式
    if cruise_control:
        if accelerate_flag != 0:
            cruise_control = False
    if accelerate_flag == 0:
        if not cruise_control:
            if base_speed - init_speed < -0.1: base_speed += speed_step
            if base_speed - init_speed > 0.1: base_speed -= speed_step
    elif accelerate_flag == 1:
        base_speed = max(max_speed, base_speed - speed_step)
    elif accelerate_flag == -1:
        base_speed = min(min_speed, base_speed + BRAKE_STEP)

    base_speed = max(max_speed, min(min_speed, base_speed))

def player_accelerate_wrapper(self, flag, follow_override=None, dynamic_limit_override=None, is_emergency=False):
    player_accelerate_logic(flag, follow_override, dynamic_limit_override, is_emergency)

Player.Accelerate = player_accelerate_wrapper

# ==================== UI 绘制 ====================
def draw_ui_panel(action, player, current_interval):
    pygame.draw.rect(screen, DARK_GRAY, (GAME_WIDTH, 0, PANEL_WIDTH, bg_height))
    pygame.draw.line(screen, WHITE, (GAME_WIDTH, 0), (GAME_WIDTH, bg_height), 3)

    screen.blit(font_small.render("控制中心", True, WHITE), (GAME_WIDTH + 80, 3))

    mode_txt = "AI 自动驾驶" if game_state["is_ai"] else "人工手动驾驶"
    mode_clr = GREEN if game_state["is_ai"] else BLUE
    screen.blit(font_small.render(mode_txt, True, mode_clr), (GAME_WIDTH + 55, 30))

    switch_rect = pygame.Rect(GAME_WIDTH + 20, 60, 200, 50)
    strategy_rect = pygame.Rect(GAME_WIDTH + 20, 120, 200, 50)

    btn_clr = BLUE if not game_state["is_ai"] else ORANGE
    pygame.draw.rect(screen, btn_clr, switch_rect)
    pygame.draw.rect(screen, WHITE, switch_rect, 2)
    txt = font_ui.render("切换为：手动驾驶" if game_state["is_ai"] else "切换为：AI 驾驶", True, WHITE)
    screen.blit(txt, (switch_rect.x + 20, switch_rect.y + 15))

    pygame.draw.rect(screen, PURPLE, strategy_rect)
    pygame.draw.rect(screen, WHITE, strategy_rect, 2)
    screen.blit(font_ui.render(f"策略：{STRATEGY_MODE}", True, WHITE), (strategy_rect.x + 15, strategy_rect.y + 15))

    indicator_rect = pygame.Rect(GAME_WIDTH + 70, 180, 100, 80)
    pygame.draw.rect(screen, GRAY, indicator_rect)
    pygame.draw.rect(screen, WHITE, indicator_rect, 2)

    if not game_state["game_over"]:
        if action.is_emergency_brake or action.braking:
            indicator_scaled = pygame.transform.scale(Brake, (indicator_rect.width, indicator_rect.height))
            screen.blit(indicator_scaled, indicator_rect)
        elif game_state["is_ai"] and action.accelerate == 0 and action.follow_speed_override is None:
            indicator_scaled = pygame.transform.scale(Cruise, (indicator_rect.width, indicator_rect.height))
            screen.blit(indicator_scaled, indicator_rect)
        elif not game_state["is_ai"] and cruise_control and action.accelerate == 0:
            indicator_scaled = pygame.transform.scale(Cruise, (indicator_rect.width, indicator_rect.height))
            screen.blit(indicator_scaled, indicator_rect)

    info_y = 270
    line_h = 18
    if game_state["is_ai"] and not game_state["game_over"]:
        density = '高' if current_interval < 1000 else '中' if current_interval < 1500 else '低'
        screen.blit(font_ui.render(f"变道速度：{LANE_CHANGE_SPEED} px/f", True, ORANGE),
                    (GAME_WIDTH + 10, info_y))
        screen.blit(font_ui.render(f"当前距离：{action.dist_curr:.0f}px", True, WHITE),
                    (GAME_WIDTH + 10, info_y + line_h))
        screen.blit(font_ui.render(f"状态：{action.debug_info}", True, WHITE),
                    (GAME_WIDTH + 10, info_y + 2*line_h))
        screen.blit(font_ui.render(f"车流密度：{density} ({current_interval:.0f}ms)", True, GREEN),
                    (GAME_WIDTH + 10, info_y + 3*line_h))
    elif game_state["game_over"]:
        screen.blit(font_ui.render("游戏已结束", True, RED), (GAME_WIDTH + 10, info_y))
        screen.blit(font_ui.render("按任意键重启", True, WHITE), (GAME_WIDTH + 10, info_y + line_h))
    else:
        cc = " (巡航中)" if cruise_control else ""
        screen.blit(font_ui.render(f"手动模式 {cc}", True, BLUE), (GAME_WIDTH + 10, info_y))

    lane_info = f"目标车道：{player.target_lane_idx + 1}"
    status = "变道中 (慢)..." if player.is_changing_lane else "稳定行驶"
    s_color = GREEN if player.is_changing_lane else WHITE
    screen.blit(font_ui.render(lane_info, True, WHITE), (GAME_WIDTH + 10, 400))
    screen.blit(font_ui.render(status, True, s_color), (GAME_WIDTH + 10, 420))

    return switch_rect, strategy_rect

# ==================== 游戏重置 ====================
def reset_game(player, enemies, all_sprites):
    global score, kilometer, base_speed, cruise_control, last_enemy_time, last_rear_spawn_time
    player.exploding = False
    player.bottom = LANE_CENTERS_BOTTOM[1]
    player.target_lane_idx = 1
    player.is_changing_lane = False
    player.left = GAME_WIDTH / 2 - player.width / 2
    player.collision_rect.left = player.left

    for e in enemies:
        e.exploding = False
        e.crashed = False
        e.has_overtaken = False   # 重置超车标志

    enemies.empty()
    all_sprites.empty()
    all_sprites.add(player)

    score = 0
    kilometer = 0
    base_speed = init_speed
    cruise_control = False
    last_enemy_time = pygame.time.get_ticks()
    last_rear_spawn_time = pygame.time.get_ticks()

# ==================== 主循环 ====================
player = Player()
enemies = pygame.sprite.Group()
all_sprites = pygame.sprite.Group()
all_sprites.add(player)

manual_controller = ManualController()
ai_controller = RuleBasedAI()
current_controller = manual_controller

last_enemy_time = pygame.time.get_ticks()
last_rear_spawn_time = pygame.time.get_ticks()
current_interval = ENEMY_START_INTERVAL
last_action = Action()
prev_base_speed_abs = abs(base_speed)
game_over_text = font_big.render("游戏结束", True, BLACK)
game_over_rect = game_over_text.get_rect(center=(GAME_WIDTH / 2, bg_height / 2))

tail_x1 = tail_x2 = 0
running = True

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if not game_state["game_over"]:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                switch_rect, strategy_rect = draw_ui_panel(last_action, player, current_interval)
                if switch_rect.collidepoint(event.pos):
                    game_state["is_ai"] = not game_state["is_ai"]
                    cruise_control = False
                    if game_state["is_ai"]:
                        current_controller = ai_controller
                        player.target_lane_idx = player._get_approx_lane()
                        player.bottom = LANE_CENTERS_BOTTOM[player.target_lane_idx]
                        player.is_changing_lane = False
                    else:
                        current_controller = manual_controller

                if strategy_rect.collidepoint(event.pos):
                    modes = list(STRATEGIES.keys())
                    current_idx = modes.index(STRATEGY_MODE)
                    STRATEGY_MODE = modes[(current_idx + 1) % len(modes)]
                    current_strategy = STRATEGIES[STRATEGY_MODE]
                    BASE_SPEED_LIMIT = -(current_strategy["target_speed_kmh"] / KM_H_PER_PIXEL_SPEED)
                    pygame.display.set_caption(
                        f"智驾 - {STRATEGY_MODE.upper()} (限速:{current_strategy['target_speed_kmh']}km/h)")
                    cruise_control = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not game_state["is_ai"]:
                cruise_control = not cruise_control

            if isinstance(current_controller, ManualController):
                current_controller.handle_event(event)

    if not game_state["game_over"]:
        current_speed_abs = abs(base_speed)

        action = current_controller.get_action(player, enemies, base_speed, STRATEGY_MODE)
        if not action.is_emergency_brake and current_speed_abs < prev_base_speed_abs:
            action.braking = True
        elif not action.is_emergency_brake:
            action.braking = False
        last_action = action

        player.update_position(action, game_state["is_ai"])
        player.Accelerate(action.accelerate, action.follow_speed_override,
                          action.dynamic_limit_override, action.is_emergency_brake)

        prev_base_speed_abs = current_speed_abs

        # ========== 超车计分：玩家超过敌车 ==========
        for enemy in enemies:
            if not enemy.has_overtaken and player.collision_rect.left > enemy.collision_rect.right:
                score += 1
                enemy.has_overtaken = True

        tail_speed = base_speed
        tail_x1 = (tail_x1 + tail_speed) % GAME_WIDTH - GAME_WIDTH
        tail_x2 = (tail_x2 + tail_speed) % GAME_WIDTH

        current_time = pygame.time.get_ticks()
        enemy_count = len(enemies)
        density_factor = 1 + enemy_count / MAX_ENEMIES

        current_interval_factor = current_strategy.get("enemy_interval_factor", 1.0)

        adjusted_interval = ENEMY_MIN_INTERVAL + (ENEMY_START_INTERVAL - ENEMY_MIN_INTERVAL) / (1 + score * DECAY_FACTOR)
        adjusted_interval *= density_factor
        adjusted_interval *= current_interval_factor
        adjusted_interval = max(ENEMY_MIN_INTERVAL, min(ENEMY_START_INTERVAL * 1.2, adjusted_interval))

        # 速度因子：速度越快，生成间隔越小（车越多）
        current_kmh = abs(base_speed) * KM_H_PER_PIXEL_SPEED
        target_kmh = current_strategy["target_speed_kmh"]
        speed_ratio = min(1.5, max(0.5, current_kmh / target_kmh))
        interval_factor = 1.0 / speed_ratio
        adjusted_interval = adjusted_interval * interval_factor
        adjusted_interval = max(ENEMY_MIN_INTERVAL, min(ENEMY_START_INTERVAL * 1.5, adjusted_interval))

        if enemy_count < MAX_ENEMIES and current_time - last_enemy_time >= adjusted_interval:
            vehicle = random.randint(0, 5)
            lane_idx = random.randint(0, 2)
            new_enemy = Enemy(vehicle, GAME_WIDTH + 50, lane_idx, is_rear_spawn=False)
            enemies.add(new_enemy)
            all_sprites.add(new_enemy)
            last_enemy_time = current_time
            current_interval = adjusted_interval

        if enemy_count < MAX_ENEMIES and current_time - last_rear_spawn_time >= REAR_SPAWN_INTERVAL:
            spawn_prob = REAR_SPAWN_CHANCE * current_interval_factor
            if random.random() < spawn_prob:
                vehicle = random.choice(FAST_CAR_INDICES)
                lane_idx = random.randint(0, 2)
                new_enemy = Enemy(vehicle, REAR_SPAWN_X, lane_idx, is_rear_spawn=True)
                enemies.add(new_enemy)
                all_sprites.add(new_enemy)
                last_rear_spawn_time = current_time

        all_vehicles = list(enemies) + [player]
        for enemy in enemies:
            enemy.move(all_vehicles, base_speed, player)

        collided_enemy = None
        for enemy in enemies:
            if player.collision_rect.colliderect(enemy.collision_rect):
                collided_enemy = enemy
                break
        if collided_enemy:
            player.exploding = True
            collided_enemy.exploding = True
            collided_enemy.crashed = True
            game_state["game_over"] = True

            # 保存数据
            if hasattr(ai_controller, 'data_collector') and ai_controller.data_collector:
                ai_controller.data_collector.save("training_data.npz")
                print(f"已保存 {len(ai_controller.data_collector.states)} 条训练数据")

        # 敌人间碰撞检测
        enemies_list = list(enemies)
        for i in range(len(enemies_list)):
            for j in range(i + 1, len(enemies_list)):
                e1 = enemies_list[i]
                e2 = enemies_list[j]

                # 检查两车是否发生碰撞
                if e1.collision_rect.colliderect(e2.collision_rect):
                    # 定义屏幕外判断函数
                    def is_fully_out(rect):
                        return rect.right < 0 or rect.left > GAME_WIDTH


                    # 如果至少一辆车完全在屏幕外，直接删除两车
                    if is_fully_out(e1.collision_rect) or is_fully_out(e2.collision_rect):
                        e1.kill()
                        e2.kill()
                    else:
                        # 两车都在屏幕内，正常碰撞处理
                        if e1.crashed or e2.crashed:
                            e1.crashed = True
                            e2.crashed = True
                            e1.exploding = True
                            e2.exploding = True
                        else:
                            e1.exploding = True
                            e2.exploding = True
                            e1.crashed = True
                            e2.crashed = True

    screen.fill(bg_color)
    screen.blit(tail, (tail_x1, tail_y))
    screen.blit(tail, (tail_x2, tail_y))
    screen.blit(rail, (tail_x1, 85))
    screen.blit(rail, (tail_x2, 85))
    screen.blit(rail, (tail_x1, 305))
    screen.blit(rail, (tail_x2, 305))

    for sprite in sorted(all_sprites, key=lambda s: s.bottom):
        sprite.Animation()

    screen.blit(font_small.render(f"超车：{score}", True, BLACK), (10, 10))
    kilometer -= base_speed * 0.1
    screen.blit(font_small.render(f"距离：{int(kilometer)}m", True, BLACK), (10, 40))

    current_kmh = abs(base_speed) * KM_H_PER_PIXEL_SPEED
    current_limit_kmh = abs(last_action.dynamic_limit_override) * KM_H_PER_PIXEL_SPEED if game_state["is_ai"] else 0
    target_kmh = current_strategy["target_speed_kmh"]

    if current_kmh > target_kmh * 1.05:
        speed_color = RED
    elif game_state["is_ai"] and current_kmh >= current_limit_kmh * 0.95:
        speed_color = RED
    else:
        speed_color = BLACK
    screen.blit(font_small.render(f"时速：{current_kmh:.1f} km/h", True, speed_color), (10, 70))

    if cruise_control:
        screen.blit(font_ui.render("定速巡航 ON (SPACE)", True, CYAN), (10, 95))

    if game_state["game_over"]:
        screen.blit(game_over_text, game_over_rect)

    draw_ui_panel(last_action, player, current_interval)
    pygame.display.flip()

    if game_state["game_over"]:
        waiting = True
        while waiting:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                    waiting = False
                if ev.type == pygame.KEYDOWN:
                    game_state["game_over"] = False
                    reset_game(player, enemies, all_sprites)
                    waiting = False

pygame.quit()
