"""
AI 决策模块 - 完整版（修复减速失效、增加队列优化）
"""
import random
import numpy as np
from collections import deque

# ==================== 配置 ====================
LANE_CENTERS_BOTTOM = [200, 250, 300]
LANE_CHANGE_DURATION_SEC = 1.0
LANE_CHANGE_DISTANCE = 0
LANE_CHANGE_SPEED = 6
KM_H_PER_PIXEL_SPEED = 6.0

MIN_SPEED_PIXEL = -6
MAX_SPEED_PIXEL = -36
SAFE_FOLLOW_DIST = 250
LANE_CHANGE_MIN_DIST = 320
LANE_CHANGE_MIN_TTC = 2.5
SIDE_CHECK_REAR_DIST = 160
MAX_DECEL = 0.25
REACTION_FRAMES = 5
SPEED_SMOOTHING = 0.3

STRATEGIES = {
    "conservative": {"target_speed_kmh": 100, "dist_factor": 3.0, "overtake_aggressiveness": 0.5},
    "balanced":     {"target_speed_kmh": 120, "dist_factor": 2.2, "overtake_aggressiveness": 1.0},
    "aggressive":   {"target_speed_kmh": 150, "dist_factor": 1.5, "overtake_aggressiveness": 1.5}
}

class Action:
    def __init__(self):
        self.up = False
        self.down = False
        self.accelerate = 0
        self.braking = False
        self.is_emergency_brake = False
        self.follow_speed_override = None
        self.dynamic_limit_override = None
        self.debug_info = ""
        self.dist_curr = 0.0
        self.ttc_curr = 0.0
        self.is_pinched = False
        self.side_vehicle_intent = {"left": 0, "right": 0}

# ==================== AI 基类 ====================
class AIBase:
    def get_action(self, player, enemies, current_speed, strategy_mode="balanced"):
        raise NotImplementedError
    def update_history(self, state, action, reward, done=False):
        pass
    def reset(self):
        pass
    def save_model(self, path):
        pass
    def load_model(self, path):
        pass

# ==================== 规则AI ====================
class RuleBasedAI(AIBase):
    def __init__(self, enable_data_collection=False):
        self.brake_hold = 0
        self.change_hold = 0
        self.overtake_timer = 0
        self.overtaking = False
        self.data_collector = None

    # ---------- 物理计算 ----------
    def _calc_emergency_brake_dist(self, rel_speed, ego_speed_abs):
        if rel_speed <= 0:
            return 50
        reaction_dist = rel_speed * REACTION_FRAMES
        brake_dist = (rel_speed ** 2) / (2 * MAX_DECEL)
        return reaction_dist + brake_dist + 40

    def _calc_lane_change_dist(self, ego_speed, front_speed, strategy_factor):
        rel_speed = ego_speed - front_speed
        if rel_speed <= 0:
            return 200
        lane_change_time = LANE_CHANGE_DISTANCE / LANE_CHANGE_SPEED
        required = rel_speed * lane_change_time + 50
        required *= strategy_factor
        return max(required, 200)

    # ---------- 感知（包含障碍物） ----------
    def _perceive(self, player, enemies, current_speed):
        lane_idx = player._get_approx_lane()
        front = [{"dist": float('inf'), "ttc": float('inf'), "speed": None, "crashed": False} for _ in range(3)]
        side_rear = [False, False, False]

        for enemy in enemies:
            # 获取车道
            v_lane = -1
            if hasattr(enemy, '_get_approx_lane'):
                v_lane = enemy._get_approx_lane()
            else:
                v_lane = enemy._get_lane_idx()
            if v_lane == -1:
                continue

            d_x = enemy.left - player.left

            # 速度：crashed车辆视为静止（速度0）
            if hasattr(enemy, 'crashed') and enemy.crashed:
                v_speed = 0
                is_crashed = True
            else:
                v_speed = current_speed + enemy.relative_speed
                is_crashed = False

            if d_x > 0 and d_x < front[v_lane]["dist"]:
                front[v_lane]["dist"] = d_x
                front[v_lane]["speed"] = v_speed
                front[v_lane]["crashed"] = is_crashed
                closing = current_speed - v_speed
                front[v_lane]["ttc"] = d_x / closing if closing > 1.0 else float('inf')

            if -SIDE_CHECK_REAR_DIST < d_x < 0:
                side_rear[v_lane] = True

        return lane_idx, front, side_rear

    def _should_overtake(self, lane_idx, front, current_speed, strategy):
        if self.overtaking:
            return False
        if front[lane_idx]["speed"] is None or front[lane_idx]["crashed"]:
            return False
        speed_diff = current_speed - front[lane_idx]["speed"]
        if speed_diff < 1.0:
            return False
        overtake_dist_threshold = 300 / strategy.get("overtake_aggressiveness", 1.0)
        if front[lane_idx]["dist"] > overtake_dist_threshold:
            return False
        left_safe = (lane_idx > 0) and self._is_lane_safe_for_overtake(lane_idx-1, front, current_speed)
        right_safe = (lane_idx < 2) and self._is_lane_safe_for_overtake(lane_idx+1, front, current_speed)
        return left_safe or right_safe

    def _is_lane_safe_for_overtake(self, target_lane, front, current_speed):
        if front[target_lane]["dist"] == float('inf'):
            return True
        return front[target_lane]["dist"] > 400

    def _execute_overtake(self, action, target_lane, current_speed, target_speed_kmh):
        action.up = (target_lane == 0)
        action.down = (target_lane == 2)
        action.follow_speed_override = -(target_speed_kmh / KM_H_PER_PIXEL_SPEED) * 1.2
        action.accelerate = 1
        action.debug_info = "Overtaking!"
        self.overtaking = True
        self.overtake_timer = 60

    # ---------- 核心决策 ----------
    def get_action(self, player, enemies, current_speed, strategy_mode="balanced"):
        action = Action()
        strategy = STRATEGIES.get(strategy_mode, STRATEGIES["balanced"])
        speed_abs = abs(current_speed)

        # 更新状态
        if self.brake_hold > 0:
            self.brake_hold -= 1
        if self.change_hold > 0:
            self.change_hold -= 1
        if self.overtake_timer > 0:
            self.overtake_timer -= 1
            if self.overtake_timer == 0:
                self.overtaking = False

        # 感知
        lane_idx, front, side_rear = self._perceive(player, enemies, current_speed)

        # UI数据
        action.dist_curr = front[lane_idx]["dist"] if front[lane_idx]["dist"] != float('inf') else 0
        action.ttc_curr = front[lane_idx]["ttc"] if front[lane_idx]["ttc"] != float('inf') else 9.99
        action.dynamic_limit_override = -(strategy["target_speed_kmh"] / KM_H_PER_PIXEL_SPEED)

        if player.is_changing_lane:
            action.debug_info = "Changing"
            return action

        # ========== 紧急制动 ==========
        curr_dist = front[lane_idx]["dist"]
        curr_ttc = front[lane_idx]["ttc"]
        if front[lane_idx]["speed"] is not None:
            rel_speed = current_speed - front[lane_idx]["speed"]
            if rel_speed > 0:
                emergency_dist = self._calc_emergency_brake_dist(rel_speed, speed_abs)
                if curr_dist < emergency_dist:
                    self.brake_hold = 12
                    action.is_emergency_brake = True
                    action.accelerate = -1
                    action.braking = True
                    action.debug_info = "!!! EMERGENCY !!!"
                    return action
        if curr_ttc != float('inf') and curr_ttc < 1.2:
            self.brake_hold = 12
            action.is_emergency_brake = True
            action.accelerate = -1
            action.braking = True
            action.debug_info = "!!! EMERGENCY !!!"
            return action
        if self.brake_hold > 0:
            action.is_emergency_brake = True
            action.accelerate = -1
            action.braking = True
            action.debug_info = "!!! BRAKE HOLD !!!"
            return action

        # ========== 超车逻辑 ==========
        if not self.overtaking and self._should_overtake(lane_idx, front, current_speed, strategy):
            target_lane = -1
            if lane_idx > 0 and self._is_lane_safe_for_overtake(lane_idx-1, front, current_speed):
                target_lane = lane_idx - 1
            elif lane_idx < 2 and self._is_lane_safe_for_overtake(lane_idx+1, front, current_speed):
                target_lane = lane_idx + 1
            if target_lane != -1:
                self._execute_overtake(action, target_lane, current_speed, strategy["target_speed_kmh"])
                return action

        # ========== 夹击检测 ==========
        front_threat = curr_dist < 400
        left_pinch = lane_idx > 0 and side_rear[lane_idx - 1]
        right_pinch = lane_idx < 2 and side_rear[lane_idx + 1]
        action.is_pinched = front_threat and (left_pinch or right_pinch)

        if action.is_pinched:
            if front[lane_idx]["speed"] is not None:
                action.follow_speed_override = max(MIN_SPEED_PIXEL, front[lane_idx]["speed"] - 0.3)
            else:
                action.follow_speed_override = max(MIN_SPEED_PIXEL, current_speed * 0.88)
            action.braking = True
            action.accelerate = 0
            action.debug_info = "Pinched - Hold"
            return action

            # ========== 跟车逻辑（改进：缓冲区间 + 平滑跟随） ==========
            curr_dist = front[lane_idx]["dist"]
            front_vehicle_speed = front[lane_idx]["speed"]
            front_crashed = front[lane_idx]["crashed"]

            # 动态安全距离（最小250，与速度成正比）
            safe_dist = max(250, abs(current_speed) * 2.5)
            # 缓冲区间系数（0.9 ~ 1.1）
            lower_bound = safe_dist * 0.9
            upper_bound = safe_dist * 1.1

            if front_vehicle_speed is not None:
                if front_crashed:
                    # 遇到障碍物，减速至最低速
                    target_speed = min_speed
                    new_speed = current_speed + (target_speed - current_speed) * 0.3
                    action.follow_speed_override = new_speed
                    action.braking = True
                    action.accelerate = 0
                    action.debug_info = "Stop for obstacle"
                    return action

                if curr_dist < lower_bound:
                    # 距离过近，减速到前车速度 - 0.2
                    target_speed = front_vehicle_speed - 0.2
                    target_speed = max(min_speed, target_speed)
                    # 平滑减速
                    new_speed = current_speed + (target_speed - current_speed) * 0.3
                    action.follow_speed_override = new_speed
                    action.braking = True
                    action.accelerate = 0
                    action.debug_info = f"Follow (brake) @{abs(target_speed) * KM_H_PER_PIXEL_SPEED:.0f}"
                    return action
                elif curr_dist > upper_bound:
                    # 距离较远，可以加速到策略限速（让巡航逻辑处理）
                    pass
                else:
                    # 在缓冲区内，维持当前速度（不操作）
                    action.accelerate = 0
                    action.debug_info = "Hold"
                    return action

        # ========== 变道决策 ==========
        strategy_factor = strategy.get("dist_factor", 2.0) / 2.0
        dynamic_dist = self._calc_lane_change_dist(speed_abs, front[lane_idx]["speed"] or 0, strategy_factor)
        min_safe_dist = min(dynamic_dist, LANE_CHANGE_MIN_DIST)

        need_change = (curr_dist < min_safe_dist or curr_ttc < LANE_CHANGE_MIN_TTC)

        if need_change and self.change_hold == 0:
            left_ok = False
            right_ok = False

            if lane_idx > 0:
                left_dist = front[lane_idx - 1]["dist"]
                left_has_rear = side_rear[lane_idx - 1]
                left_ok = (left_dist > LANE_CHANGE_MIN_DIST or left_dist == float('inf')) and not left_has_rear

            if lane_idx < 2:
                right_dist = front[lane_idx + 1]["dist"]
                right_has_rear = side_rear[lane_idx + 1]
                right_ok = (right_dist > LANE_CHANGE_MIN_DIST or right_dist == float('inf')) and not right_has_rear

            if left_ok and right_ok:
                left_d = front[lane_idx - 1]["dist"] if front[lane_idx - 1]["dist"] != float('inf') else 9999
                right_d = front[lane_idx + 1]["dist"] if front[lane_idx + 1]["dist"] != float('inf') else 9999
                if left_d > right_d:
                    action.up = True
                    action.debug_info = "Shift L"
                else:
                    action.down = True
                    action.debug_info = "Shift R"
            elif left_ok:
                action.up = True
                action.debug_info = "Shift L"
            elif right_ok:
                action.down = True
                action.debug_info = "Shift R"
            else:
                action.debug_info = "No Lane - Follow"

            if action.up or action.down:
                self.change_hold = 25
                return action

        # ========== 正常巡航 ==========
        target_speed = action.dynamic_limit_override
        if speed_abs < abs(target_speed):
            action.accelerate = 1
            action.debug_info = f"Accel @{abs(target_speed)*KM_H_PER_PIXEL_SPEED:.0f}"
        else:
            action.accelerate = 0
            action.debug_info = f"Cruise @{abs(target_speed)*KM_H_PER_PIXEL_SPEED:.0f}"

        return action

    def reset(self):
        self.brake_hold = 0
        self.change_hold = 0
        self.overtake_timer = 0
        self.overtaking = False

# ==================== 神经网络AI占位 ====================
class NeuralAI(AIBase):
    def __init__(self, model_path=None):
        self.model = None
    def get_action(self, player, enemies, current_speed, strategy_mode="balanced"):
        action = Action()
        action.accelerate = 0
        action.debug_info = "NeuralAI (demo)"
        return action