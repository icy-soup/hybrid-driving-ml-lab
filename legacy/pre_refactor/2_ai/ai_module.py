"""
AI 决策模块 - 统一版（规则AI + 神经网络AI + 自动训练）
支持动态切换、数据收集、自动训练
"""
import random
import numpy as np
import pickle
import os
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
    """AI决策输出"""
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
        self.strategy_mode = "balanced"


# ==================== 数据收集器 ====================
class DataCollector:
    def __init__(self, max_size=100000):
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)

    def add(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    def save(self, filename="training_data.npz"):
        if len(self.states) > 0:
            np.savez_compressed(filename,
                                states=np.array(self.states),
                                actions=np.array(self.actions))
            print(f"已保存 {len(self.states)} 条训练数据到 {filename}")
            return True
        return False

    def get_count(self):
        return len(self.states)

    def clear(self):
        self.states.clear()
        self.actions.clear()


# ==================== 神经网络模型 ====================
class SimpleNeuralNetwork:
    def __init__(self, input_size=11, hidden_size=32, output_size=5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_z = np.exp(self.z2 - np.max(self.z2))
        self.probs = exp_z / np.sum(exp_z)
        return self.probs

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.W1, self.b1, self.W2, self.b2), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.W1, self.b1, self.W2, self.b2 = pickle.load(f)


# ==================== 规则AI（带数据收集） ====================
class RuleBasedAI:
    """基于规则的AI，实现稳定的跟车、变道、超车逻辑，可选数据收集"""
    def __init__(self, enable_collection=False):
        self.brake_hold = 0
        self.change_hold = 0
        self.overtake_timer = 0
        self.overtaking = False
        self.data_collector = DataCollector() if enable_collection else None

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

    def _perceive(self, player, enemies, current_speed):
        lane_idx = player._get_approx_lane()
        front = [{"dist": float('inf'), "ttc": float('inf'), "speed": None, "crashed": False} for _ in range(3)]
        side_rear = [False, False, False]

        for enemy in enemies:
            v_lane = -1
            if hasattr(enemy, '_get_approx_lane'):
                v_lane = enemy._get_approx_lane()
            else:
                v_lane = enemy._get_lane_idx()
            if v_lane == -1:
                continue

            d_x = enemy.left - player.left
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

    def extract_features(self, player, enemies, current_speed):
        """提取状态特征向量（11维）"""
        lane_idx = player._get_approx_lane()
        _, front, side_rear = self._perceive(player, enemies, current_speed)

        norm_speed = (current_speed - MAX_SPEED_PIXEL) / (MIN_SPEED_PIXEL - MAX_SPEED_PIXEL) * 2 - 1
        features = [norm_speed]

        lane_onehot = [0, 0, 0]
        lane_onehot[lane_idx] = 1
        features.extend(lane_onehot)

        curr_dist = front[lane_idx]["dist"]
        curr_speed_rel = 0
        if front[lane_idx]["speed"] is not None:
            curr_speed_rel = current_speed - front[lane_idx]["speed"]
        norm_dist = min(1.0, curr_dist / 500.0) if curr_dist != float('inf') else 1.0
        features.append(norm_dist)
        features.append(max(-1.0, min(1.0, curr_speed_rel / 20.0)))
        features.append(min(1.0, front[lane_idx]["ttc"] / 5.0) if front[lane_idx]["ttc"] != float('inf') else 1.0)

        if lane_idx > 0:
            left_dist = front[lane_idx-1]["dist"]
            norm_left = min(1.0, left_dist / 500.0) if left_dist != float('inf') else 1.0
        else:
            norm_left = 1.0
        features.append(norm_left)

        if lane_idx < 2:
            right_dist = front[lane_idx+1]["dist"]
            norm_right = min(1.0, right_dist / 500.0) if right_dist != float('inf') else 1.0
        else:
            norm_right = 1.0
        features.append(norm_right)

        left_pinch = lane_idx > 0 and side_rear[lane_idx-1]
        right_pinch = lane_idx < 2 and side_rear[lane_idx+1]
        features.append(1.0 if left_pinch else 0.0)
        features.append(1.0 if right_pinch else 0.0)

        return np.array(features, dtype=np.float32)

    def _action_to_index(self, action):
        if action.up:
            return 2
        if action.down:
            return 3
        if action.accelerate == 1:
            return 0
        if action.accelerate == -1 or action.braking:
            return 1
        return 4

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

    def get_action(self, player, enemies, current_speed, strategy_mode="balanced"):
        """对外入口：先决策，再统一记录 (state, action)。

        采集必须放在这一层：_decide 有多个提前 return 的分支
        （紧急制动 / 跟车 / 变道 / 超车 / 被夹），采集写在 _decide 末尾
        会把这些状态全部漏掉，训出来的网络就学不会刹车和变道。
        """
        action = self._decide(player, enemies, current_speed, strategy_mode)
        if self.data_collector is not None:
            state = self.extract_features(player, enemies, current_speed)
            self.data_collector.add(state, self._action_to_index(action))
        return action

    def _decide(self, player, enemies, current_speed, strategy_mode="balanced"):
        action = Action()
        strategy = STRATEGIES.get(strategy_mode, STRATEGIES["balanced"])
        speed_abs = abs(current_speed)

        if self.brake_hold > 0:
            self.brake_hold -= 1
        if self.change_hold > 0:
            self.change_hold -= 1
        if self.overtake_timer > 0:
            self.overtake_timer -= 1
            if self.overtake_timer == 0:
                self.overtaking = False

        lane_idx, front, side_rear = self._perceive(player, enemies, current_speed)

        action.dist_curr = front[lane_idx]["dist"] if front[lane_idx]["dist"] != float('inf') else 0
        action.ttc_curr = front[lane_idx]["ttc"] if front[lane_idx]["ttc"] != float('inf') else 9.99
        action.dynamic_limit_override = -(strategy["target_speed_kmh"] / KM_H_PER_PIXEL_SPEED)

        if player.is_changing_lane:
            action.debug_info = "Changing"
            return action

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

        if not self.overtaking and self._should_overtake(lane_idx, front, current_speed, strategy):
            target_lane = -1
            if lane_idx > 0 and self._is_lane_safe_for_overtake(lane_idx-1, front, current_speed):
                target_lane = lane_idx - 1
            elif lane_idx < 2 and self._is_lane_safe_for_overtake(lane_idx+1, front, current_speed):
                target_lane = lane_idx + 1
            if target_lane != -1:
                self._execute_overtake(action, target_lane, current_speed, strategy["target_speed_kmh"])
                return action

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

        front_vehicle_speed = front[lane_idx]["speed"]
        front_crashed = front[lane_idx]["crashed"]
        safe_dist = max(250, abs(current_speed) * 2.5)
        lower_bound = safe_dist * 0.9
        upper_bound = safe_dist * 1.1

        if front_vehicle_speed is not None:
            if front_crashed:
                target_speed = MIN_SPEED_PIXEL
                new_speed = current_speed + (target_speed - current_speed) * 0.3
                action.follow_speed_override = new_speed
                action.braking = True
                action.accelerate = 0
                action.debug_info = "Stop for obstacle"
                return action

            if curr_dist < lower_bound:
                target_speed = front_vehicle_speed - 0.2
                target_speed = max(MIN_SPEED_PIXEL, target_speed)
                new_speed = current_speed + (target_speed - current_speed) * 0.3
                action.follow_speed_override = new_speed
                action.braking = True
                action.accelerate = 0
                action.debug_info = f"Follow (brake) @{abs(target_speed)*KM_H_PER_PIXEL_SPEED:.0f}"
                return action
            elif curr_dist > upper_bound:
                pass
            else:
                action.accelerate = 0
                action.debug_info = "Hold"
                return action

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


# ==================== 神经网络AI ====================
class NeuralAI:
    """神经网络AI"""
    def __init__(self, model_path=None):
        self.model = SimpleNeuralNetwork()
        if model_path and os.path.exists(model_path):
            self.model.load(model_path)
            print(f"已加载模型: {model_path}")
        else:
            print("未找到模型文件，使用随机策略")
        self._feature_extractor = RuleBasedAI()

    def get_action(self, player, enemies, current_speed, strategy_mode="balanced"):
        state = self._feature_extractor.extract_features(player, enemies, current_speed)
        action_idx = self.model.predict(state)
        return self._index_to_action(action_idx, strategy_mode)

    def _index_to_action(self, idx, strategy_mode):
        action = Action()
        strategy = STRATEGIES.get(strategy_mode, STRATEGIES["balanced"])
        action.dynamic_limit_override = -(strategy["target_speed_kmh"] / KM_H_PER_PIXEL_SPEED)
        action.strategy_mode = strategy_mode

        if idx == 0:
            action.accelerate = 1
            action.debug_info = "NN: Accelerate"
        elif idx == 1:
            action.accelerate = -1
            action.braking = True
            action.debug_info = "NN: Brake"
        elif idx == 2:
            action.up = True
            action.debug_info = "NN: Shift Left"
        elif idx == 3:
            action.down = True
            action.debug_info = "NN: Shift Right"
        else:
            action.accelerate = 0
            action.debug_info = "NN: Cruise"
        return action

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.load(path)


# ==================== 训练函数 ====================
def train_model(data_file, model_save_path, epochs=30, batch_size=64, lr=0.001):
    """使用收集的数据训练神经网络"""
    if not os.path.exists(data_file):
        print(f"未找到数据文件: {data_file}")
        return False

    data = np.load(data_file)
    X = data['states']
    y = data['actions']
    num_classes = 5
    y_onehot = np.eye(num_classes)[y]
    model = SimpleNeuralNetwork(input_size=X.shape[1], output_size=num_classes)

    print(f"训练数据: {len(X)} 条, 特征维度: {X.shape[1]}")

    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y_onehot[indices]
        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            probs = model.forward(batch_X)
            loss = -np.sum(batch_y * np.log(probs + 1e-8)) / batch_X.shape[0]

            d_z2 = probs - batch_y
            d_W2 = np.dot(model.a1.T, d_z2) / batch_X.shape[0]
            d_b2 = np.mean(d_z2, axis=0)
            d_a1 = np.dot(d_z2, model.W2.T)
            d_z1 = d_a1 * (model.z1 > 0)
            d_W1 = np.dot(batch_X.T, d_z1) / batch_X.shape[0]
            d_b1 = np.mean(d_z1, axis=0)

            model.W2 -= lr * d_W2
            model.b2 -= lr * d_b2
            model.W1 -= lr * d_W1
            model.b1 -= lr * d_b1

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    model.save(model_save_path)
    print(f"模型已保存到 {model_save_path}")
    return True