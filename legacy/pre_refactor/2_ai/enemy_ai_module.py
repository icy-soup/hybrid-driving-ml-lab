"""
敌车 AI 模块 - 最终稳定版（速度下限、平滑跟车、强制同步）
"""
import random

LANE_CENTERS_BOTTOM = [200, 250, 300]
LANE_CHANGE_SPEED = 6
LANE_CHANGE_DISTANCE = 0
FPS = 60

SAFE_FOLLOW_DIST = 220
TTC_EMERGENCY_THRESHOLD = 1.0
MAX_DECEL = 0.25
REACTION_FRAMES = 5
SPEED_SMOOTHING = 0.3
MIN_ABS_SPEED = -4      # 最低绝对速度（像素/帧），约 24 km/h


class EnemyAI:
    def __init__(self):
        self.warning_timer = 0
        self.decision_to_change = None
        self.last_lane_change_time = -2000
        self.stuck_time = 0
        self.is_changing_lane = False
        self.target_bottom = None
        self.lane_change_direction = 0

    def calc_emergency_brake_dist(self, rel_speed, ego_speed_abs):
        if rel_speed <= 0:
            return 50
        reaction_dist = rel_speed * REACTION_FRAMES
        brake_dist = (rel_speed ** 2) / (2 * MAX_DECEL)
        return reaction_dist + brake_dist + 40

    def calc_required_follow_dist(self, ego_speed, front_speed):
        rel_speed = ego_speed - front_speed
        if rel_speed <= 0:
            return 200
        lane_change_time = LANE_CHANGE_DISTANCE / LANE_CHANGE_SPEED
        required = rel_speed * lane_change_time + 80
        return max(required, 200)

    def clamp_speed(self, desired_rel, current_base_speed):
        """限制相对速度，使绝对速度不低于 MIN_ABS_SPEED"""
        abs_speed = current_base_speed + desired_rel
        if abs_speed < MIN_ABS_SPEED:
            desired_rel = MIN_ABS_SPEED - current_base_speed
        return desired_rel

    def perceive(self, all_vehicles, self_idx, current_base_speed, my_lane, ego_speed):
        dist = [float('inf'), float('inf'), float('inf')]
        ttc = [float('inf'), float('inf'), float('inf')]
        front_speed = [None, None, None]
        rear_dist = [float('inf'), float('inf'), float('inf')]
        rear_speed = [0, 0, 0]

        for veh in all_vehicles:
            if veh == self_idx:
                continue
            v_lane = -1
            if hasattr(veh, '_get_approx_lane'):
                v_lane = veh._get_approx_lane()
            else:
                v_lane = veh._get_lane_idx()
            if v_lane == -1:
                continue

            d_x = veh.left - self_idx.left
            if hasattr(veh, 'crashed') and veh.crashed:
                v_speed = current_base_speed
            else:
                v_speed = current_base_speed + (veh.relative_speed if hasattr(veh, 'relative_speed') else 0)

            if d_x > 0:
                if d_x < dist[v_lane]:
                    dist[v_lane] = d_x
                    front_speed[v_lane] = v_speed
                    closing = ego_speed - v_speed
                    ttc[v_lane] = d_x / closing if closing > 1.0 else float('inf')
            elif d_x < 0:
                d_rear = abs(d_x)
                if d_rear < rear_dist[v_lane]:
                    rear_dist[v_lane] = d_rear
                    rear_speed[v_lane] = v_speed

        return dist, ttc, front_speed, rear_dist, rear_speed

    def decide(self, all_vehicles, self_idx, current_base_speed, current_time):
        # 默认决策（防止任何路径漏返回）
        default_decision = {
            "need_change": False, "change_dir": 0,
            "warning_timer": self.warning_timer,
            "target_rel_speed": self.clamp_speed(self_idx.base_relative_speed, current_base_speed),
            "is_emergency": False,
            "debug_info": "Default"
        }

        if self.warning_timer > 0:
            self.warning_timer -= 1

        player = None
        for veh in all_vehicles:
            if hasattr(veh, '_get_approx_lane'):
                player = veh
                break

        ego_speed = current_base_speed + self_idx.relative_speed
        my_lane = self_idx._get_lane_idx()
        if my_lane == -1:
            return default_decision

        dist, ttc, front_speed, rear_dist, rear_speed = self.perceive(
            all_vehicles, self_idx, current_base_speed, my_lane, ego_speed)

        # ========== 强制速度同步（距离很近时） ==========
        if dist[my_lane] != float('inf') and dist[my_lane] < 80:
            desired_rel = front_speed[my_lane] - current_base_speed
            desired_rel = self.clamp_speed(desired_rel, current_base_speed)
            return self._normal_decision(desired_rel, "Force sync")

        # ========== 紧急制动（物理极限或 TTC 过低） ==========
        if front_speed[my_lane] is not None:
            rel_speed = ego_speed - front_speed[my_lane]
            if rel_speed > 0:
                emergency_dist = self.calc_emergency_brake_dist(rel_speed, abs(ego_speed))
                if dist[my_lane] < emergency_dist:
                    return self._emergency_decision(current_base_speed)
        if ttc[my_lane] != float('inf') and ttc[my_lane] < TTC_EMERGENCY_THRESHOLD:
            return self._emergency_decision(current_base_speed)

        # ========== 变道决策 ==========
        need_change = False
        if front_speed[my_lane] is not None:
            dynamic_dist = self.calc_required_follow_dist(ego_speed, front_speed[my_lane])
            hardcoded_dist = 350
            min_safe_dist = min(dynamic_dist, hardcoded_dist)
            if (dist[my_lane] < min_safe_dist and (ego_speed - front_speed[my_lane]) > 1.0) or \
               (ttc[my_lane] != float('inf') and ttc[my_lane] < 2.0):
                need_change = True
        if self.stuck_time > 8 * FPS:
            need_change = True

        if current_time - self.last_lane_change_time < 2000:
            need_change = False

        if need_change:
            left_safe, right_safe = self._check_side_safe(my_lane, dist, rear_dist, rear_speed, ego_speed)
            if left_safe or right_safe:
                change_dir = self._choose_side(my_lane, left_safe, right_safe, dist)
                self.warning_timer = 60
                self.last_lane_change_time = current_time + 2000
                return self._change_decision(change_dir, self_idx.base_relative_speed, current_base_speed)

        # ========== 跟车/减速（平滑跟随） ==========
        desired_rel = self_idx.base_relative_speed
        if front_speed[my_lane] is not None:
            # 动态安全距离（与速度成正比）
            safe_dist = max(250, abs(ego_speed) * 2.5)
            # 如果距离小于安全距离或 TTC 危险，则减速
            if dist[my_lane] < safe_dist or (ttc[my_lane] != float('inf') and ttc[my_lane] < 2.5):
                # 目标相对速度 = 前车速度 - 基础速度 - 0.2（留出安全余量）
                base_desired = front_speed[my_lane] - current_base_speed - 0.2
                desired_rel = max(-1.0, base_desired)
                debug = "Follow/Decelerate"
            else:
                desired_rel = self_idx.base_relative_speed
                debug = "Cruise"
        else:
            # 前方无车，检查后方密度
            rear_count = 0
            for i in range(3):
                if rear_dist[i] != float('inf') and rear_dist[i] < 300:
                    rear_count += 1
            if rear_count >= 2:
                desired_rel = min(self_idx.base_relative_speed * 1.1, 18)
                debug = "Rear dense - accelerate"
            else:
                if self_idx.relative_speed < self_idx.base_relative_speed * 1.05:
                    desired_rel = min(self_idx.relative_speed + 0.03, self_idx.base_relative_speed * 1.05)
                    debug = "No front - accelerate"
                else:
                    desired_rel = self_idx.base_relative_speed
                    debug = "No front - cruise"

        # 指数平滑，避免震荡
        smoothed_rel = self_idx.relative_speed + (desired_rel - self_idx.relative_speed) * SPEED_SMOOTHING
        # 速度下限限制
        smoothed_rel = self.clamp_speed(smoothed_rel, current_base_speed)
        return self._normal_decision(smoothed_rel, debug)

    def _check_side_safe(self, my_lane, dist, rear_dist, rear_speed, ego_speed):
        left_safe = right_safe = False
        if my_lane > 0:
            left_safe = (dist[my_lane-1] == float('inf') or dist[my_lane-1] > 350)
            if rear_dist[my_lane-1] != float('inf') and rear_dist[my_lane-1] < 80:
                left_safe = False
            rear_approach = rear_speed[my_lane-1] - ego_speed
            if rear_approach > 6.0:
                left_safe = False
        if my_lane < 2:
            right_safe = (dist[my_lane+1] == float('inf') or dist[my_lane+1] > 350)
            if rear_dist[my_lane+1] != float('inf') and rear_dist[my_lane+1] < 80:
                right_safe = False
            rear_approach = rear_speed[my_lane+1] - ego_speed
            if rear_approach > 6.0:
                right_safe = False
        return left_safe, right_safe

    def _choose_side(self, my_lane, left_safe, right_safe, dist):
        if left_safe and right_safe:
            left_gap = dist[my_lane-1] if dist[my_lane-1] != float('inf') else 0
            right_gap = dist[my_lane+1] if dist[my_lane+1] != float('inf') else 0
            return -1 if left_gap > right_gap else 1
        return -1 if left_safe else 1

    def _emergency_decision(self, current_base_speed):
        target_rel = -1.0
        target_rel = self.clamp_speed(target_rel, current_base_speed)
        return {
            "need_change": False, "change_dir": 0,
            "warning_timer": self.warning_timer,
            "target_rel_speed": target_rel,
            "is_emergency": True,
            "debug_info": "Emergency Brake"
        }

    def _normal_decision(self, target_rel, debug):
        return {
            "need_change": False, "change_dir": 0,
            "warning_timer": self.warning_timer,
            "target_rel_speed": target_rel,
            "is_emergency": False,
            "debug_info": debug
        }

    def _change_decision(self, change_dir, target_rel, current_base_speed):
        target_rel = self.clamp_speed(target_rel, current_base_speed)
        return {
            "need_change": True, "change_dir": change_dir,
            "warning_timer": self.warning_timer,
            "target_rel_speed": target_rel,
            "is_emergency": False,
            "debug_info": f"Change lane {'L' if change_dir == -1 else 'R'}"
        }