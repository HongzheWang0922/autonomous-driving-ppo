"""
完整版交叉路口环境 - 修复Length=1的bug
修复内容：
1. 修复has_arrived过早触发
2. 防止初始位置碰撞
3. 确保_randomize_task在vehicle创建之前调用
"""

from typing import Dict, Text
import numpy as np
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

try:
    from highway_env.road.lane import LineType, StraightLane, CircularLane
except ImportError:
    try:
        from highway_env.road.graphics import LineType
    except ImportError:
        class LineType:
            NONE = 0
            STRIPED = 1
            CONTINUOUS = 2
    try:
        StraightLane = utils.StraightLane
        CircularLane = utils.CircularLane
    except AttributeError:
        raise ImportError("无法找到StraightLane和CircularLane")

import random


class IntersectionEnv(AbstractEnv):
    """完整版交叉路口环境 - 修复版"""

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        print("🔥 完整版配置（修复版）: 密集奖励 + 安全意识 + Bug修复 🔥")
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 20,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-20, 20],
                        "vy": [-20, 20],
                    },
                    "absolute": False,
                    "flatten": False,
                    "observe_intentions": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True,
                },
                "duration": 25,
                "destination": "o1",
                "controlled_vehicles": 1,
                "initial_vehicle_count": 10,
                "spawn_probability": 0.6,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.6],
                "scaling": 5.5 * 1.3,
                "collision_reward": -10.0,
                "arrived_reward": 20.0,
                "distance_reward": 1.0,
                "progress_reward": 5.0,
                "heading_reward": 0.2,
                "lane_reward": 0.3,
                "safety_distance_reward": 1.0,
                "collision_risk_reward": 2.0,
                "smooth_driving_reward": 0.3,
                "dynamic_speed_reward": 0.4,
                "safe_distance": 15.0,
                "warning_distance": 8.0,
                "danger_distance": 3.0,
                "ttc_threshold": 3.0,
                "reward_speed_range": [6.0, 9.0],
                "normalize_reward": False,
                "offroad_terminal": False,
            }
        )
        return config

    def _reward(self, action: Action) -> float:
        """完整的奖励函数"""
        rewards = self._rewards(action)
        
        if self.vehicle.crashed:
            return self.config["collision_reward"]
        
        if rewards["arrived_reward"]:
            return self.config["arrived_reward"]
        
        reward = (
            self.config.get("distance_reward", 0) * rewards["distance_reward"] +
            self.config.get("progress_reward", 0) * rewards["progress_reward"] +
            self.config.get("heading_reward", 0) * rewards["heading_reward"] +
            self.config.get("lane_reward", 0) * rewards["lane_reward"] +
            self.config.get("safety_distance_reward", 0) * rewards["safety_distance_reward"] +
            self.config.get("collision_risk_reward", 0) * rewards["collision_risk_reward"] +
            self.config.get("smooth_driving_reward", 0) * rewards["smooth_driving_reward"] +
            self.config.get("dynamic_speed_reward", 0) * rewards["dynamic_speed_reward"]
        )
        
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """计算各项奖励"""
        rewards = {
            "collision_reward": float(self.vehicle.crashed),
            "arrived_reward": float(self.has_arrived),
            "on_road_reward": float(self.vehicle.on_road),
        }
        
        rewards["distance_reward"] = self._compute_distance_reward()
        rewards["progress_reward"] = self._compute_progress_reward()
        rewards["heading_reward"] = self._compute_heading_reward()
        rewards["lane_reward"] = self._compute_lane_reward()
        rewards["safety_distance_reward"] = self._compute_safety_distance_reward()
        rewards["collision_risk_reward"] = self._compute_collision_risk_reward()
        rewards["smooth_driving_reward"] = self._compute_smooth_driving_reward()
        rewards["dynamic_speed_reward"] = self._compute_dynamic_speed_reward()
        
        return rewards

    def _compute_distance_reward(self) -> float:
        current_distance = self._get_distance_to_goal()
        if not hasattr(self, '_last_distance'):
            self._last_distance = current_distance
            return 0.0
        distance_change = self._last_distance - current_distance
        self._last_distance = current_distance
        return np.clip(distance_change * 0.1, -1.0, 1.0)

    def _compute_progress_reward(self) -> float:
        current_progress = self._get_route_progress()
        if not hasattr(self, '_last_progress'):
            self._last_progress = current_progress
            return 0.0
        progress_delta = current_progress - self._last_progress
        self._last_progress = current_progress
        return np.clip(progress_delta * 10.0, 0.0, 1.0)

    def _compute_heading_reward(self) -> float:
        goal_pos = self._get_goal_position()
        if goal_pos is None:
            return 0.0
        to_goal = goal_pos - self.vehicle.position
        to_goal_angle = np.arctan2(to_goal[1], to_goal[0])
        heading_diff = np.abs(self.vehicle.heading - to_goal_angle)
        heading_diff = min(heading_diff, 2*np.pi - heading_diff)
        return np.cos(heading_diff)

    def _compute_lane_reward(self) -> float:
        if hasattr(self.vehicle, 'route') and self.vehicle.route:
            current_lane_index = self.vehicle.lane_index
            for route_lane in self.vehicle.route:
                if current_lane_index[:2] == route_lane[:2]:
                    return 1.0
            return -0.5
        return 0.0

    def _compute_safety_distance_reward(self) -> float:
        min_distance = self._get_min_distance_to_vehicles()
        if min_distance is None or min_distance > 50:
            return 0.5
        safe_dist = self.config["safe_distance"]
        warning_dist = self.config["warning_distance"]
        danger_dist = self.config["danger_distance"]
        if min_distance > safe_dist:
            return 0.5
        elif min_distance > warning_dist:
            return 0.0
        elif min_distance > danger_dist:
            ratio = (min_distance - danger_dist) / (warning_dist - danger_dist)
            return -0.5 * (1 - ratio)
        else:
            return -2.0

    def _compute_collision_risk_reward(self) -> float:
        ttc = self._compute_time_to_collision()
        if ttc is None or ttc > self.config["ttc_threshold"]:
            return 0.0
        if ttc < 1.0:
            return -2.0
        elif ttc < 2.0:
            return -1.0
        elif ttc < 3.0:
            return -0.3
        else:
            return 0.0

    def _compute_smooth_driving_reward(self) -> float:
        current_speed = self.vehicle.speed
        if not hasattr(self, '_last_speed'):
            self._last_speed = current_speed
            return 0.0
        acceleration = abs(current_speed - self._last_speed) / 0.1
        self._last_speed = current_speed
        if acceleration < 3.0:
            return 0.5
        elif acceleration < 5.0:
            return 0.0
        else:
            return -0.3

    def _compute_dynamic_speed_reward(self) -> float:
        density = self._get_vehicle_density()
        if density > 0.5:
            optimal_speed = 6.0
        elif density > 0.3:
            optimal_speed = 7.5
        else:
            optimal_speed = 9.0
        speed_diff = abs(self.vehicle.speed - optimal_speed)
        if speed_diff < 1.0:
            return 1.0
        elif speed_diff < 2.0:
            return 0.5
        else:
            return 0.0

    def _get_min_distance_to_vehicles(self) -> float:
        if len(self.road.vehicles) <= 1:
            return None
        min_distance = float('inf')
        ego_pos = self.vehicle.position
        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            distance = np.linalg.norm(vehicle.position - ego_pos)
            min_distance = min(min_distance, distance)
        return min_distance if min_distance < float('inf') else None

    def _compute_time_to_collision(self) -> float:
        if len(self.road.vehicles) <= 1:
            return None
        ego_pos = self.vehicle.position
        ego_vel = self.vehicle.velocity
        min_ttc = float('inf')
        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            rel_pos = vehicle.position - ego_pos
            rel_vel = vehicle.velocity - ego_vel
            if np.dot(rel_pos, ego_vel) < 0:
                continue
            rel_speed = np.dot(rel_vel, rel_pos) / (np.linalg.norm(rel_pos) + 1e-6)
            if rel_speed > 0.1:
                distance = np.linalg.norm(rel_pos)
                ttc = distance / rel_speed
                min_ttc = min(min_ttc, ttc)
        return min_ttc if min_ttc < float('inf') else None

    def _get_vehicle_density(self) -> float:
        if len(self.road.vehicles) <= 1:
            return 0.0
        ego_pos = self.vehicle.position
        nearby_count = 0
        search_radius = 20.0
        for vehicle in self.road.vehicles:
            if vehicle is self.vehicle:
                continue
            distance = np.linalg.norm(vehicle.position - ego_pos)
            if distance < search_radius:
                nearby_count += 1
        return min(nearby_count / 10.0, 1.0)

    def _get_distance_to_goal(self) -> float:
        goal_pos = self._get_goal_position()
        if goal_pos is None:
            return 0.0
        return np.linalg.norm(self.vehicle.position - goal_pos)

    def _get_goal_position(self) -> np.ndarray:
        try:
            destination = self.config.get("destination", "o1")
            for i in range(4):
                lane_index = (f"il{i}", destination, 0)
                try:
                    lane = self.road.network.get_lane(lane_index)
                    return lane.position(lane.length, 0)
                except:
                    continue
            return None
        except:
            return None

    def _get_route_progress(self) -> float:
        if not hasattr(self.vehicle, 'lane') or self.vehicle.lane is None:
            return 0.0
        try:
            longitudinal, _ = self.vehicle.lane.local_coordinates(self.vehicle.position)
            progress = longitudinal / self.vehicle.lane.length if self.vehicle.lane.length > 0 else 0
            if hasattr(self.vehicle, 'route') and self.vehicle.route:
                current_lane = self.vehicle.lane_index
                for idx, route_lane in enumerate(self.vehicle.route):
                    if current_lane[:2] == route_lane[:2]:
                        total_progress = (idx + progress) / len(self.vehicle.route)
                        return np.clip(total_progress, 0.0, 1.0)
            return np.clip(progress, 0.0, 1.0)
        except:
            return 0.0

    def _reset(self) -> None:
        # 🔧 修复：先选择任务，再创建车辆
        self._select_task()  # 先选择起点/终点
        
        # 重置追踪变量
        for attr in ['_last_distance', '_last_progress', '_last_speed']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self._make_road()
        self._make_vehicles()  # 根据任务创建车辆
        
        # 🔧 修复：确保初始位置没有碰撞
        self._ensure_no_initial_collision()

    def _select_task(self) -> None:
        """🔧 新增：在创建车辆之前选择任务"""
        scenarios = [
            ("S", "W", "left"), ("S", "N", "straight"), ("S", "E", "right"),
            ("E", "N", "left"), ("E", "W", "straight"), ("E", "S", "right"),
            ("N", "E", "left"), ("N", "S", "straight"), ("N", "W", "right"),
            ("W", "S", "left"), ("W", "E", "straight"), ("W", "N", "right"),
        ]
        start_dir, target_dir, task_type = random.choice(scenarios)
        self.start_direction = start_dir
        self.target_direction = target_dir
        self.task_type = task_type
        
        # 设置destination
        destination_map = {"S": "o3", "N": "o1", "E": "o2", "W": "o0"}
        if target_dir in destination_map:
            self.config["destination"] = destination_map[target_dir]

    def _ensure_no_initial_collision(self) -> None:
        """🔧 新增：确保初始位置没有碰撞"""
        max_attempts = 5
        for attempt in range(max_attempts):
            # 检查是否碰撞
            if not self.vehicle.crashed:
                return
            
            # 如果碰撞，稍微调整位置
            if hasattr(self.vehicle, 'lane') and self.vehicle.lane is not None:
                try:
                    # 往后移5米
                    new_long = max(5, self.vehicle.lane.local_coordinates(self.vehicle.position)[0] + 5)
                    self.vehicle.position = self.vehicle.lane.position(new_long, 0)
                    self.vehicle.heading = self.vehicle.lane.heading_at(new_long)
                except:
                    pass

    def _is_terminated(self) -> bool:
        return (
            self.vehicle.crashed
            or self.has_arrived
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _info(self, obs, action) -> dict:
        info = super()._info(obs, action)
        info["rewards"] = self._rewards(action)
        info["start_direction"] = getattr(self, "start_direction", "S")
        info["target_direction"] = getattr(self, "target_direction", "W")
        info["task_type"] = getattr(self, "task_type", "left")
        info["crashed"] = self.vehicle.crashed
        info["arrived"] = self.has_arrived
        info["distance_to_goal"] = self._get_distance_to_goal()
        info["route_progress"] = self._get_route_progress()
        info["min_distance_to_vehicles"] = self._get_min_distance_to_vehicles()
        info["ttc"] = self._compute_time_to_collision()
        info["vehicle_density"] = self._get_vehicle_density()
        return info

    def _make_road(self) -> None:
        lane_width = 4.0
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50

        net = RoadNetwork()
        
        try:
            n = LineType.NONE
            c = LineType.CONTINUOUS
            s = LineType.STRIPED
        except:
            n, c, s = 0, 2, 1

        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner), "ir" + str(corner),
                StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10),
            )
            
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner), "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center, right_turn_radius,
                    angle + np.radians(180), angle + np.radians(270),
                    line_types=[n, c], priority=priority, speed_limit=10,
                ),
            )
            
            l_center = rotation @ (np.array([
                -left_turn_radius + lane_width / 2,
                left_turn_radius - lane_width / 2,
            ]))
            net.add_lane(
                "ir" + str(corner), "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center, left_turn_radius,
                    angle + np.radians(0), angle + np.radians(-90),
                    clockwise=False, line_types=[n, n],
                    priority=priority - 1, speed_limit=10,
                ),
            )
            
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner), "il" + str((corner + 2) % 4),
                StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10),
            )
            
            start = rotation @ np.array([lane_width / 2, -outer_distance])
            end = rotation @ np.array([lane_width / 2, -access_length - outer_distance])
            net.add_lane(
                "il" + str(corner), "o" + str(corner),
                StraightLane(start, end, line_types=[n, c], priority=priority, speed_limit=10),
            )

        road = Road(network=net, np_random=self.np_random,
                    record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """🔧 修复：根据预选的任务创建车辆"""
        # 根据起点方向确定初始lane
        dir_to_corner = {"W": 0, "N": 1, "E": 2, "S": 3}
        corner = dir_to_corner.get(self.start_direction, 3)
        
        ego_lane_index = (f"o{corner}", f"ir{corner}", 0)
        ego_lane = self.road.network.get_lane(ego_lane_index)
        destination = self.config.get("destination", "o1")
        
        ego_vehicle = self.action_type.vehicle_class(
            self.road, 
            ego_lane.position(10, 0),  # 🔧 从lane开始10米处开始，避免太前面
            speed=0,  # 🔧 初始速度为0
            heading=ego_lane.heading_at(10),
        )
        
        try:
            ego_vehicle.plan_route_to(destination)
            ego_vehicle.speed_index = ego_vehicle.speed_to_index(0)
            ego_vehicle.target_speed = 0
        except AttributeError:
            pass

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        for i in range(n_vehicles):
            self._spawn_vehicle(np_random=self.np_random)

    def _spawn_vehicle(self, longitudinal: float = 0, position_deviation: float = 1.0,
                       speed_deviation: float = 1.0, spawn_probability: float = 0.6,
                       go_straight: bool = False, np_random = None) -> None:
        if np_random is None:
            np_random = self.np_random

        if np_random.uniform() > spawn_probability:
            return

        route = np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        vehicle = vehicle_type.make_on_lane(
            self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
            longitudinal=(longitudinal + 5 + np_random.normal() * position_deviation),
            speed=8 + np_random.normal() * speed_deviation,
        )
        
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    @property
    def has_arrived(self) -> bool:
        """
        🔧 修复：更严格的到达判断
        必须：
        1. 在出口lane（"il"开头）
        2. 朝向正确的出口（"o"）
        3. 接近lane末端
        4. 🆕 已经运行至少5步（防止初始就判定到达）
        """
        # 🆕 必须至少运行5步
        if self.time < 0.5:  # 0.1秒/步，5步=0.5秒
            return False
        
        return (
            "il" in self.vehicle.lane_index[0]
            and "o" in self.vehicle.lane_index[1]
            and self.vehicle.lane.local_coordinates(self.vehicle.position)[0] >= self.vehicle.lane.length - 10
        )


def IntersectionEnvWrapper(difficulty="easy", **kwargs):
    """创建环境的便捷函数 - 修复版"""
    print(f"🎯 创建完整版环境（修复版） (难度: {difficulty})")
    
    env = IntersectionEnv()
    
    difficulty_configs = {
        "easy": {"initial_vehicle_count": 5, "duration": 25, "spawn_probability": 0.4},
        "medium": {"initial_vehicle_count": 8, "duration": 25, "spawn_probability": 0.6},
        "hard": {"initial_vehicle_count": 12, "duration": 25, "spawn_probability": 0.8},
    }

    if difficulty in difficulty_configs:
        env.configure(difficulty_configs[difficulty])

    return env
