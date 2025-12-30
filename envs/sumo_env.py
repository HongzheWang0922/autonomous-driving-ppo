"""
基于SUMO的自动驾驶环境 - 支持四阶段课程学习
使用真实美国街道地图，支持从简单到复杂的渐进式学习

Stage 1: 空路导航 - 学习不偏离车道、到达终点
Stage 2: 红绿灯遵守 - 学习遵守交通信号
Stage 3: 动态避障 - 学习与其他车辆交互
Stage 4: 综合场景 - 行人 + 长距离导航
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import random
from typing import Dict, Tuple, Optional, List

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("请设置环境变量 'SUMO_HOME'")

import traci
import sumolib


class SUMODrivingEnv(gym.Env):
    """
    基于SUMO的自动驾驶环境
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(
        self,
        stage: int = 1,
        net_file: str = None,
        route_file: str = None,
        use_gui: bool = False,
        max_episode_steps: int = 500,
        step_length: float = 0.1,
        num_background_vehicles: int = 0,
        num_pedestrians: int = 0,
        min_route_length: float = 200.0,  # Stage 1-3最小路由长度(米)
        max_route_length: float = 500.0,  # Stage 4增加到更长
        seed: Optional[int] = None,
    ):
        """
        Args:
            stage: 训练阶段 (1-4)
            net_file: SUMO网络文件路径
            route_file: SUMO路由文件路径（可选，会动态生成）
            use_gui: 是否使用GUI
            max_episode_steps: 最大步数
            step_length: 仿真步长(秒)
            num_background_vehicles: 背景车辆数量
            num_pedestrians: 行人数量
            min_route_length: 最小路由长度
            max_route_length: 最大路由长度
            seed: 随机种子
        """
        super().__init__()
        
        self.stage = stage
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.max_episode_steps = max_episode_steps
        self.step_length = step_length
        self.min_route_length = min_route_length
        self.max_route_length = max_route_length
        
        # 根据stage设置环境参数
        self.num_background_vehicles = self._get_stage_vehicles(stage, num_background_vehicles)
        self.num_pedestrians = self._get_stage_pedestrians(stage, num_pedestrians)
        
        # SUMO相关
        self.sumo_cmd = None
        self.sumo_running = False
        self.net = None
        self.ego_id = "ego"
        
        # Episode相关
        self.current_step = 0
        self.episode_count = 0
        self.start_edge = None
        self.goal_edge = None
        self.route_edges = []
        self.route_length = 0.0
        
        # 奖励追踪
        self.total_reward = 0.0
        self.last_distance_to_goal = 0.0
        self.last_speed = 0.0
        self.collision_occurred = False
        self.goal_reached = False
        
        # 统计信息
        self.stats = {
            "red_light_violations": 0,
            "collisions": 0,
            "off_route_count": 0,
            "total_distance": 0.0,
        }
        
        # 加载SUMO网络
        if net_file:
            self._load_network()
        
        # 定义观测空间和动作空间
        self._define_spaces()
        
        # 设置随机种子
        if seed is not None:
            self.seed(seed)
    
    def _get_stage_vehicles(self, stage: int, override: int) -> int:
        """根据阶段返回背景车辆数量"""
        if override > 0:
            return override
        stage_vehicles = {1: 0, 2: 0, 3: 15, 4: 20}
        return stage_vehicles.get(stage, 0)
    
    def _get_stage_pedestrians(self, stage: int, override: int) -> int:
        """根据阶段返回行人数量"""
        if override > 0:
            return override
        stage_pedestrians = {1: 0, 2: 0, 3: 0, 4: 10}
        return stage_pedestrians.get(stage, 0)
    
    def _load_network(self):
        """加载SUMO网络"""
        try:
            self.net = sumolib.net.readNet(self.net_file)
            print(f"✅ 加载SUMO网络: {self.net_file}")
            print(f"   - 路段数: {len(list(self.net.getEdges()))}")
            print(f"   - 交叉口数: {len(list(self.net.getNodes()))}")
        except Exception as e:
            print(f"❌ 加载SUMO网络失败: {e}")
            raise
    
    def _define_spaces(self):
        """定义观测空间和动作空间"""
        # 观测空间：
        # - ego车辆状态: [speed, acceleration, position_x, position_y, heading, distance_to_goal]
        # - 周围车辆: 最近8辆车的相对位置和速度 [rel_x, rel_y, rel_speed] * 8
        # - 红绿灯状态: 前方最近红绿灯状态 [distance, is_red, is_yellow, is_green]
        # - 路由信息: [progress_ratio, angle_to_goal]
        
        obs_dim = 6 + 8*3 + 4 + 2  # 6 + 24 + 4 + 2 = 36
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 动作空间：连续动作 [加速度, 转向角]
        # 加速度: [-4.5, 2.6] m/s^2
        # 转向角: [-30, 30] 度
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
    
    def seed(self, seed=None):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def _start_sumo(self):
        """启动SUMO仿真"""
        if self.sumo_running:
            self._close_sumo()
        
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        self.sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",  # 禁用瞬移
            "--collision.action", "warn",  # 碰撞时警告
            "--start", "true" if self.use_gui else "false",
        ]
        
        # 如果有路由文件，添加
        if self.route_file and os.path.exists(self.route_file):
            self.sumo_cmd.extend(["-r", self.route_file])
        
        try:
            traci.start(self.sumo_cmd)
            self.sumo_running = True
        except Exception as e:
            print(f"❌ 启动SUMO失败: {e}")
            raise
    
    def _close_sumo(self):
        """关闭SUMO仿真"""
        if self.sumo_running:
            try:
                traci.close()
            except:
                pass
            self.sumo_running = False
    
    def _select_random_route(self) -> Tuple[str, str, List[str], float]:
        """
        随机选择起点和终点，计算路由
        
        Returns:
            (start_edge_id, goal_edge_id, route_edges, route_length)
        """
        if not self.net:
            raise ValueError("网络未加载")
        
        # 获取所有可行驶的边（排除内部边）
        all_edges = [e for e in self.net.getEdges() 
                     if not e.isSpecial() and e.allows("passenger")]
        
        if len(all_edges) < 2:
            raise ValueError("可用路段太少")
        
        max_attempts = 50
        for attempt in range(max_attempts):
            start_edge = random.choice(all_edges)
            goal_edge = random.choice(all_edges)
            
            if start_edge == goal_edge:
                continue
            
            # 计算路由
            try:
                route_edges = self.net.getShortestPath(start_edge, goal_edge)
                if route_edges[0] is None or len(route_edges[0]) < 2:
                    continue
                
                route = route_edges[0]
                route_length = sum([e.getLength() for e in route])
                
                # 根据stage检查路由长度
                min_len = self.min_route_length
                max_len = self.max_route_length
                
                if self.stage == 4:
                    # Stage 4: 长距离路由
                    min_len = 500.0
                    max_len = 1500.0
                
                if min_len <= route_length <= max_len:
                    edge_ids = [e.getID() for e in route]
                    return start_edge.getID(), goal_edge.getID(), edge_ids, route_length
            
            except Exception as e:
                continue
        
        # 如果找不到合适路由，使用任意两个边
        start_edge = all_edges[0]
        goal_edge = all_edges[-1]
        return start_edge.getID(), goal_edge.getID(), [start_edge.getID(), goal_edge.getID()], 100.0
    
    def _spawn_ego_vehicle(self):
        """生成ego车辆"""
        try:
            # 移除旧的ego车辆（如果存在）
            if self.ego_id in traci.vehicle.getIDList():
                traci.vehicle.remove(self.ego_id)
            
            # 添加车辆类型
            if self.ego_id not in traci.vehicletype.getIDList():
                traci.vehicletype.add(
                    self.ego_id,
                    accel=2.6,
                    decel=4.5,
                    sigma=0.0,
                    length=5.0,
                    maxSpeed=15.0,
                    vClass="passenger",
                    color=(0, 255, 0, 255)  # 绿色
                )
            
            # 添加ego车辆
            traci.vehicle.add(
                vehID=self.ego_id,
                routeID="",
                typeID=self.ego_id,
                depart="now",
                departLane="best",
                departSpeed="0"
            )
            
            # 设置路由
            traci.vehicle.setRoute(self.ego_id, self.route_edges)
            
            # 设置为手动控制
            traci.vehicle.setSpeedMode(self.ego_id, 0)  # 完全手动控制速度
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)  # 完全手动控制变道
            
        except Exception as e:
            print(f"❌ 生成ego车辆失败: {e}")
            raise
    
    def _spawn_background_vehicles(self):
        """生成背景车辆"""
        if self.num_background_vehicles == 0:
            return
        
        # 获取所有可用边
        all_edges = [e.getID() for e in self.net.getEdges() 
                     if not self.net.getEdge(e.getID()).isSpecial()]
        
        if not all_edges:
            return
        
        # 添加背景车辆类型
        if "background" not in traci.vehicletype.getIDList():
            traci.vehicletype.add(
                "background",
                accel=2.6,
                decel=4.5,
                sigma=0.5,
                length=5.0,
                maxSpeed=13.89,  # ~50 km/h
                vClass="passenger",
                color=(255, 255, 0, 255)  # 黄色
            )
        
        # 生成背景车辆
        spawned = 0
        max_attempts = self.num_background_vehicles * 3
        
        for i in range(max_attempts):
            if spawned >= self.num_background_vehicles:
                break
            
            try:
                veh_id = f"bg_{i}"
                edge = random.choice(all_edges)
                
                # 随机路由
                goal_edge = random.choice(all_edges)
                try:
                    route = self.net.getShortestPath(
                        self.net.getEdge(edge), 
                        self.net.getEdge(goal_edge)
                    )[0]
                    if route:
                        route_ids = [e.getID() for e in route]
                    else:
                        route_ids = [edge]
                except:
                    route_ids = [edge]
                
                traci.vehicle.add(
                    vehID=veh_id,
                    routeID="",
                    typeID="background",
                    depart="now",
                    departLane="random",
                    departSpeed="random"
                )
                traci.vehicle.setRoute(veh_id, route_ids)
                spawned += 1
            
            except Exception as e:
                continue
    
    def _spawn_pedestrians(self):
        """生成行人（Stage 4）"""
        if self.num_pedestrians == 0:
            return
        
        # TODO: 在SUMO中生成行人需要人行道网络
        # 这里先预留接口，后续可以完善
        pass
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            self.seed(seed)
        
        # 启动SUMO
        if not self.sumo_running:
            self._start_sumo()
        else:
            # 清空所有车辆
            for veh_id in traci.vehicle.getIDList():
                traci.vehicle.remove(veh_id)
        
        # 选择随机路由
        self.start_edge, self.goal_edge, self.route_edges, self.route_length = \
            self._select_random_route()
        
        # 重置状态
        self.current_step = 0
        self.total_reward = 0.0
        self.collision_occurred = False
        self.goal_reached = False
        self.stats = {
            "red_light_violations": 0,
            "collisions": 0,
            "off_route_count": 0,
            "total_distance": 0.0,
        }
        
        # 生成车辆
        self._spawn_ego_vehicle()
        self._spawn_background_vehicles()
        self._spawn_pedestrians()
        
        # 执行一步仿真
        traci.simulationStep()
        
        # 获取初始观测
        obs = self._get_observation()
        self.last_distance_to_goal = self._get_distance_to_goal()
        
        self.episode_count += 1
        
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        """执行一步"""
        if not self.sumo_running:
            raise RuntimeError("SUMO未启动，请先调用reset()")
        
        # 解析动作
        accel = action[0] * 4.5  # [-4.5, 4.5] m/s^2
        steer = action[1] * 30.0  # [-30, 30] 度
        
        # 应用动作到ego车辆
        try:
            if self.ego_id in traci.vehicle.getIDList():
                current_speed = traci.vehicle.getSpeed(self.ego_id)
                new_speed = max(0, current_speed + accel * self.step_length)
                traci.vehicle.setSpeed(self.ego_id, new_speed)
                
                # 转向（简化处理：调整车道偏移）
                # 注意：SUMO的转向控制比较复杂，这里简化处理
                # 实际应用中可能需要更精细的控制
        except:
            pass
        
        # 执行仿真步
        traci.simulationStep()
        self.current_step += 1
        
        # 获取观测
        obs = self._get_observation()
        
        # 计算奖励
        reward = self._compute_reward()
        self.total_reward += reward
        
        # 检查终止条件
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        # 获取info
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return obs
            
            # Ego车辆状态
            speed = traci.vehicle.getSpeed(self.ego_id)
            accel = traci.vehicle.getAcceleration(self.ego_id)
            pos = traci.vehicle.getPosition(self.ego_id)
            heading = traci.vehicle.getAngle(self.ego_id)
            distance_to_goal = self._get_distance_to_goal()
            
            obs[0] = speed / 15.0  # 归一化
            obs[1] = accel / 4.5
            obs[2] = pos[0] / 1000.0  # 归一化位置
            obs[3] = pos[1] / 1000.0
            obs[4] = np.cos(np.radians(heading))
            obs[5] = np.sin(np.radians(heading))
            
            # 周围车辆（最近8辆）
            nearby_vehicles = self._get_nearby_vehicles(max_count=8)
            ego_pos = np.array(pos)
            for i, veh_id in enumerate(nearby_vehicles):
                if i >= 8:
                    break
                try:
                    veh_pos = np.array(traci.vehicle.getPosition(veh_id))
                    veh_speed = traci.vehicle.getSpeed(veh_id)
                    rel_pos = veh_pos - ego_pos
                    obs[6 + i*3] = rel_pos[0] / 50.0  # 归一化
                    obs[6 + i*3 + 1] = rel_pos[1] / 50.0
                    obs[6 + i*3 + 2] = (veh_speed - speed) / 15.0
                except:
                    pass
            
            # 红绿灯状态
            tls_state = self._get_traffic_light_state()
            obs[30:34] = tls_state
            
            # 路由信息
            progress = self._get_route_progress()
            angle_to_goal = self._get_angle_to_goal()
            obs[34] = progress
            obs[35] = angle_to_goal / 180.0  # 归一化
        
        except Exception as e:
            pass
        
        return obs
    
    def _get_nearby_vehicles(self, max_count: int = 8) -> List[str]:
        """获取附近车辆ID列表"""
        if self.ego_id not in traci.vehicle.getIDList():
            return []
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
            vehicles = []
            
            for veh_id in traci.vehicle.getIDList():
                if veh_id == self.ego_id:
                    continue
                veh_pos = np.array(traci.vehicle.getPosition(veh_id))
                distance = np.linalg.norm(veh_pos - ego_pos)
                vehicles.append((distance, veh_id))
            
            vehicles.sort(key=lambda x: x[0])
            return [veh_id for _, veh_id in vehicles[:max_count]]
        
        except:
            return []
    
    def _get_traffic_light_state(self) -> np.ndarray:
        """获取前方红绿灯状态 [distance, is_red, is_yellow, is_green]"""
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 默认：很远，绿灯
        
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return state
            
            tls_ids = traci.vehicle.getNextTLS(self.ego_id)
            if tls_ids:
                # 获取最近的红绿灯
                tls_id, _, distance, link_state = tls_ids[0]
                state[0] = min(distance / 100.0, 1.0)  # 归一化距离
                
                # 解析红绿灯状态
                if link_state in ['r', 'R']:  # 红灯
                    state[1] = 1.0
                elif link_state in ['y', 'Y']:  # 黄灯
                    state[2] = 1.0
                elif link_state in ['g', 'G']:  # 绿灯
                    state[3] = 1.0
        
        except:
            pass
        
        return state
    
    def _get_distance_to_goal(self) -> float:
        """计算到目标的距离"""
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return self.route_length
            
            # 获取当前位置
            route_index = traci.vehicle.getRouteIndex(self.ego_id)
            lanepos = traci.vehicle.getLanePosition(self.ego_id)
            
            # 计算剩余距离
            remaining_dist = 0.0
            for i in range(route_index, len(self.route_edges)):
                edge_id = self.route_edges[i]
                edge = self.net.getEdge(edge_id)
                if i == route_index:
                    remaining_dist += edge.getLength() - lanepos
                else:
                    remaining_dist += edge.getLength()
            
            return remaining_dist
        
        except:
            return self.route_length
    
    def _get_route_progress(self) -> float:
        """获取路由完成进度 [0, 1]"""
        try:
            distance_to_goal = self._get_distance_to_goal()
            progress = 1.0 - (distance_to_goal / max(self.route_length, 1.0))
            return np.clip(progress, 0.0, 1.0)
        except:
            return 0.0
    
    def _get_angle_to_goal(self) -> float:
        """获取到目标的角度偏差（度）"""
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return 0.0
            
            ego_angle = traci.vehicle.getAngle(self.ego_id)
            
            # 获取目标方向（简化：使用当前车道方向）
            route_index = traci.vehicle.getRouteIndex(self.ego_id)
            if route_index < len(self.route_edges) - 1:
                next_edge_id = self.route_edges[route_index + 1]
                # 这里简化处理，实际可以计算更精确的角度
                return 0.0
            
            return 0.0
        
        except:
            return 0.0
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        if self.ego_id not in traci.vehicle.getIDList():
            return -10.0  # 车辆消失，严重惩罚
        
        try:
            # 1. 目标到达奖励 (+50)
            if self.goal_reached:
                return 50.0
            
            # 2. 碰撞惩罚 (-20)
            if self.collision_occurred:
                return -20.0
            
            # 3. 前进奖励（基于距离减少）
            current_distance = self._get_distance_to_goal()
            distance_reward = (self.last_distance_to_goal - current_distance) * 0.05
            reward += distance_reward
            self.last_distance_to_goal = current_distance
            
            # 4. 速度奖励（保持合理速度）
            speed = traci.vehicle.getSpeed(self.ego_id)
            optimal_speed = 10.0  # m/s (~36 km/h)
            speed_diff = abs(speed - optimal_speed)
            if speed_diff < 2.0:
                reward += 0.5
            elif speed_diff < 5.0:
                reward += 0.2
            
            # 5. 红绿灯遵守奖励 (Stage 2+)
            if self.stage >= 2:
                tls_state = self._get_traffic_light_state()
                if tls_state[1] > 0.5:  # 红灯
                    if speed < 0.5:  # 停车
                        reward += 0.5
                    else:  # 闯红灯
                        reward -= 5.0
                        self.stats["red_light_violations"] += 1
            
            # 6. 保持在路由上
            current_edge = traci.vehicle.getRoadID(self.ego_id)
            if current_edge not in self.route_edges:
                reward -= 1.0
                self.stats["off_route_count"] += 1
            else:
                reward += 0.1
            
            # 7. 时间惩罚（鼓励尽快到达）
            reward -= 0.01
        
        except Exception as e:
            reward = 0.0
        
        return reward
    
    def _check_terminated(self) -> bool:
        """检查是否终止"""
        # 车辆消失
        if self.ego_id not in traci.vehicle.getIDList():
            return True
        
        # 碰撞检测
        try:
            if traci.simulation.getCollidingVehiclesNumber() > 0:
                colliding = traci.simulation.getCollidingVehiclesIDList()
                if self.ego_id in colliding:
                    self.collision_occurred = True
                    self.stats["collisions"] += 1
                    return True
        except:
            pass
        
        # 到达目标
        try:
            distance_to_goal = self._get_distance_to_goal()
            if distance_to_goal < 10.0:  # 10米内算到达
                self.goal_reached = True
                return True
        except:
            pass
        
        return False
    
    def _get_info(self) -> Dict:
        """获取info字典"""
        info = {
            "episode": self.episode_count,
            "step": self.current_step,
            "stage": self.stage,
            "total_reward": self.total_reward,
            "collision": self.collision_occurred,
            "goal_reached": self.goal_reached,
            "route_length": self.route_length,
            "route_progress": self._get_route_progress(),
            "distance_to_goal": self._get_distance_to_goal(),
            **self.stats,
        }
        
        # 添加成功率指标（用于LLM训练顾问）
        if self.goal_reached:
            info["success"] = 1.0
        elif self.collision_occurred:
            info["success"] = 0.0
        else:
            info["success"] = 0.5  # 未完成
        
        return info
    
    def render(self):
        """渲染（SUMO自带GUI）"""
        # SUMO-GUI会自动渲染
        pass
    
    def close(self):
        """关闭环境"""
        self._close_sumo()
    
    def __del__(self):
        """析构函数"""
        self.close()


def make_sumo_env(stage: int, map_name: str = "sf_mission", **kwargs):
    """
    便捷函数：创建SUMO环境
    
    Args:
        stage: 训练阶段 (1-4)
        map_name: 地图名称
        **kwargs: 传递给SUMODrivingEnv的其他参数
    
    Returns:
        SUMODrivingEnv实例
    """
    # 查找地图文件
    script_dir = Path(__file__).parent.parent
    maps_dir = script_dir / "maps"
    net_file = maps_dir / f"{map_name}.net.xml"
    route_file = maps_dir / f"{map_name}_stage{stage}.rou.xml"
    
    if not net_file.exists():
        raise FileNotFoundError(
            f"找不到地图文件: {net_file}\n"
            f"请先运行: python scripts/download_map.py --region {map_name}"
        )
    
    # 根据stage设置默认参数
    stage_defaults = {
        1: {"num_background_vehicles": 0, "num_pedestrians": 0, "max_episode_steps": 500},
        2: {"num_background_vehicles": 0, "num_pedestrians": 0, "max_episode_steps": 600},
        3: {"num_background_vehicles": 15, "num_pedestrians": 0, "max_episode_steps": 700},
        4: {"num_background_vehicles": 20, "num_pedestrians": 10, "max_episode_steps": 1000},
    }
    
    defaults = stage_defaults.get(stage, {})
    defaults.update(kwargs)
    
    env = SUMODrivingEnv(
        stage=stage,
        net_file=str(net_file),
        route_file=str(route_file) if route_file.exists() else None,
        **defaults
    )
    
    return env


if __name__ == "__main__":
    # 测试环境
    print("测试SUMO环境...")
    
    # 检查SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        print("❌ 请设置环境变量 SUMO_HOME")
        sys.exit(1)
    
    print(f"✅ SUMO_HOME: {os.environ['SUMO_HOME']}")
    
    # 注意：需要先运行download_map.py下载地图
    print("\n💡 使用前请先下载地图:")
    print("   python scripts/download_map.py --region sf_mission")

