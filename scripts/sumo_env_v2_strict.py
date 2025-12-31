"""
基于SUMO的自动驾驶环境 - 支持四阶段课程学习 (V2.0)
使用真实美国街道地图，支持从简单到复杂的渐进式学习

V2.0 新特性：
- 102维观测空间（自车8维 + 车辆72维 + 行人16维 + 红绿灯4维 + 路由2维）
- 动态背景车管理（50-150米生成，>200米消失）
- 动态行人管理（30-80米生成，>100米消失）
- 舒适度奖励（惩罚急刹车、急转弯）
- 更丰富的车辆感知（加速度、相对加速度、航向差）

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
    基于SUMO的自动驾驶环境 V2.0
    
    观测空间 (102维):
        - 自车状态 [0:8]: 速度、加速度、位置x/y、航向cos/sin、车道偏移、转向角
        - 周围车辆 [8:80]: 12辆 × 6维（相对位置x/y、相对速度、加速度、相对加速度、航向差）
        - 行人 [80:96]: 4个 × 4维（相对位置x/y、相对速度x/y）
        - 红绿灯 [96:100]: 距离、红/黄/绿状态
        - 路由 [100:102]: 进度、角度
    
    动作空间 (2维):
        - 加速度 [-1, 1] -> [-4.5, 4.5] m/s²
        - 转向 [-1, 1] -> [-30, 30] 度
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    # 观测空间维度常量
    EGO_DIM = 8
    NUM_VEHICLES = 12
    VEHICLE_DIM = 6
    NUM_PEDESTRIANS = 4
    PEDESTRIAN_DIM = 4
    TLS_DIM = 4
    ROUTE_DIM = 2
    
    # 动态生成距离常量
    VEHICLE_SPAWN_MIN = 50.0    # 车辆最小生成距离
    VEHICLE_SPAWN_MAX = 150.0   # 车辆最大生成距离
    VEHICLE_DESPAWN = 200.0     # 车辆消失距离
    
    PEDESTRIAN_SPAWN_MIN = 30.0  # 行人最小生成距离
    PEDESTRIAN_SPAWN_MAX = 80.0  # 行人最大生成距离
    PEDESTRIAN_DESPAWN = 100.0   # 行人消失距离
    
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
        min_route_length: float = 200.0,
        max_route_length: float = 500.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        # 每个环境实例使用唯一的连接标签
        self.connection_label = f"sumo_{id(self)}_{random.randint(0, 999999)}"
        
        self.stage = stage
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.max_episode_steps = max_episode_steps
        self.step_length = step_length
        self.min_route_length = min_route_length
        self.max_route_length = max_route_length
        
        self.num_background_vehicles = self._get_stage_vehicles(stage, num_background_vehicles)
        self.num_pedestrians = self._get_stage_pedestrians(stage, num_pedestrians)
        
        self.sumo_cmd = None
        self.sumo_running = False
        self.net = None
        self.ego_id = "ego"
        
        # Episode状态
        self.current_step = 0
        self.episode_count = 0
        self.start_edge = None
        self.goal_edge = None
        self.route_edges = []
        self.route_length = 0.0
        
        # 奖励相关
        self.total_reward = 0.0
        self.last_distance_to_goal = 0.0
        self.last_speed = 0.0
        self.last_accel = 0.0
        self.last_heading = 0.0
        self.collision_occurred = False
        self.goal_reached = False
        
        # 动态车辆/行人管理
        self.active_bg_vehicles = set()
        self.active_pedestrians = set()
        self.bg_vehicle_counter = 0
        self.pedestrian_counter = 0
        
        # 统计信息
        self.stats = {
            "red_light_violations": 0,
            "collisions": 0,
            "off_route_count": 0,
            "total_distance": 0.0,
            "harsh_braking_count": 0,
            "harsh_steering_count": 0,
        }
        
        if net_file:
            self._load_network()
        
        self._define_spaces()
        
        if seed is not None:
            self.seed(seed)
    
    def _get_stage_vehicles(self, stage: int, override: int) -> int:
        if override > 0:
            return override
        stage_vehicles = {1: 0, 2: 0, 3: 15, 4: 20}
        return stage_vehicles.get(stage, 0)
    
    def _get_stage_pedestrians(self, stage: int, override: int) -> int:
        if override > 0:
            return override
        stage_pedestrians = {1: 0, 2: 0, 3: 0, 4: 10}
        return stage_pedestrians.get(stage, 0)
    
    def _load_network(self):
        try:
            self.net = sumolib.net.readNet(self.net_file)
            print(f"加载SUMO网络: {self.net_file}")
            print(f"   - 路段数: {len(list(self.net.getEdges()))}")
            print(f"   - 交叉口数: {len(list(self.net.getNodes()))}")
        except Exception as e:
            print(f"加载SUMO网络失败: {e}")
            raise
    
    def _define_spaces(self):
        """定义102维观测空间和2维动作空间"""
        obs_dim = (self.EGO_DIM + 
                   self.NUM_VEHICLES * self.VEHICLE_DIM + 
                   self.NUM_PEDESTRIANS * self.PEDESTRIAN_DIM + 
                   self.TLS_DIM + 
                   self.ROUTE_DIM)  # 8 + 72 + 16 + 4 + 2 = 102
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        print(f"📐 观测空间维度: {obs_dim}")
        print(f"   - 自车状态: {self.EGO_DIM}")
        print(f"   - 周围车辆: {self.NUM_VEHICLES} × {self.VEHICLE_DIM} = {self.NUM_VEHICLES * self.VEHICLE_DIM}")
        print(f"   - 行人: {self.NUM_PEDESTRIANS} × {self.PEDESTRIAN_DIM} = {self.NUM_PEDESTRIANS * self.PEDESTRIAN_DIM}")
        print(f"   - 红绿灯: {self.TLS_DIM}")
        print(f"   - 路由: {self.ROUTE_DIM}")
    
    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def _ensure_connection(self):
        """确保使用正确的traci连接"""
        try:
            traci.switch(self.connection_label)
        except traci.exceptions.TraCIException:
            pass
    
    def _start_sumo(self):
        if self.sumo_running:
            self._close_sumo()
        
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        self.sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--start", "true" if self.use_gui else "false",
            "--pedestrian.model", "nonInteracting",  # 启用行人
        ]
        
        if self.route_file and os.path.exists(self.route_file):
            self.sumo_cmd.extend(["-r", self.route_file])
        
        try:
            traci.start(self.sumo_cmd, label=self.connection_label)
            self.sumo_running = True
            self._setup_traffic_lights()  # 初始化红绿灯
        except Exception as e:
            print(f"启动SUMO失败: {e}")
            raise
    
    def _setup_traffic_lights(self):
        """给所有红绿灯设置正确的红绿周期"""
        self._ensure_connection()
        
        try:
            tls_ids = traci.trafficlight.getIDList()
            
            for tls_id in tls_ids:
                try:
                    state = traci.trafficlight.getRedYellowGreenState(tls_id)
                    num_links = len(state)
                    
                    if num_links == 0:
                        continue
                    
                    # 创建简单的两相位：一半绿一半红，然后交换
                    half = max(1, num_links // 2)
                    phase1_state = 'G' * half + 'r' * (num_links - half)
                    phase2_state = 'r' * half + 'G' * (num_links - half)
                    
                    # 随机起始相位，让不同红绿灯不同步
                    import random
                    start_phase = random.randint(0, 3)
                    
                    phases = [
                        traci.trafficlight.Phase(25, phase1_state),   # 25秒绿灯
                        traci.trafficlight.Phase(4, 'y' * num_links), # 4秒黄灯
                        traci.trafficlight.Phase(25, phase2_state),   # 25秒红灯
                        traci.trafficlight.Phase(4, 'y' * num_links), # 4秒黄灯
                    ]
                    
                    logic = traci.trafficlight.Logic('custom', 0, start_phase, phases)
                    traci.trafficlight.setProgramLogic(tls_id, logic)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"红绿灯初始化警告: {e}")
    
    def _close_sumo(self):
        if self.sumo_running:
            try:
                traci.switch(self.connection_label)
                traci.close()
            except:
                pass
            self.sumo_running = False
    
    def _select_random_route(self) -> Tuple[str, str, List[str], float]:
        if not self.net:
            raise ValueError("网络未加载")
        
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
            
            try:
                route_edges = self.net.getShortestPath(start_edge, goal_edge)
                if route_edges[0] is None or len(route_edges[0]) < 2:
                    continue
                
                route = route_edges[0]
                route_length = sum([e.getLength() for e in route])
                
                min_len = self.min_route_length
                max_len = self.max_route_length
                
                if self.stage == 4:
                    min_len = 500.0
                    max_len = 1500.0
                
                if min_len <= route_length <= max_len:
                    edge_ids = [e.getID() for e in route]
                    return start_edge.getID(), goal_edge.getID(), edge_ids, route_length
            
            except Exception as e:
                continue
        
        start_edge = all_edges[0]
        goal_edge = all_edges[-1]
        return start_edge.getID(), goal_edge.getID(), [start_edge.getID(), goal_edge.getID()], 100.0
    
    def _spawn_ego_vehicle(self):
        self._ensure_connection()
        try:
            if self.ego_id in traci.vehicle.getIDList():
                traci.vehicle.remove(self.ego_id)
            
            if self.ego_id not in traci.vehicletype.getIDList():
                traci.vehicletype.copy("DEFAULT_VEHTYPE", self.ego_id)
                traci.vehicletype.setAccel(self.ego_id, 2.6)
                traci.vehicletype.setDecel(self.ego_id, 4.5)
                traci.vehicletype.setMaxSpeed(self.ego_id, 15.0)
                traci.vehicletype.setColor(self.ego_id, (0, 255, 0, 255))
            
            traci.vehicle.add(
                vehID=self.ego_id,
                routeID="",
                typeID=self.ego_id,
                depart="now",
                departLane="best",
                departSpeed="0"
            )
            
            try:
                traci.vehicle.setRoute(self.ego_id, self.route_edges)
            except Exception as route_err:
                traci.vehicle.setRoute(self.ego_id, [self.route_edges[0]])
            
            traci.vehicle.setSpeedMode(self.ego_id, 0)
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
            
        except Exception as e:
            print(f"生成ego车辆失败: {e}")
            raise
    
    def _get_nearby_edges(self) -> List[str]:
        """获取自车路线附近的所有边"""
        nearby_edges = set(self.route_edges)
        for edge_id in self.route_edges:
            try:
                edge = self.net.getEdge(edge_id)
                # 添加相邻边
                for neighbor in edge.getOutgoing():
                    if neighbor.allows("passenger"):
                        nearby_edges.add(neighbor.getID())
                for neighbor in edge.getIncoming():
                    if neighbor.allows("passenger"):
                        nearby_edges.add(neighbor.getID())
            except:
                pass
        return list(nearby_edges)
    
    def _spawn_background_vehicles(self):
        """初始生成背景车辆（在自车附近）"""
        if self.num_background_vehicles == 0:
            return
        
        self._ensure_connection()
        
        # 创建背景车辆类型
        if "background" not in traci.vehicletype.getIDList():
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "background")
            traci.vehicletype.setAccel("background", 2.6)
            traci.vehicletype.setDecel("background", 4.5)
            traci.vehicletype.setMaxSpeed("background", 13.89)
            traci.vehicletype.setColor("background", (255, 255, 0, 255))
        
        nearby_edges = self._get_nearby_edges()
        if not nearby_edges:
            return
        
        # 初始生成一半数量的车辆
        initial_count = self.num_background_vehicles // 2
        for _ in range(initial_count):
            self._try_spawn_one_vehicle(nearby_edges)
    
    def _try_spawn_one_vehicle(self, nearby_edges: List[str] = None) -> bool:
        """尝试在自车附近生成一辆背景车"""
        self._ensure_connection()
        
        if nearby_edges is None:
            nearby_edges = self._get_nearby_edges()
        
        if not nearby_edges:
            return False
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
        except:
            return False
        
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                edge_id = random.choice(nearby_edges)
                edge = self.net.getEdge(edge_id)
                
                # 随机选择边上的位置
                lane = edge.getLane(0)
                lane_length = lane.getLength()
                pos_on_lane = random.uniform(0, lane_length)
                
                # 计算实际位置
                shape = lane.getShape()
                if len(shape) >= 2:
                    spawn_pos = np.array(shape[0])
                else:
                    continue
                
                # 检查距离
                distance = np.linalg.norm(spawn_pos - ego_pos)
                if distance < self.VEHICLE_SPAWN_MIN or distance > self.VEHICLE_SPAWN_MAX:
                    continue
                
                # 生成车辆
                veh_id = f"bg_{self.bg_vehicle_counter}"
                self.bg_vehicle_counter += 1
                
                # 为背景车选择路线
                all_edges = [e.getID() for e in self.net.getEdges() 
                            if not e.isSpecial() and e.allows("passenger")]
                goal_edge = random.choice(all_edges)
                
                try:
                    route = self.net.getShortestPath(edge, self.net.getEdge(goal_edge))[0]
                    if route:
                        route_ids = [e.getID() for e in route]
                    else:
                        route_ids = [edge_id]
                except:
                    route_ids = [edge_id]
                
                traci.vehicle.add(
                    vehID=veh_id,
                    routeID="",
                    typeID="background",
                    depart="now",
                    departLane="random",
                    departSpeed="random"
                )
                traci.vehicle.setRoute(veh_id, route_ids)
                self.active_bg_vehicles.add(veh_id)
                return True
                
            except Exception as e:
                continue
        
        return False
    
    def _update_background_vehicles(self):
        """动态更新背景车辆：移除远离的，生成新的"""
        if self.num_background_vehicles == 0:
            return
        
        self._ensure_connection()
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
        except:
            return
        
        # 移除离自车太远的车辆
        vehicles_to_remove = []
        for veh_id in list(self.active_bg_vehicles):
            try:
                if veh_id not in traci.vehicle.getIDList():
                    vehicles_to_remove.append(veh_id)
                    continue
                
                veh_pos = np.array(traci.vehicle.getPosition(veh_id))
                distance = np.linalg.norm(veh_pos - ego_pos)
                
                if distance > self.VEHICLE_DESPAWN:
                    traci.vehicle.remove(veh_id)
                    vehicles_to_remove.append(veh_id)
            except:
                vehicles_to_remove.append(veh_id)
        
        for veh_id in vehicles_to_remove:
            self.active_bg_vehicles.discard(veh_id)
        
        # 如果车辆数量不足，生成新的
        nearby_edges = self._get_nearby_edges()
        while len(self.active_bg_vehicles) < self.num_background_vehicles:
            if not self._try_spawn_one_vehicle(nearby_edges):
                break
    
    def _spawn_pedestrians(self):
        """初始生成行人"""
        if self.num_pedestrians == 0:
            return
        
        self._ensure_connection()
        
        # 获取可用的人行道
        try:
            # 初始生成一半数量的行人
            initial_count = self.num_pedestrians // 2
            for _ in range(initial_count):
                self._try_spawn_one_pedestrian()
        except Exception as e:
            pass
    
    def _try_spawn_one_pedestrian(self) -> bool:
        """尝试在自车附近生成一个行人"""
        self._ensure_connection()
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
        except:
            return False
        
        # 获取附近的边
        nearby_edges = self._get_nearby_edges()
        if not nearby_edges:
            return False
        
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                edge_id = random.choice(nearby_edges)
                edge = self.net.getEdge(edge_id)
                
                # 获取边的形状
                shape = edge.getShape()
                if len(shape) < 2:
                    continue
                
                # 随机选择边上的位置
                idx = random.randint(0, len(shape) - 1)
                spawn_pos = np.array(shape[idx])
                
                # 添加一些偏移（模拟人行道位置）
                offset = np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
                spawn_pos = spawn_pos + offset
                
                # 检查距离
                distance = np.linalg.norm(spawn_pos - ego_pos)
                if distance < self.PEDESTRIAN_SPAWN_MIN or distance > self.PEDESTRIAN_SPAWN_MAX:
                    continue
                
                # 生成行人
                ped_id = f"ped_{self.pedestrian_counter}"
                self.pedestrian_counter += 1
                
                # 选择目标位置
                goal_edge_id = random.choice(nearby_edges)
                goal_edge = self.net.getEdge(goal_edge_id)
                goal_shape = goal_edge.getShape()
                goal_pos = goal_shape[-1] if goal_shape else spawn_pos
                
                traci.person.add(
                    personID=ped_id,
                    edgeID=edge_id,
                    pos=0,
                    depart=0,
                    typeID="DEFAULT_PEDTYPE"
                )
                
                # 添加行走阶段
                traci.person.appendWalkingStage(
                    personID=ped_id,
                    edges=[edge_id],
                    arrivalPos=edge.getLength()
                )
                
                self.active_pedestrians.add(ped_id)
                return True
                
            except Exception as e:
                continue
        
        return False
    
    def _update_pedestrians(self):
        """动态更新行人：移除远离的，生成新的"""
        if self.num_pedestrians == 0:
            return
        
        self._ensure_connection()
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
        except:
            return
        
        # 移除离自车太远的行人
        peds_to_remove = []
        for ped_id in list(self.active_pedestrians):
            try:
                if ped_id not in traci.person.getIDList():
                    peds_to_remove.append(ped_id)
                    continue
                
                ped_pos = np.array(traci.person.getPosition(ped_id))
                distance = np.linalg.norm(ped_pos - ego_pos)
                
                if distance > self.PEDESTRIAN_DESPAWN:
                    traci.person.remove(ped_id)
                    peds_to_remove.append(ped_id)
            except:
                peds_to_remove.append(ped_id)
        
        for ped_id in peds_to_remove:
            self.active_pedestrians.discard(ped_id)
        
        # 如果行人数量不足，生成新的
        while len(self.active_pedestrians) < self.num_pedestrians:
            if not self._try_spawn_one_pedestrian():
                break
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        if not self.sumo_running:
            self._start_sumo()
        else:
            self._ensure_connection()
            # 移除所有车辆和行人
            for veh_id in traci.vehicle.getIDList():
                try:
                    traci.vehicle.remove(veh_id)
                except:
                    pass
            for ped_id in traci.person.getIDList():
                try:
                    traci.person.remove(ped_id)
                except:
                    pass
        
        # 重置动态管理状态
        self.active_bg_vehicles = set()
        self.active_pedestrians = set()
        
        self.start_edge, self.goal_edge, self.route_edges, self.route_length = \
            self._select_random_route()
        
        self.current_step = 0
        self.total_reward = 0.0
        self.collision_occurred = False
        self.goal_reached = False
        self.last_speed = 0.0
        self.last_accel = 0.0
        self.last_heading = 0.0
        self.stationary_steps = 0  # 连续静止步数计数器
        
        self.stats = {
            "red_light_violations": 0,
            "collisions": 0,
            "off_route_count": 0,
            "total_distance": 0.0,
            "harsh_braking_count": 0,
            "harsh_steering_count": 0,
            "stationary_timeout": False,  # 是否因静止超时
        }
        
        self.route_traffic_lights = self._count_route_traffic_lights()
        
        # 根据红绿灯数量动态调整步数限制
        extra_steps = self.route_traffic_lights * 30
        self.dynamic_max_steps = min(self.max_episode_steps + extra_steps, 1500)
        
        self._spawn_ego_vehicle()
        self._spawn_background_vehicles()
        self._spawn_pedestrians()
        
        self._ensure_connection()
        traci.simulationStep()
        
        obs = self._get_observation()
        self.last_distance_to_goal = self._get_distance_to_goal()
        
        self.episode_count += 1
        
        info = self._get_info()
        self._red_light_punished = False
        return obs, info
    
    def step(self, action):
        if not self.sumo_running:
            raise RuntimeError("SUMO未启动，请先调用reset()")
        
        self._ensure_connection()
        
        accel = action[0] * 4.5
        steer = action[1] * 30.0
        
        try:
            if self.ego_id in traci.vehicle.getIDList():
                current_speed = traci.vehicle.getSpeed(self.ego_id)
                new_speed = max(0, current_speed + accel * self.step_length)
                traci.vehicle.setSpeed(self.ego_id, new_speed)
        except:
            pass
        
        traci.simulationStep()
        self.current_step += 1
        
        # 动态更新背景车辆和行人
        self._update_background_vehicles()
        self._update_pedestrians()
        
        terminated = self._check_terminated()
        obs = self._get_observation()
        reward = self._compute_reward()
        self.total_reward += reward
        
        # 静止超时惩罚
        if self.stats.get("stationary_timeout", False):
            reward = -100.0 - self.total_reward  # 比正常超时更重的惩罚
            self.total_reward = -100.0
        
        truncated = self.current_step >= getattr(self, 'dynamic_max_steps', self.max_episode_steps)
        if truncated and not self.goal_reached:
            reward = -150.0 - self.total_reward
            self.total_reward = -150.0
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取102维观测"""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        self._ensure_connection()
        
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return obs
            
            # ==================== 自车状态 (8维) ====================
            speed = traci.vehicle.getSpeed(self.ego_id)
            accel = traci.vehicle.getAcceleration(self.ego_id)
            pos = traci.vehicle.getPosition(self.ego_id)
            heading = traci.vehicle.getAngle(self.ego_id)
            
            # 获取车道偏移
            try:
                lane_id = traci.vehicle.getLaneID(self.ego_id)
                lane_pos = traci.vehicle.getLanePosition(self.ego_id)
                lateral_offset = traci.vehicle.getLateralLanePosition(self.ego_id)
            except:
                lateral_offset = 0.0
            
            # 计算转向角（航向变化率）
            heading_diff = heading - self.last_heading if self.last_heading != 0 else 0
            # 归一化到 [-180, 180]
            if heading_diff > 180:
                heading_diff -= 360
            elif heading_diff < -180:
                heading_diff += 360
            
            obs[0] = speed / 15.0
            obs[1] = accel / 4.5
            obs[2] = pos[0] / 1000.0
            obs[3] = pos[1] / 1000.0
            obs[4] = np.cos(np.radians(heading))
            obs[5] = np.sin(np.radians(heading))
            obs[6] = lateral_offset / 3.0  # 车道宽度约3米
            obs[7] = heading_diff / 30.0   # 归一化转向角
            
            self.last_heading = heading
            
            # ==================== 周围车辆 (72维 = 12辆 × 6维) ====================
            ego_pos = np.array(pos)
            nearby_vehicles = self._get_nearby_vehicles_detailed(max_count=self.NUM_VEHICLES)
            
            idx_base = self.EGO_DIM  # 8
            for i, veh_info in enumerate(nearby_vehicles):
                if i >= self.NUM_VEHICLES:
                    break
                idx = idx_base + i * self.VEHICLE_DIM
                obs[idx:idx+self.VEHICLE_DIM] = veh_info
            
            # ==================== 行人 (16维 = 4个 × 4维) ====================
            idx_base = self.EGO_DIM + self.NUM_VEHICLES * self.VEHICLE_DIM  # 8 + 72 = 80
            nearby_peds = self._get_nearby_pedestrians_detailed(max_count=self.NUM_PEDESTRIANS)
            
            for i, ped_info in enumerate(nearby_peds):
                if i >= self.NUM_PEDESTRIANS:
                    break
                idx = idx_base + i * self.PEDESTRIAN_DIM
                obs[idx:idx+self.PEDESTRIAN_DIM] = ped_info
            
            # ==================== 红绿灯 (4维) ====================
            idx_base = self.EGO_DIM + self.NUM_VEHICLES * self.VEHICLE_DIM + self.NUM_PEDESTRIANS * self.PEDESTRIAN_DIM  # 96
            tls_state = self._get_traffic_light_state()
            obs[idx_base:idx_base+self.TLS_DIM] = tls_state
            
            # ==================== 路由 (2维) ====================
            idx_base = idx_base + self.TLS_DIM  # 100
            progress = self._get_route_progress()
            angle_to_goal = self._get_angle_to_goal()
            obs[idx_base] = progress
            obs[idx_base + 1] = angle_to_goal / 180.0
        
        except Exception as e:
            pass
        
        return obs
    
    def _get_nearby_vehicles_detailed(self, max_count: int = 12) -> List[np.ndarray]:
        """获取附近车辆的详细信息"""
        self._ensure_connection()
        
        result = []
        
        if self.ego_id not in traci.vehicle.getIDList():
            return result
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            ego_accel = traci.vehicle.getAcceleration(self.ego_id)
            ego_heading = traci.vehicle.getAngle(self.ego_id)
            
            vehicles = []
            for veh_id in traci.vehicle.getIDList():
                if veh_id == self.ego_id:
                    continue
                try:
                    veh_pos = np.array(traci.vehicle.getPosition(veh_id))
                    distance = np.linalg.norm(veh_pos - ego_pos)
                    vehicles.append((distance, veh_id, veh_pos))
                except:
                    pass
            
            vehicles.sort(key=lambda x: x[0])
            
            for distance, veh_id, veh_pos in vehicles[:max_count]:
                try:
                    veh_speed = traci.vehicle.getSpeed(veh_id)
                    veh_accel = traci.vehicle.getAcceleration(veh_id)
                    veh_heading = traci.vehicle.getAngle(veh_id)
                    
                    rel_pos = veh_pos - ego_pos
                    rel_speed = veh_speed - ego_speed
                    rel_accel = veh_accel - ego_accel
                    heading_diff = veh_heading - ego_heading
                    
                    # 归一化航向差到 [-180, 180]
                    if heading_diff > 180:
                        heading_diff -= 360
                    elif heading_diff < -180:
                        heading_diff += 360
                    
                    # 6维：相对位置x/y、相对速度、加速度、相对加速度、航向差
                    info = np.array([
                        rel_pos[0] / 50.0,
                        rel_pos[1] / 50.0,
                        rel_speed / 15.0,
                        veh_accel / 4.5,
                        rel_accel / 4.5,
                        heading_diff / 180.0
                    ], dtype=np.float32)
                    
                    result.append(info)
                except:
                    pass
        
        except:
            pass
        
        return result
    
    def _get_nearby_pedestrians_detailed(self, max_count: int = 4) -> List[np.ndarray]:
        """获取附近行人的详细信息"""
        self._ensure_connection()
        
        result = []
        
        if self.ego_id not in traci.vehicle.getIDList():
            return result
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
            ego_speed = traci.vehicle.getSpeed(self.ego_id)
            
            pedestrians = []
            for ped_id in traci.person.getIDList():
                try:
                    ped_pos = np.array(traci.person.getPosition(ped_id))
                    distance = np.linalg.norm(ped_pos - ego_pos)
                    pedestrians.append((distance, ped_id, ped_pos))
                except:
                    pass
            
            pedestrians.sort(key=lambda x: x[0])
            
            for distance, ped_id, ped_pos in pedestrians[:max_count]:
                try:
                    ped_speed = traci.person.getSpeed(ped_id)
                    ped_angle = traci.person.getAngle(ped_id)
                    
                    rel_pos = ped_pos - ego_pos
                    
                    # 计算行人速度分量
                    ped_vx = ped_speed * np.sin(np.radians(ped_angle))
                    ped_vy = ped_speed * np.cos(np.radians(ped_angle))
                    
                    # 4维：相对位置x/y、相对速度x/y
                    info = np.array([
                        rel_pos[0] / 30.0,
                        rel_pos[1] / 30.0,
                        ped_vx / 5.0,  # 行人速度约1-2 m/s
                        ped_vy / 5.0
                    ], dtype=np.float32)
                    
                    result.append(info)
                except:
                    pass
        
        except:
            pass
        
        return result
    
    def _get_nearby_vehicles(self, max_count: int = 8) -> List[str]:
        """获取附近车辆ID列表（兼容旧代码）"""
        self._ensure_connection()
        
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
    
    def _count_route_traffic_lights(self) -> int:
        """统计路线上的红绿灯数量"""
        self._ensure_connection()
        try:
            tls_ids = set()
            for edge_id in self.route_edges:
                lanes = traci.edge.getLaneNumber(edge_id)
                for i in range(lanes):
                    lane_id = f"{edge_id}_{i}"
                    links = traci.lane.getLinks(lane_id)
                    for link in links:
                        if len(link) >= 5 and link[4]:
                            tls_ids.add(link[4])
            return len(tls_ids)
        except:
            return 0
    
    def _get_traffic_light_state(self) -> np.ndarray:
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        self._ensure_connection()
        
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return state
            
            tls_ids = traci.vehicle.getNextTLS(self.ego_id)
            if tls_ids:
                tls_id, _, distance, link_state = tls_ids[0]
                state[0] = min(distance / 100.0, 1.0)
                
                if link_state in ['r', 'R']:
                    state[1] = 1.0
                elif link_state in ['y', 'Y']:
                    state[2] = 1.0
                elif link_state in ['g', 'G']:
                    state[3] = 1.0
        
        except:
            pass
        
        return state
    
    def _get_distance_to_goal(self) -> float:
        self._ensure_connection()
        
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return self.route_length
            
            route_index = traci.vehicle.getRouteIndex(self.ego_id)
            lanepos = traci.vehicle.getLanePosition(self.ego_id)
            
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
        try:
            distance_to_goal = self._get_distance_to_goal()
            progress = 1.0 - (distance_to_goal / max(self.route_length, 1.0))
            return np.clip(progress, 0.0, 1.0)
        except:
            return 0.0
    
    def _get_angle_to_goal(self) -> float:
        self._ensure_connection()
        
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return 0.0
            
            ego_angle = traci.vehicle.getAngle(self.ego_id)
            route_index = traci.vehicle.getRouteIndex(self.ego_id)
            if route_index < len(self.route_edges) - 1:
                next_edge_id = self.route_edges[route_index + 1]
                return 0.0
            
            return 0.0
        
        except:
            return 0.0
    
    def _compute_reward(self) -> float:
        reward = 0.0
        
        self._ensure_connection()
        
        if self.ego_id not in traci.vehicle.getIDList():
            return -10.0
        
        try:
            if self.goal_reached:
                return 200.0
            
            if self.collision_occurred:
                return -50.0  # 增加碰撞惩罚
            
            # ==================== 距离奖励 ====================
            current_distance = self._get_distance_to_goal()
            distance_reward = (self.last_distance_to_goal - current_distance) * 0.2  # 提高，鼓励前进
            reward += distance_reward
            self.last_distance_to_goal = current_distance
            
            # ==================== 速度奖励 ====================
            speed = traci.vehicle.getSpeed(self.ego_id)
            optimal_speed = 10.0
            speed_diff = abs(speed - optimal_speed)
            if speed_diff < 2.0:
                reward += 0.02  # 降低，让红灯惩罚更突出
            elif speed_diff < 5.0:
                reward += 0.01
            
            # ==================== 舒适度奖励 ====================
            accel = traci.vehicle.getAcceleration(self.ego_id)
            
            # 急刹车惩罚
            if accel < -3.0:
                reward -= 0.5
                self.stats["harsh_braking_count"] += 1
            
            # 急加速惩罚
            if accel > 2.5:
                reward -= 0.2
            
            # 急转弯惩罚
            heading = traci.vehicle.getAngle(self.ego_id)
            heading_diff = abs(heading - self.last_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            if heading_diff > 15:  # 每步超过15度算急转弯
                reward -= 0.3
                self.stats["harsh_steering_count"] += 1
            
            self.last_accel = accel
            
            # ==================== 红绿灯奖励 ====================
            if self.stage >= 2:
                tls_state = self._get_traffic_light_state()
                distance_to_light = tls_state[0] * 100
                is_red = tls_state[1] > 0.5
                is_yellow = tls_state[2] > 0.5
                is_green = tls_state[3] > 0.5
                
                if is_red:
                    # 红灯逻辑
                    if distance_to_light < 50:
                        expected_speed = max(0, distance_to_light / 5)
                        if speed <= expected_speed + 1:
                            reward += 0.5  # 合理减速
                        else:
                            reward -= (speed - expected_speed) * 1.0  # 超速惩罚
                    
                    if distance_to_light < 10:
                        if speed < 0.5:
                            reward += 2.0  # 红灯停车奖励
                        elif speed < 3.0:
                            reward += 0.3
                        else:
                            # 闯红灯！
                            if not getattr(self, '_red_light_punished', False):
                                reward -= 200.0
                                self.stats["red_light_violations"] += 1
                                self._red_light_punished = True
                else:
                    # 非红灯（绿灯或黄灯或无灯）
                    self._red_light_punished = False
                    
                    # 绿灯近距离却不走 → 惩罚
                    if is_green and distance_to_light < 50:
                        if speed < 1.0:
                            reward -= 1.0  # 绿灯不走，严重！
                        elif speed < 3.0:
                            reward -= 0.3  # 绿灯太慢
                    
                    # 非红灯时停着不动 → 惩罚（鼓励前进）
                    if speed < 0.5 and not is_red:
                        reward -= 0.3
            
            # ==================== 避障奖励 (Stage 3+) ====================
            if self.stage >= 3:
                # 与前车保持安全距离
                nearby_vehicles = self._get_nearby_vehicles(max_count=1)
                if nearby_vehicles:
                    try:
                        front_veh_id = nearby_vehicles[0]
                        front_pos = np.array(traci.vehicle.getPosition(front_veh_id))
                        ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
                        distance = np.linalg.norm(front_pos - ego_pos)
                        
                        # 安全距离 = 速度 × 2秒
                        safe_distance = max(speed * 2, 5)
                        
                        if distance < safe_distance:
                            reward -= (safe_distance - distance) * 0.1
                        elif distance < safe_distance * 2:
                            reward += 0.1  # 保持安全距离奖励
                    except:
                        pass
            
            # ==================== 行人避让奖励 (Stage 4+) ====================
            if self.stage >= 4:
                nearby_peds = self._get_nearby_pedestrians_detailed(max_count=1)
                if nearby_peds:
                    ped_info = nearby_peds[0]
                    ped_distance = np.sqrt(ped_info[0]**2 + ped_info[1]**2) * 30  # 反归一化
                    
                    if ped_distance < 10:
                        # 行人太近，必须减速
                        if speed > 3:
                            reward -= 2.0
                        else:
                            reward += 1.0
                    elif ped_distance < 20:
                        if speed < 5:
                            reward += 0.5
            
            # ==================== 路线奖励 ====================
            current_edge = traci.vehicle.getRoadID(self.ego_id)
            if current_edge not in self.route_edges:
                reward -= 1.0
                self.stats["off_route_count"] += 1
            else:
                reward += 0.01  # 降低，让红灯惩罚更突出
            
            # 时间惩罚
            reward -= 0.1
        
        except Exception as e:
            reward = 0.0
        
        return reward
    
    def _check_terminated(self) -> bool:
        self._ensure_connection()
        
        if self.collision_occurred:
            return True
        
        if self.ego_id not in traci.vehicle.getIDList():
            return True
        
        try:
            if traci.simulation.getCollidingVehiclesNumber() > 0:
                colliding = traci.simulation.getCollidingVehiclesIDList()
                if self.ego_id in colliding:
                    self.collision_occurred = True
                    self.stats["collisions"] += 1
                    return True
        except:
            pass
        
        try:
            distance_to_goal = self._get_distance_to_goal()
            if distance_to_goal < 10.0:
                self.goal_reached = True
                return True
        except:
            pass
        
        # 检测连续静止（非红灯时）
        try:
            speed = traci.vehicle.getSpeed(self.ego_id)
            tls_state = self._get_traffic_light_state()
            is_red = tls_state[1] > 0.5
            
            if speed < 0.5 and not is_red:
                self.stationary_steps += 1
            else:
                self.stationary_steps = 0
            
            # 连续静止100步（非红灯）→ 终止
            if self.stationary_steps >= 100:
                self.stats["stationary_timeout"] = True
                return True
        except:
            pass
        
        return False
    
    def _get_info(self) -> Dict:
        info = {
            "ep_count": self.episode_count,
            "step": self.current_step,
            "stage": self.stage,
            "total_reward": self.total_reward,
            "collision": self.collision_occurred,
            "goal_reached": self.goal_reached,
            "route_length": self.route_length,
            "route_progress": self._get_route_progress(),
            "distance_to_goal": self._get_distance_to_goal(),
            "route_traffic_lights": getattr(self, 'route_traffic_lights', 0),
            "max_steps": getattr(self, 'dynamic_max_steps', self.max_episode_steps),
            "active_vehicles": len(self.active_bg_vehicles),
            "active_pedestrians": len(self.active_pedestrians),
            **self.stats,
        }
        
        if self.goal_reached:
            info["success"] = 1.0
        elif self.collision_occurred:
            info["success"] = 0.0
        else:
            info["success"] = 0.5
        
        return info
    
    def render(self):
        pass
    
    def close(self):
        self._close_sumo()
    
    def __del__(self):
        self.close()


def make_sumo_env(stage: int, map_name: str = "sf_mission", **kwargs):
    script_dir = Path(__file__).parent.parent
    maps_dir = script_dir / "maps"
    net_file = maps_dir / f"{map_name}.net.xml"
    route_file = maps_dir / f"{map_name}_stage{stage}.rou.xml"
    
    if not net_file.exists():
        raise FileNotFoundError(
            f"找不到地图文件: {net_file}\n"
            f"请先运行: python scripts/download_map.py --region {map_name}"
        )
    
    stage_defaults = {
        1: {"num_background_vehicles": 0, "num_pedestrians": 0, "max_episode_steps": 800,
            "min_route_length": 200.0, "max_route_length": 500.0},
        2: {"num_background_vehicles": 0, "num_pedestrians": 0, "max_episode_steps": 1500,
            "min_route_length": 600.0, "max_route_length": 1200.0},  # 路线拉长，红绿灯更多！
        3: {"num_background_vehicles": 15, "num_pedestrians": 0, "max_episode_steps": 1500,
            "min_route_length": 600.0, "max_route_length": 1200.0},
        4: {"num_background_vehicles": 20, "num_pedestrians": 10, "max_episode_steps": 2000,
            "min_route_length": 800.0, "max_route_length": 1500.0},
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
    print("=" * 60)
    print("SUMO自动驾驶环境 V2.0")
    print("=" * 60)
    
    if 'SUMO_HOME' not in os.environ:
        print("请设置环境变量 SUMO_HOME")
        sys.exit(1)
    
    print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")
    print("\n📐 观测空间: 102维")
    print("   - 自车状态: 8维")
    print("   - 周围车辆: 12辆 × 6维 = 72维")
    print("   - 行人: 4个 × 4维 = 16维")
    print("   - 红绿灯: 4维")
    print("   - 路由: 2维")
    print("\n动态背景车: 50-150m生成, >200m消失")
    print("动态行人: 30-80m生成, >100m消失")
    print("\n使用前请先下载地图:")
    print("   python scripts/download_map.py --region sf_mission")
