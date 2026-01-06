"""
基于SUMO的自动驾驶环境 - 支持四阶段课程学习 (V2.3 - 背景车辆分布优化版)
使用真实美国街道地图，支持从简单到复杂的渐进式学习

V2.3 修复内容：
- 扩大背景车生成范围：2-3跳邻居边，让车分布到更多平行道路
- 限制同边车辆数：每条边最多3辆，避免堆积
- 降低ego路线生成概率：70%在其他路，30%在ego路线，增加转弯避障场景

V2.2 修复内容：
- 修复背景车辆堆积问题：先创建路由再添加车辆，避免"幽灵车"
- 移除到达终点的背景车辆
- 降低车辆生成频率：每5步最多生成1辆

V2.1 修复内容：
- 扩大红绿灯观测距离（200米分段归一化）
- 添加红绿灯剩余时间信息（第5维）
- 持续跟踪红绿灯状态，防止高速跳过检测
- 渐进式减速奖励
- 更平衡的奖惩比例

观测空间：103维（自车8维 + 车辆72维 + 行人16维 + 红绿灯5维 + 路由2维）

Stage 1: 空路导航 - 学习不偏离车道、到达终点
Stage 2: 红绿灯遵守 - 学习遵守交通信号
Stage 3: 动态避障 - 学习与其他车辆交互
Stage 4: 综合场景 - 行人 + 长距离导航
"""

import os
import sys
import uuid
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
    基于SUMO的自动驾驶环境 V2.3 (背景车辆分布优化版，103维)
    
    观测空间 (103维):
        - 自车状态 [0:8]: 速度、加速度、位置x/y、航向cos/sin、车道偏移、转向角
        - 周围车辆 [8:80]: 12辆 × 6维（相对位置x/y、相对速度、加速度、相对加速度、航向差）
        - 行人 [80:96]: 4个 × 4维（相对位置x/y、相对速度x/y）
        - 红绿灯 [96:101]: 距离(分段归一化)、红/黄/绿状态、剩余时间
        - 路由 [101:103]: 进度、角度
    
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
    TLS_DIM = 5  # 103维版本，增加红绿灯剩余时间
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
        # 使用uuid确保绝对唯一的ego_id
        self.ego_id = f"ego_{uuid.uuid4().hex[:12]}"
        
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
        
        # ========== 修复：红绿灯跟踪状态 ==========
        self.approaching_red_light = False
        self.red_light_distance_when_detected = 0.0
        self.passed_traffic_lights = set()  # 已通过的红绿灯
        self.current_tls_id = None  # 当前接近的红绿灯ID
        self._red_light_punished = False
        # ==========================================
        
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
        """定义103维观测空间和2维动作空间（修复版）"""
        obs_dim = (self.EGO_DIM + 
                   self.NUM_VEHICLES * self.VEHICLE_DIM + 
                   self.NUM_PEDESTRIANS * self.PEDESTRIAN_DIM + 
                   self.TLS_DIM + 
                   self.ROUTE_DIM)  # 8 + 72 + 16 + 5 + 2 = 103
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        print(f"📐 观测空间维度: {obs_dim} (V2.3优化版)")
        print(f"   - 自车状态: {self.EGO_DIM}")
        print(f"   - 周围车辆: {self.NUM_VEHICLES} × {self.VEHICLE_DIM} = {self.NUM_VEHICLES * self.VEHICLE_DIM}")
        print(f"   - 行人: {self.NUM_PEDESTRIANS} × {self.PEDESTRIAN_DIM} = {self.NUM_PEDESTRIANS * self.PEDESTRIAN_DIM}")
        print(f"   - 红绿灯: {self.TLS_DIM} (含剩余时间)")
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
        
        # Windows用NUL，Linux/Mac用/dev/null
        error_log = "NUL" if sys.platform == "win32" else "/dev/null"
        
        self.sumo_cmd = [
            sumo_binary,
            "-n", self.net_file,
            "--step-length", str(self.step_length),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--error-log", error_log,  # 隐藏错误输出
            "--message-log", error_log,  # 隐藏消息输出
            "-v", "false",  # 关闭详细输出
            "--duration-log.disable", "true",  # 关闭持续时间日志
            "--time-to-teleport", "-1",
            "--collision.action", "warn",
            "--start", "true" if self.use_gui else "false",
            "--pedestrian.model", "nonInteracting",
        ]
        
        if self.route_file and os.path.exists(self.route_file):
            self.sumo_cmd.extend(["-r", self.route_file])
        
        try:
            traci.start(self.sumo_cmd, label=self.connection_label)
            self.sumo_running = True
        except Exception as e:
            print(f"启动SUMO失败: {e}")
            raise
    
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
        # 切换到正确的连接
        traci.switch(self.connection_label)
        
        # 确保vehicletype存在
        vtype_id = "ego_type"
        if vtype_id not in traci.vehicletype.getIDList():
            traci.vehicletype.copy("DEFAULT_VEHTYPE", vtype_id)
            traci.vehicletype.setAccel(vtype_id, 2.6)
            traci.vehicletype.setDecel(vtype_id, 4.5)
            traci.vehicletype.setMaxSpeed(vtype_id, 15.0)
            traci.vehicletype.setColor(vtype_id, (0, 255, 0, 255))
        
        # 添加车辆
        traci.vehicle.add(
            vehID=self.ego_id,
            routeID="",
            typeID=vtype_id,
            depart="now",
            departLane="best",
            departSpeed="0"
        )
        
        # 设置路由
        try:
            traci.vehicle.setRoute(self.ego_id, self.route_edges)
        except Exception as route_err:
            traci.vehicle.setRoute(self.ego_id, [self.route_edges[0]])
        
        traci.vehicle.setSpeedMode(self.ego_id, 0)
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)
    
    def _get_nearby_edges(self, hops: int = 2) -> List[str]:
        """
        获取自车路线附近的所有边（V2.3：扩大到多跳邻居）
        
        Args:
            hops: 邻居跳数，默认2跳
        
        Returns:
            nearby_edges: 附近边的列表
        """
        nearby_edges = set(self.route_edges)
        current_level = set(self.route_edges)
        
        for _ in range(hops):
            next_level = set()
            for edge_id in current_level:
                try:
                    edge = self.net.getEdge(edge_id)
                    for neighbor in edge.getOutgoing():
                        if neighbor.allows("passenger"):
                            neighbor_id = neighbor.getID()
                            if neighbor_id not in nearby_edges:
                                next_level.add(neighbor_id)
                                nearby_edges.add(neighbor_id)
                    for neighbor in edge.getIncoming():
                        if neighbor.allows("passenger"):
                            neighbor_id = neighbor.getID()
                            if neighbor_id not in nearby_edges:
                                next_level.add(neighbor_id)
                                nearby_edges.add(neighbor_id)
                except:
                    pass
            current_level = next_level
            if not current_level:
                break
        
        return list(nearby_edges)
    
    def _spawn_background_vehicles(self):
        """初始生成背景车辆（在自车附近）"""
        if self.num_background_vehicles == 0:
            return
        
        self._ensure_connection()
        
        if "background" not in traci.vehicletype.getIDList():
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "background")
            traci.vehicletype.setAccel("background", 2.6)
            traci.vehicletype.setDecel("background", 4.5)
            traci.vehicletype.setMaxSpeed("background", 13.89)
            traci.vehicletype.setColor("background", (255, 255, 0, 255))
        
        nearby_edges = self._get_nearby_edges()
        if not nearby_edges:
            return
        
        initial_count = self.num_background_vehicles // 2
        for _ in range(initial_count):
            self._try_spawn_one_vehicle(nearby_edges)
    
    def _try_spawn_one_vehicle(self, nearby_edges: List[str] = None) -> bool:
        """
        尝试在自车附近生成一辆背景车（V2.3优化版）
        
        改进：
        - 先创建路由再添加车辆，避免幽灵车
        - 每条边最多3辆车，避免堆积
        - 70%概率在非ego路线生成，增加转弯避障场景
        """
        self._ensure_connection()
        
        if nearby_edges is None:
            nearby_edges = self._get_nearby_edges()
        
        if not nearby_edges or len(nearby_edges) < 2:
            return False
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
        except:
            return False
        
        # 方案3：将边分为 ego 路线和其他路线
        ego_route_edges = [e for e in nearby_edges if e in self.route_edges]
        other_edges = [e for e in nearby_edges if e not in self.route_edges]
        
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                # 方案3：70% 概率选其他路线，30% 选 ego 路线
                if other_edges and random.random() < 0.7:
                    edge_id = random.choice(other_edges)
                elif ego_route_edges:
                    edge_id = random.choice(ego_route_edges)
                else:
                    edge_id = random.choice(nearby_edges)
                
                # 方案2：检查这条边上已有几辆车，最多3辆
                try:
                    vehicles_on_edge = traci.edge.getLastStepVehicleIDs(edge_id)
                    # 过滤掉 ego 车辆
                    bg_on_edge = [v for v in vehicles_on_edge if v != self.ego_id]
                    if len(bg_on_edge) >= 3:
                        continue  # 这条边已经有3辆车了，换一条
                except:
                    pass  # 查询失败就继续，不阻塞生成
                
                edge = self.net.getEdge(edge_id)
                
                lane = edge.getLane(0)
                lane_length = lane.getLength()
                pos_on_lane = random.uniform(0, lane_length)
                
                shape = lane.getShape()
                if len(shape) >= 2:
                    spawn_pos = np.array(shape[0])
                else:
                    continue
                
                distance = np.linalg.norm(spawn_pos - ego_pos)
                if distance < self.VEHICLE_SPAWN_MIN or distance > self.VEHICLE_SPAWN_MAX:
                    continue
                
                veh_id = f"bg_{self.bg_vehicle_counter}"
                self.bg_vehicle_counter += 1
                
                # 先验证路由可行性
                route_ids = None
                for route_attempt in range(5):
                    try:
                        goal_edge = random.choice(nearby_edges)
                        route = self.net.getShortestPath(edge, self.net.getEdge(goal_edge))[0]
                        if route and len(route) > 0:
                            route_ids = [e.getID() for e in route]
                            break
                    except:
                        continue
                
                if route_ids is None or len(route_ids) == 0:
                    continue
                
                # 关键修复：先创建路由，再添加车辆
                route_id = f"route_bg_{self.bg_vehicle_counter}"
                try:
                    traci.route.add(route_id, route_ids)
                    traci.vehicle.add(
                        vehID=veh_id,
                        routeID=route_id,
                        typeID="background",
                        depart="now",
                        departLane="random",
                        departSpeed="random"
                    )
                    self.active_bg_vehicles.add(veh_id)
                    return True
                except:
                    try:
                        traci.route.remove(route_id)
                    except:
                        pass
                    continue
                
            except Exception as e:
                continue
        
        return False
    
    def _update_background_vehicles(self):
        """动态更新背景车辆（修复版：移除到达终点的车，降低生成频率）"""
        if self.num_background_vehicles == 0:
            return
        
        self._ensure_connection()
        
        try:
            ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
        except:
            return
        
        vehicles_to_remove = []
        for veh_id in list(self.active_bg_vehicles):
            try:
                if veh_id not in traci.vehicle.getIDList():
                    vehicles_to_remove.append(veh_id)
                    continue
                
                veh_pos = np.array(traci.vehicle.getPosition(veh_id))
                distance = np.linalg.norm(veh_pos - ego_pos)
                
                # 条件1: 距离超过消失距离
                if distance > self.VEHICLE_DESPAWN:
                    traci.vehicle.remove(veh_id)
                    vehicles_to_remove.append(veh_id)
                    continue
                
                # 条件2: 车辆已到达路线终点（关键修复）
                try:
                    route_index = traci.vehicle.getRouteIndex(veh_id)
                    route = traci.vehicle.getRoute(veh_id)
                    lane_pos = traci.vehicle.getLanePosition(veh_id)
                    
                    if route and route_index >= len(route) - 1:
                        edge_id = route[-1]
                        edge_length = traci.lane.getLength(f"{edge_id}_0")
                        if lane_pos > edge_length - 5:  # 接近终点
                            traci.vehicle.remove(veh_id)
                            vehicles_to_remove.append(veh_id)
                            continue
                except:
                    pass
                    
            except:
                vehicles_to_remove.append(veh_id)
        
        for veh_id in vehicles_to_remove:
            self.active_bg_vehicles.discard(veh_id)
        
        # 每步尝试生成一辆新车（V2.3调整：从每5步改为每步）
        if len(self.active_bg_vehicles) < self.num_background_vehicles:
            nearby_edges = self._get_nearby_edges()
            self._try_spawn_one_vehicle(nearby_edges)
    
    def _spawn_pedestrians(self):
        """初始生成行人"""
        if self.num_pedestrians == 0:
            return
        
        self._ensure_connection()
        
        try:
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
        
        nearby_edges = self._get_nearby_edges()
        if not nearby_edges:
            return False
        
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                edge_id = random.choice(nearby_edges)
                edge = self.net.getEdge(edge_id)
                
                shape = edge.getShape()
                if len(shape) < 2:
                    continue
                
                idx = random.randint(0, len(shape) - 1)
                spawn_pos = np.array(shape[idx])
                
                offset = np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
                spawn_pos = spawn_pos + offset
                
                distance = np.linalg.norm(spawn_pos - ego_pos)
                if distance < self.PEDESTRIAN_SPAWN_MIN or distance > self.PEDESTRIAN_SPAWN_MAX:
                    continue
                
                ped_id = f"ped_{self.pedestrian_counter}"
                self.pedestrian_counter += 1
                
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
        
        while len(self.active_pedestrians) < self.num_pedestrians:
            if not self._try_spawn_one_pedestrian():
                break
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        # 每次reset都重启SUMO，确保干净状态
        if self.sumo_running:
            self._close_sumo()
        self._start_sumo()
        
        # 重置动态管理状态
        self.active_bg_vehicles = set()
        self.active_pedestrians = set()
        
        # 统计每步的背景车数量
        self.bg_vehicle_counts = []
        
        self.start_edge, self.goal_edge, self.route_edges, self.route_length = \
            self._select_random_route()
        
        self.current_step = 0
        self.total_reward = 0.0
        self.collision_occurred = False
        self.goal_reached = False
        self.last_speed = 0.0
        self.last_accel = 0.0
        self.last_heading = 0.0
        self.stationary_steps = 0
        
        # ========== 修复：重置红绿灯跟踪状态 ==========
        self.approaching_red_light = False
        self.red_light_distance_when_detected = 0.0
        self.passed_traffic_lights = set()
        self.current_tls_id = None
        self._red_light_punished = False
        # =============================================
        
        self.stats = {
            "red_light_violations": 0,
            "collisions": 0,
            "off_route_count": 0,
            "total_distance": 0.0,
            "harsh_braking_count": 0,
            "harsh_steering_count": 0,
            "stationary_timeout": False,
        }
        
        self.route_traffic_lights = self._count_route_traffic_lights()
        
        extra_steps = self.route_traffic_lights * 50
        self.dynamic_max_steps = self.max_episode_steps + extra_steps
        
        self._spawn_ego_vehicle()
        self._spawn_background_vehicles()
        self._spawn_pedestrians()
        
        self._ensure_connection()
        traci.simulationStep()
        
        obs = self._get_observation()
        self.last_distance_to_goal = self._get_distance_to_goal()
        
        self.episode_count += 1
        
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        if not self.sumo_running:
            raise RuntimeError("SUMO未启动，请先调用reset()")
        
        self._ensure_connection()
        
        accel = action[0] * 4.5
        steer = action[1] * 30.0
        
        try:
            if self.ego_id in traci.vehicle.getIDList():
                # 修复：使用slowDown而不是setSpeed，尊重物理限制
                current_speed = traci.vehicle.getSpeed(self.ego_id)
                target_speed = max(0, min(current_speed + accel * self.step_length, 15.0))
                traci.vehicle.slowDown(self.ego_id, target_speed, self.step_length)
        except:
            pass
        
        traci.simulationStep()
        self.current_step += 1
        
        self._update_background_vehicles()
        self._update_pedestrians()
        
        # 记录当前背景车数量
        self.bg_vehicle_counts.append(len(self.active_bg_vehicles))
        
        terminated = self._check_terminated()
        obs = self._get_observation()
        reward = self._compute_reward()
        self.total_reward += reward
        
        # 静止超时惩罚
        if self.stats.get("stationary_timeout", False):
            reward = -100.0 - self.total_reward
            self.total_reward = -100.0
        
        truncated = self.current_step >= getattr(self, 'dynamic_max_steps', self.max_episode_steps)
        if truncated and not self.goal_reached:
            reward = -1000.0 - self.total_reward
            self.total_reward = -1000.0
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取103维观测（修复版）"""
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
            
            try:
                lane_id = traci.vehicle.getLaneID(self.ego_id)
                lane_pos = traci.vehicle.getLanePosition(self.ego_id)
                lateral_offset = traci.vehicle.getLateralLanePosition(self.ego_id)
            except:
                lateral_offset = 0.0
            
            heading_diff = heading - self.last_heading if self.last_heading != 0 else 0
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
            obs[6] = lateral_offset / 3.0
            obs[7] = heading_diff / 30.0
            
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
            
            # ==================== 红绿灯 (5维，修复版) ====================
            idx_base = self.EGO_DIM + self.NUM_VEHICLES * self.VEHICLE_DIM + self.NUM_PEDESTRIANS * self.PEDESTRIAN_DIM  # 96
            tls_state = self._get_traffic_light_state()
            obs[idx_base:idx_base+self.TLS_DIM] = tls_state
            
            # ==================== 路由 (2维) ====================
            idx_base = idx_base + self.TLS_DIM  # 101
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
                    
                    if heading_diff > 180:
                        heading_diff -= 360
                    elif heading_diff < -180:
                        heading_diff += 360
                    
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
                    
                    ped_vx = ped_speed * np.sin(np.radians(ped_angle))
                    ped_vy = ped_speed * np.cos(np.radians(ped_angle))
                    
                    info = np.array([
                        rel_pos[0] / 30.0,
                        rel_pos[1] / 30.0,
                        ped_vx / 5.0,
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
    
    # ========== 修复：改进的红绿灯状态获取（5维版本） ==========
    def _get_traffic_light_state(self) -> np.ndarray:
        """
        改进版红绿灯状态获取（5维）
        返回: [归一化距离, 红灯, 黄灯, 绿灯, 剩余时间]
        
        修复:
        - 距离范围扩大到200米，使用分段归一化
        - 添加红绿灯剩余时间信息
        - 更精确的状态检测
        """
        # 5维: 距离, 红, 黄, 绿, 剩余时间
        state = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        self._ensure_connection()
        
        try:
            if self.ego_id not in traci.vehicle.getIDList():
                return state
            
            tls_list = traci.vehicle.getNextTLS(self.ego_id)
            
            if tls_list:
                tls_id, tls_index, distance, link_state = tls_list[0]
                
                # 修复：扩大距离范围到200米，使用分段归一化
                # 这样agent可以更好地感知远处的红绿灯
                if distance <= 50:
                    state[0] = distance / 50.0 * 0.25  # 0-50米 -> 0-0.25
                elif distance <= 100:
                    state[0] = 0.25 + (distance - 50) / 50.0 * 0.25  # 50-100米 -> 0.25-0.5
                elif distance <= 200:
                    state[0] = 0.5 + (distance - 100) / 100.0 * 0.5  # 100-200米 -> 0.5-1.0
                else:
                    state[0] = 1.0
                
                # 状态编码
                if link_state in ['r', 'R']:
                    state[1] = 1.0  # 红灯
                elif link_state in ['y', 'Y']:
                    state[2] = 1.0  # 黄灯
                elif link_state in ['g', 'G', 'o', 'O']:  # 包括off状态（可通行）
                    state[3] = 1.0  # 绿灯
                
                # 获取剩余时间
                try:
                    remaining = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
                    state[4] = min(max(remaining / 30.0, 0.0), 1.0)  # 归一化到0-30秒
                except:
                    state[4] = 0.5  # 默认值
        
        except Exception as e:
            pass
        
        return state
    # ================================================
    
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
    
    # ========== 修复：改进的奖励函数 ==========
    def _compute_reward(self) -> float:
        """完整的修复版奖励函数"""
        reward = 0.0
        
        self._ensure_connection()
        
        if self.ego_id not in traci.vehicle.getIDList():
            return -10.0
        
        try:
            # 终止奖励
            if self.goal_reached:
                return 200.0
            
            if self.collision_occurred:
                return -100.0  # 增加碰撞惩罚
            
            # ==================== 距离奖励（大幅提高）====================
            current_distance = self._get_distance_to_goal()
            distance_reward = (self.last_distance_to_goal - current_distance) * 1.0  # 0.15 -> 1.0
            reward += distance_reward
            self.last_distance_to_goal = current_distance
            
            # ==================== 速度奖励 ====================
            speed = traci.vehicle.getSpeed(self.ego_id)
            
            # 获取红绿灯状态来决定最优速度
            tls_state = self._get_traffic_light_state()
            is_red = tls_state[1] > 0.5
            
            if not is_red:
                # 非红灯时鼓励保持速度
                optimal_speed = 10.0
                speed_diff = abs(speed - optimal_speed)
                if speed_diff < 2.0:
                    reward += 0.1  # 增加速度奖励
                elif speed_diff < 5.0:
                    reward += 0.05
            
            # ==================== 舒适度奖励 ====================
            accel = traci.vehicle.getAcceleration(self.ego_id)
            
            if accel < -3.0:
                reward -= 0.2  # 降低急刹车惩罚（红灯前需要刹车）
                self.stats["harsh_braking_count"] += 1
            
            if accel > 2.5:
                reward -= 0.2
            
            heading = traci.vehicle.getAngle(self.ego_id)
            heading_diff = abs(heading - self.last_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff
            if heading_diff > 15:
                reward -= 0.3
                self.stats["harsh_steering_count"] += 1
            
            self.last_accel = accel
            
            # ==================== 红绿灯奖励 (修复版) ====================
            if self.stage >= 2:
                tls_reward = self._compute_traffic_light_reward(speed, tls_state)
                reward += tls_reward
            
            # ==================== 避障奖励 (Stage 3+) ====================
            if self.stage >= 3:
                nearby_vehicles = self._get_nearby_vehicles(max_count=1)
                if nearby_vehicles:
                    try:
                        front_veh_id = nearby_vehicles[0]
                        front_pos = np.array(traci.vehicle.getPosition(front_veh_id))
                        ego_pos = np.array(traci.vehicle.getPosition(self.ego_id))
                        distance = np.linalg.norm(front_pos - ego_pos)
                        
                        # 检测是否在红绿灯附近（放宽车距要求）
                        is_near_traffic_light = False
                        tls_distance = 999
                        if self.stage >= 2:
                            # 反归一化红绿灯距离
                            normalized_distance = tls_state[0]
                            is_red = tls_state[1] > 0.5
                            is_yellow = tls_state[2] > 0.5
                            
                            if normalized_distance <= 0.25:
                                tls_distance = normalized_distance / 0.25 * 50
                            elif normalized_distance <= 0.5:
                                tls_distance = 50 + (normalized_distance - 0.25) / 0.25 * 50
                            else:
                                tls_distance = 100 + (normalized_distance - 0.5) / 0.5 * 100
                            
                            # 在红绿灯50米内且是红灯/黄灯时，认为是排队等待
                            if tls_distance < 50 and (is_red or is_yellow):
                                is_near_traffic_light = True
                        
                        # 根据是否在红绿灯附近使用不同的车距标准
                        if is_near_traffic_light:
                            # 红绿灯附近：放宽标准（允许更近的跟车）
                            if distance < 2.0:
                                reward -= 10.0  # 极度危险
                            elif distance < 3.0:
                                reward -= 3.0  # 太近但可容忍
                            elif distance < 5.0:
                                reward -= 0.5  # 略近
                            elif distance < 15.0:
                                reward += 0.5  # 合理排队距离
                        else:
                            # 正常路段：标准车距要求
                            if distance < 3.0:
                                reward -= 10.0  # 极度危险
                            elif distance < 5.0:
                                reward -= 5.0  # 太近
                            elif distance < 10.0:
                                reward -= 1.0  # 偏近
                            elif distance < 20.0:
                                reward += 0.5  # 保持安全距离
                        # distance >= 20m: 无额外奖惩
                    except:
                        pass
            
            # ==================== 行人避让奖励 (Stage 4+) ====================
            if self.stage >= 4:
                nearby_peds = self._get_nearby_pedestrians_detailed(max_count=1)
                if nearby_peds:
                    ped_info = nearby_peds[0]
                    ped_distance = np.sqrt(ped_info[0]**2 + ped_info[1]**2) * 30
                    
                    if ped_distance < 10:
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
                reward += 0.01
            
            # 时间惩罚（降低，因为等红灯是必要的）
            reward -= 0.1
        
        except Exception as e:
            reward = 0.0
        
        return reward
    
    # ========== 修复：新增的红绿灯奖励计算函数 ==========
    def _compute_traffic_light_reward(self, speed: float, tls_state: np.ndarray) -> float:
        """
        修复版红绿灯奖励计算 V4 - 最终版
        
        核心改进:
        1. 降低接近红灯的减速奖励（避免停着不动赚分）
        2. 降低停车奖励（仅补偿必要等待）
        3. 大幅提高闯红灯惩罚（-600）
        4. 增加非红灯时静止惩罚
        """
        reward = 0.0
        
        # 解析红绿灯状态
        normalized_distance = tls_state[0]
        is_red = tls_state[1] > 0.5
        is_yellow = tls_state[2] > 0.5
        is_green = tls_state[3] > 0.5
        
        # 反归一化距离
        if normalized_distance <= 0.25:
            distance = normalized_distance / 0.25 * 50
        elif normalized_distance <= 0.5:
            distance = 50 + (normalized_distance - 0.25) / 0.25 * 50
        else:
            distance = 100 + (normalized_distance - 0.5) / 0.5 * 100
        
        # 获取当前红绿灯ID
        try:
            tls_list = traci.vehicle.getNextTLS(self.ego_id)
            current_tls = tls_list[0][0] if tls_list else None
        except:
            current_tls = None
        
        # ==================== 红灯/黄灯处理 ====================
        if is_red or is_yellow:
            
            # 开始接近红灯时记录
            if not self.approaching_red_light and distance < 150:
                self.approaching_red_light = True
                self.red_light_distance_when_detected = distance
                self.current_tls_id = current_tls
            
            # 接近红灯时的减速奖励（降低系数）
            if distance < 150 and distance >= 5:
                # 计算理想速度曲线
                if distance < 20:
                    target_speed = 2.0
                elif distance < 50:
                    target_speed = 5.0
                elif distance < 100:
                    target_speed = 8.0
                else:
                    target_speed = 10.0
                
                # 速度符合预期 -> 小奖励
                if speed <= target_speed + 1:
                    reward += 0.3 * (1 - distance / 150)  # 降低系数
                else:
                    # 超速 -> 惩罚
                    overspeed = speed - target_speed
                    penalty_factor = 1 + (1 - distance / 150) * 3
                    reward -= overspeed * 0.5 * penalty_factor
            
            # 红灯前5米内的特殊处理
            if distance < 5:
                if speed < 0.5:
                    reward += 0.2  # 降低停车奖励
                elif speed < 2.0:
                    reward += 0.1
                else:
                    # 闯红灯检测
                    if current_tls and current_tls not in self.passed_traffic_lights:
                        reward -= 200.0  # 降低惩罚，让模型优先避免超时
                        self.stats["red_light_violations"] += 1
                        self.passed_traffic_lights.add(current_tls)
        
        # ==================== 绿灯处理 ====================
        elif is_green:
            self.approaching_red_light = False
            self.current_tls_id = None
            self._red_light_punished = False
            
            if distance < 50:
                # 绿灯时应该正常通过
                if speed < 1.0:
                    reward -= 15.0  # 绿灯不走严重惩罚
                elif speed < 3.0:
                    reward -= 2.0  # 降低慢速惩罚（允许谨慎）
                elif speed < 5.0:
                    reward += 1.0  # 中速奖励
                else:
                    reward += 2.0  # 高速通过奖励
        
        # ==================== 无红绿灯/远离红绿灯 ====================
        else:
            self.approaching_red_light = False
            # 正常行驶，如果停着不动则严厉惩罚
            if speed < 0.5:
                reward -= 5.0  # 增加静止惩罚
            elif speed < 2.0:
                reward -= 0.5  # 低速惩罚
        
        return reward
    # ====================================================
    
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
        
        # 检测连续静止（非红灯/黄灯时）
        try:
            speed = traci.vehicle.getSpeed(self.ego_id)
            tls_state = self._get_traffic_light_state()
            is_red = tls_state[1] > 0.5
            is_yellow = tls_state[2] > 0.5
            
            # 只有在非红灯/黄灯时才计算静止步数
            if speed < 0.5 and not is_red and not is_yellow:
                self.stationary_steps += 1
            else:
                self.stationary_steps = 0
            
            # 连续静止150步（15秒，非红灯）→ 终止
            if self.stationary_steps >= 150:  # 增加到150步
                self.stats["stationary_timeout"] = True
                return True
        except:
            pass
        
        return False
    
    def _get_info(self) -> Dict:
        # 计算平均背景车数量
        avg_bg_vehicles = np.mean(self.bg_vehicle_counts) if self.bg_vehicle_counts else 0.0
        
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
            "avg_bg_vehicles": avg_bg_vehicles,
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
            "min_route_length": 600.0, "max_route_length": 1200.0},
        3: {"num_background_vehicles": 8, "num_pedestrians": 0, "max_episode_steps": 1500,
            "min_route_length": 600.0, "max_route_length": 1200.0},
        4: {"num_background_vehicles": 12, "num_pedestrians": 5, "max_episode_steps": 2000,
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
    print("SUMO自动驾驶环境 V2.3 (背景车辆分布优化版)")
    print("=" * 60)
    
    if 'SUMO_HOME' not in os.environ:
        print("请设置环境变量 SUMO_HOME")
        sys.exit(1)
    
    print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")
    print("\n📐 观测空间: 103维")
    print("   - 自车状态: 8维")
    print("   - 周围车辆: 12辆 × 6维 = 72维")
    print("   - 行人: 4个 × 4维 = 16维")
    print("   - 红绿灯: 5维 (含剩余时间)")
    print("   - 路由: 2维")
    print("\n🔧 V2.3 优化内容:")
    print("   - 扩大生成范围：2跳邻居边")
    print("   - 每条边最多3辆车")
    print("   - 70%在其他路线，30%在ego路线")
    print("\n动态背景车: 50-150m生成, >200m消失")
    print("动态行人: 30-80m生成, >100m消失")
    print("\n使用前请先下载地图:")
    print("   python scripts/download_map.py --region sf_mission")
