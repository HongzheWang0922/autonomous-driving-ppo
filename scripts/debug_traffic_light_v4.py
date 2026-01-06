#!/usr/bin/env python3
"""
红绿灯调试脚本 V4 - 验证绿灯不起步的真正原因
新增：isStopped、getStopState、getLeader、实际执行的加速度
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "envs").exists() else SCRIPT_DIR
for path in [REPO_DIR, SCRIPT_DIR, SCRIPT_DIR / ".."]:
    if (path / "envs").exists():
        sys.path.insert(0, str(path))
        break

import traci
from envs.sumo_env import make_sumo_env


def format_tls(state):
    if state is None: return "---"
    s = state.lower()
    return "🔴" if s == 'r' else ("🟡" if s == 'y' else ("🟢" if s == 'g' else f"?{state}"))


def decode_stop_state(state):
    """解码SUMO的stopState位掩码"""
    if state == 0:
        return "正常"
    flags = []
    if state & 1: flags.append("stopped")
    if state & 2: flags.append("parking")
    if state & 4: flags.append("triggered")
    if state & 8: flags.append("containerTriggered")
    if state & 16: flags.append("atBusStop")
    if state & 32: flags.append("atContainerStop")
    if state & 64: flags.append("atChargingStation")
    if state & 128: flags.append("atParkingArea")
    return "|".join(flags) if flags else f"unknown({state})"


def main():
    print("=" * 90)
    print("调试V4 - 验证绿灯不起步原因")
    print("新增: isStopped | stopState | leader | 实际加速度")
    print("=" * 90)
    
    env = make_sumo_env(stage=2, use_gui=True)
    obs, info = env.reset()
    done = False
    step = 0
    
    print(f"路线: {info.get('route_length', 0):.0f}m | 红绿灯: {info.get('route_traffic_lights', 0)}")
    print("-" * 90)
    
    # 状态追踪
    last_in_junction = False
    last_tls_state = None
    stuck_counter = 0
    last_speed = 0
    
    try:
        while not done and step < 800:
            # 获取数据
            speed = traci.vehicle.getSpeed("ego")
            accel = traci.vehicle.getAcceleration("ego")
            road_id = traci.vehicle.getRoadID("ego")
            in_junction = road_id.startswith(':')
            allowed_speed = traci.vehicle.getAllowedSpeed("ego")
            
            # ========== 新增检查 ==========
            is_stopped = traci.vehicle.isStopped("ego")
            stop_state = traci.vehicle.getStopState("ego")
            
            # 前方车辆
            leader_info = traci.vehicle.getLeader("ego", 50)
            leader_str = f"{leader_info[0]}@{leader_info[1]:.1f}m" if leader_info else "无"
            
            # 红绿灯
            tls_list = traci.vehicle.getNextTLS("ego")
            if tls_list:
                tls_dist = tls_list[0][2]
                tls_state = tls_list[0][3]
            else:
                tls_dist = None
                tls_state = None
            
            # 规则动作
            is_red = obs[97] > 0.5
            obs_dist = obs[96] * 200
            action_accel = -1.0 if (is_red and obs_dist < 50) else 0.6
            action = np.array([action_accel, 0.0], dtype=np.float32)
            
            # 检测关键事件
            events = []
            
            # 事件1: 进入/离开junction
            if in_junction and not last_in_junction:
                events.append("进入JUNC")
            if not in_junction and last_in_junction:
                events.append("离开JUNC")
            
            # 事件2: 灯变化
            if tls_state != last_tls_state and tls_dist and tls_dist < 150:
                events.append(f"灯变:{format_tls(last_tls_state)}→{format_tls(tls_state)}")
            
            # 事件3: 发出加速但车不动（关键！）
            if action_accel > 0 and speed < 0.3:
                stuck_counter += 1
                if stuck_counter == 1 or stuck_counter % 50 == 0:
                    # 详细诊断
                    stop_str = decode_stop_state(stop_state)
                    events.append(f"⚠️不动! isStopped={is_stopped} stopState={stop_str} leader={leader_str}")
            else:
                stuck_counter = 0
            
            # 事件4: 闯红灯
            if tls_dist and tls_dist < 10 and speed > 2 and tls_state in ['r', 'R']:
                events.append("⚠️闯红灯!")
            
            # 事件5: 高速
            if speed > 15:
                events.append(f"高速v={speed:.0f}")
            
            # 事件6: 异常减速（可能是SUMO强制干预）
            if last_speed - speed > 10:  # 1步内速度降10m/s以上
                events.append(f"⚠️异常减速! {last_speed:.0f}→{speed:.0f}")
            
            # 只在有事件或每100步时输出
            if events or step % 100 == 0:
                pos_str = "JUNC" if in_junction else road_id[:15]
                dist_str = f"{tls_dist:.0f}m" if tls_dist else "---"
                event_str = " | ".join(events) if events else ""
                
                # 简化输出
                print(f"{step:>4} | {speed:>5.1f}m/s | 实际加速:{accel:>5.1f} | {format_tls(tls_state)} {dist_str:>5} | {event_str}")
            
            # 更新状态
            last_in_junction = in_junction
            last_tls_state = tls_state
            last_speed = speed
            
            # 执行
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            time.sleep(0.03)
        
        print("-" * 90)
        result = "✅成功" if info.get('goal_reached') else "❌失败"
        print(f"结果: {result} | 步数: {step} | 闯红灯: {info.get('red_light_violations', 0)}")
        input("按Enter结束...")
        
    finally:
        env.close()


if __name__ == "__main__":
    if 'SUMO_HOME' not in os.environ:
        print("错误: 未设置 SUMO_HOME")
        sys.exit(1)
    main()
