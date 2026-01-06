"""
性能分析脚本 - 找出SUMO环境的瓶颈
"""
import sys
import os
import time

# 添加项目路径
sys.path.insert(0, r"F:\study\self_drive_proj1\autonomous-driving-ppo")

from envs.sumo_env import make_sumo_env
import numpy as np

def profile_env(stage=3, n_steps=500):
    print(f"创建Stage {stage}环境...")
    env = make_sumo_env(stage=stage, map_name="sf_mission", use_gui=False)
    
    # 计时器
    timings = {
        'reset': [],
        'step_total': [],
        'action_sample': [],
    }
    
    print(f"开始性能测试 ({n_steps} steps)...\n")
    
    # Reset计时
    t0 = time.perf_counter()
    obs, info = env.reset()
    timings['reset'].append(time.perf_counter() - t0)
    print(f"Reset耗时: {timings['reset'][0]*1000:.1f}ms")
    
    # Step计时
    for i in range(n_steps):
        t0 = time.perf_counter()
        action = env.action_space.sample()
        timings['action_sample'].append(time.perf_counter() - t0)
        
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        timings['step_total'].append(time.perf_counter() - t0)
        
        if terminated or truncated:
            t0 = time.perf_counter()
            obs, info = env.reset()
            timings['reset'].append(time.perf_counter() - t0)
    
    env.close()
    
    # 输出结果
    print("\n" + "="*50)
    print("性能分析结果")
    print("="*50)
    
    avg_step = np.mean(timings['step_total']) * 1000
    avg_reset = np.mean(timings['reset']) * 1000
    
    print(f"Step平均耗时: {avg_step:.2f}ms")
    print(f"Reset平均耗时: {avg_reset:.2f}ms (共{len(timings['reset'])}次)")
    print(f"理论最大速度: {1000/avg_step:.0f} it/s (不含reset)")
    
    # 估算实际速度（假设每episode 300步）
    episode_len = 300
    episode_time = episode_len * avg_step + avg_reset
    effective_speed = episode_len / (episode_time / 1000)
    print(f"估算实际速度: {effective_speed:.0f} it/s (含reset)")
    
    print("\n如果速度远低于预期，请检查:")
    print("1. 是否有其他程序占用CPU")
    print("2. 杀毒软件是否扫描SUMO进程")
    print("3. 是否有多个SUMO实例在后台运行")

if __name__ == "__main__":
    profile_env(stage=3, n_steps=500)
