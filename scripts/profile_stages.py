"""
对比不同Stage的性能
"""
import sys
sys.path.insert(0, r"F:\study\self_drive_proj1\autonomous-driving-ppo")

from envs.sumo_env import make_sumo_env
import time
import numpy as np

for stage in [1, 2, 3]:
    print(f"\n{'='*50}")
    print(f"Stage {stage} 性能测试")
    print(f"{'='*50}")
    
    env = make_sumo_env(stage=stage, map_name="sf_mission", use_gui=False)
    
    obs, _ = env.reset()
    
    step_times = []
    for i in range(200):
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        step_times.append(time.perf_counter() - t0)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    avg_ms = np.mean(step_times) * 1000
    print(f"背景车数量: {env.num_background_vehicles}")
    print(f"Step平均耗时: {avg_ms:.2f}ms")
    print(f"理论速度: {1000/avg_ms:.0f} it/s")
