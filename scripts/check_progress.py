#!/usr/bin/env python3
"""
查看当前训练进度
"""

import numpy as np
import os
from pathlib import Path

def check_progress(stage=1):
    eval_file = f"../outputs/logs/stage{stage}/evaluations.npz'
    
    if not os.path.exists(eval_file):
        print(f"⚠️ Stage {stage} 还没有评估数据")
        print(f"   文件不存在: {eval_file}")
        print(f"   请等待训练超过5000步")
        return
    
    data = np.load(eval_file)
    timesteps = data['timesteps']
    results = data['results']
    
    print("="*60)
    print(f"📊 Stage {stage} 训练进度")
    print("="*60)
    
    print(f"\n已完成训练步数: {timesteps[-1]:,}")
    print(f"评估次数: {len(timesteps)}")
    print(f"评估频率: 每5000步")
    
    print("\n📈 Reward变化:")
    print("-"*60)
    print(f"{'步数':<12} {'平均Reward':<15} {'标准差':<10} {'趋势'}")
    print("-"*60)
    
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    # 显示最近10次评估
    start_idx = max(0, len(timesteps) - 10)
    for i in range(start_idx, len(timesteps)):
        trend = ""
        if i > 0:
            diff = mean_rewards[i] - mean_rewards[i-1]
            if diff > 0.5:
                trend = "📈 ↑↑"
            elif diff > 0:
                trend = "↗ ↑"
            elif diff < -0.5:
                trend = "📉 ↓↓"
            elif diff < 0:
                trend = "↘ ↓"
            else:
                trend = "→"
        
        print(f"{timesteps[i]:<12,} {mean_rewards[i]:<15.2f} {std_rewards[i]:<10.2f} {trend}")
    
    print("-"*60)
    print(f"\n💡 总结:")
    print(f"   初始Reward: {mean_rewards[0]:.2f}")
    print(f"   当前Reward: {mean_rewards[-1]:.2f}")
    print(f"   提升幅度: {mean_rewards[-1] - mean_rewards[0]:+.2f}")
    print(f"   最佳Reward: {np.max(mean_rewards):.2f}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='查看训练进度')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    
    check_progress(args.stage)
