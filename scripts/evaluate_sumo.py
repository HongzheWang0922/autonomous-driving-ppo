#!/usr/bin/env python3
"""
SUMO环境评估脚本
评估训练好的模型在各个阶段的表现

Usage:
    python evaluate_sumo.py --stage 1 --n-episodes 100
    python evaluate_sumo.py --stage 2 --model path/to/model.zip --render
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# 添加项目路径
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

from stable_baselines3 import PPO
from envs.sumo_env import make_sumo_env


def evaluate_model(
    model_path: str,
    stage: int,
    map_name: str,
    n_episodes: int = 100,
    render: bool = False,
    deterministic: bool = True,
) -> Dict:
    """
    评估模型
    
    Args:
        model_path: 模型路径
        stage: 训练阶段
        map_name: 地图名称
        n_episodes: 评估episode数
        render: 是否渲染（使用SUMO-GUI）
        deterministic: 是否使用确定性策略
    
    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估 Stage {stage} 模型")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"地图: {map_name}")
    print(f"Episodes: {n_episodes}")
    print(f"渲染: {render}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("📥 加载模型...")
    model = PPO.load(model_path)
    
    # 创建环境
    print("📦 创建环境...")
    env = make_sumo_env(
        stage=stage,
        map_name=map_name,
        use_gui=render,
        seed=42
    )
    
    # 评估
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    red_light_violations = []
    route_progresses = []
    
    print(f"\n开始评估 {n_episodes} 个episodes...\n")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1
        
        # 记录统计
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        if info.get("goal_reached", False):
            success_count += 1
        
        if info.get("collision", False):
            collision_count += 1
        
        if "red_light_violations" in info:
            red_light_violations.append(info["red_light_violations"])
        
        route_progresses.append(info.get("route_progress", 0.0))
        
        # 打印进度
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"Reward={ep_reward:.2f}, "
                  f"Length={ep_length}, "
                  f"Success={info.get('goal_reached', False)}, "
                  f"Collision={info.get('collision', False)}")
    
    # 计算统计
    results = {
        "stage": stage,
        "n_episodes": n_episodes,
        "success_rate": (success_count / n_episodes) * 100,
        "collision_rate": (collision_count / n_episodes) * 100,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_progress": np.mean(route_progresses) * 100,
    }
    
    if red_light_violations:
        results["mean_red_light_violations"] = np.mean(red_light_violations)
        results["total_red_light_violations"] = sum(red_light_violations)
    
    # 关闭环境
    env.close()
    
    return results


def print_results(results: Dict):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"📊 评估结果 - Stage {results['stage']}")
    print(f"{'='*60}\n")
    
    print(f"成功率指标:")
    print(f"  ✓ 成功率 (到达终点): {results['success_rate']:.2f}%")
    print(f"  ✗ 碰撞率: {results['collision_rate']:.2f}%")
    print(f"  📍 平均路由完成度: {results['mean_progress']:.2f}%")
    
    print(f"\n奖励指标:")
    print(f"  平均 Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Reward 范围: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    
    print(f"\nEpisode长度:")
    print(f"  平均: {results['mean_length']:.1f} ± {results['std_length']:.1f} 步")
    
    if "mean_red_light_violations" in results:
        print(f"\n红绿灯遵守:")
        print(f"  平均每episode闯红灯: {results['mean_red_light_violations']:.2f} 次")
        print(f"  总闯红灯次数: {results['total_red_light_violations']}")
    
    print(f"\n{'='*60}")
    
    # 判断是否可以进入下一阶段
    if results['success_rate'] >= 80.0:
        print(f"🎉 成功率 >= 80%，可以进入下一阶段！")
        if results['stage'] < 4:
            print(f"   运行: python scripts/train_sumo.py --stage {results['stage'] + 1}")
        else:
            print(f"   🏆 恭喜！已完成所有训练阶段！")
    else:
        print(f"⚠️  成功率 < 80%，建议继续训练当前阶段")
        print(f"   或调整超参数/奖励函数")
    
    print(f"{'='*60}\n")


def save_results(results: Dict, output_file: str):
    """保存评估结果到JSON文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='评估SUMO训练模型')
    
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4],
                        help='评估阶段')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径 (默认: 使用最新的final模型)')
    parser.add_argument('--map', type=str, default='sf_mission',
                        help='地图名称')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='评估episode数')
    parser.add_argument('--render', action='store_true',
                        help='使用SUMO-GUI渲染')
    parser.add_argument('--stochastic', action='store_true',
                        help='使用随机策略（默认使用确定性策略）')
    parser.add_argument('--output', type=str, default=None,
                        help='结果保存路径 (JSON格式)')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.model is None:
        model_path = REPO_DIR / f"outputs/models/best_stage{args.stage}/ppo_final.zip"
        if not model_path.exists():
            print(f"❌ 找不到模型: {model_path}")
            print(f"   请指定模型路径: --model path/to/model.zip")
            sys.exit(1)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ 找不到模型: {model_path}")
            sys.exit(1)
    
    # 检查SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        print("❌ 错误: 未设置环境变量 SUMO_HOME")
        sys.exit(1)
    
    # 检查地图文件
    map_file = REPO_DIR / "maps" / f"{args.map}.net.xml"
    if not map_file.exists():
        print(f"❌ 找不到地图文件: {map_file}")
        print(f"   请先运行: python scripts/download_map.py --region {args.map}")
        sys.exit(1)
    
    # 评估
    results = evaluate_model(
        model_path=str(model_path),
        stage=args.stage,
        map_name=args.map,
        n_episodes=args.n_episodes,
        render=args.render,
        deterministic=not args.stochastic,
    )
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if args.output:
        save_results(results, args.output)
    else:
        # 默认保存位置
        default_output = REPO_DIR / f"outputs/eval_stage{args.stage}_results.json"
        save_results(results, str(default_output))


if __name__ == "__main__":
    main()

