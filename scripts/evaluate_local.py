#!/usr/bin/env python3
"""
模型评估脚本 - OpenMP修复版
使用方法: python evaluate_fixed.py --stage 1
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 修复OpenMP冲突

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加Git仓库到Python路径
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

from envs.intersection_env import IntersectionEnvWrapper
from stable_baselines3 import PPO


def evaluate_model(model_path, n_episodes=50, difficulty='easy'):
    """评估模型性能"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_path}")
    print(f"难度: {difficulty}, Episodes: {n_episodes}")
    print(f"{'='*60}\n")
    
    model = PPO.load(model_path)
    env = IntersectionEnvWrapper(difficulty=difficulty)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_length = 0
        done = False
        
        while not done and ep_length < 600:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            
            if done or truncated:
                if info.get('crashed', False):
                    collision_count += 1
                
                if 'rewards' in info and info['rewards'].get('arrived_reward', 0) > 0:
                    success_count += 1
                
                if truncated and not info.get('crashed', False):
                    if 'rewards' not in info or info['rewards'].get('arrived_reward', 0) == 0:
                        timeout_count += 1
                
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep+1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    env.close()
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes * 100,
        'collision_rate': collision_count / n_episodes * 100,
        'timeout_rate': timeout_count / n_episodes * 100,
        'rewards': episode_rewards,
        'lengths': episode_lengths
    }
    
    return results


def plot_evaluation_results(results, stage, save_path="../outputs/figures'):
    """绘制评估结果"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{stage} Evaluation Results', fontsize=16, fontweight='bold')
    
    # 图1: Reward分布
    axes[0].hist(results['rewards'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(results['mean_reward'], color='red', linestyle='--', 
                    linewidth=2, label=f"Mean: {results['mean_reward']:.2f}")
    axes[0].set_xlabel('Episode Reward', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Reward Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 图2: Episode长度分布
    axes[1].hist(results['lengths'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(results['mean_length'], color='red', linestyle='--', 
                    linewidth=2, label=f"Mean: {results['mean_length']:.1f}")
    axes[1].set_xlabel('Episode Length (steps)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Episode Length Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 图3: 性能指标
    metrics = ['Success', 'Collision', 'Timeout']
    values = [results['success_rate'], results['collision_rate'], results['timeout_rate']]
    colors = ['green', 'red', 'orange']
    
    bars = axes[2].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Percentage (%)', fontsize=12)
    axes[2].set_title('Performance Metrics', fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{value:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    filename = f'{save_path}/{stage.replace(" ", "_")}_evaluation.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {filename}")
    
    plt.show()


def print_results(results, stage):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"📊 {stage} 评估结果:")
    print(f"{'='*60}")
    print(f"平均Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"平均长度: {results['mean_length']:.1f} steps")
    print(f"成功率: {results['success_rate']:.1f}%")
    print(f"碰撞率: {results['collision_rate']:.1f}%")
    print(f"超时率: {results['timeout_rate']:.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='评估RL模型')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='评估阶段: 1/2/3')
    parser.add_argument('--n-episodes', type=int, default=50,
                        help='测试episodes数量 (默认: 50)')
    parser.add_argument('--no-plot', action='store_true',
                        help='不显示图表')
    
    args = parser.parse_args()
    
    # 设置模型路径和难度
    stage_config = {
        1: ('ppo_stage1_final', 'easy', 'Stage 1'),
        2: ('ppo_stage2_final', 'medium', 'Stage 2'),
        3: ('ppo_final', 'hard', 'Stage 3')
    }
    
    model_name, difficulty, stage_name = stage_config[args.stage]
    model_path = f"../outputs/models/{model_name}'
    
    # 评估
    results = evaluate_model(model_path, n_episodes=args.n_episodes, difficulty=difficulty)
    
    # 打印结果
    print_results(results, stage_name)
    
    # 绘图
    if not args.no_plot:
        plot_evaluation_results(results, stage_name)


if __name__ == "__main__":
    main()
