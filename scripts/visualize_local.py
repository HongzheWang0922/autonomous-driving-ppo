#!/usr/bin/env python3
"""
训练过程可视化脚本
使用方法: python visualize_local.py --stage 1
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(stage=1, save_path="../outputs/figures'):
    """从evaluations.npz绘制训练曲线"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    log_path = f"../outputs/logs/stage{stage}/'
    eval_file = f'{log_path}evaluations.npz'
    stage_name = f'Stage {stage}'
    
    fig = plt.figure(figsize=(18, 10))
    
    if not os.path.exists(eval_file):
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'⚠️ 未找到评估数据\n\n'
                f'路径: {eval_file}\n\n'
                f'请确保训练时使用了EvalCallback',
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        plt.suptitle(f'{stage_name} Training Analysis', fontsize=18, fontweight='bold')
        filename = f'{save_path}/{stage_name.replace(" ", "_")}_reward_curves.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"⚠️ 未找到评估数据: {eval_file}")
        return
    
    # 读取评估数据
    data = np.load(eval_file)
    timesteps = data['timesteps']
    results = data['results']  # shape: (n_eval, n_episodes)
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    # 创建4个子图
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Reward曲线占上面整行
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 图1: Reward曲线
    ax1.plot(timesteps, mean_rewards, linewidth=2.5, color='#2ecc71', label='Mean Reward')
    ax1.fill_between(timesteps,
                    mean_rewards - std_rewards,
                    mean_rewards + std_rewards,
                    alpha=0.3, color='#2ecc71', label='Std Dev')
    ax1.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    ax1.set_title(f'{stage_name} - Reward Curve', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 添加趋势信息
    if len(mean_rewards) > 1:
        improvement = mean_rewards[-1] - mean_rewards[0]
        color = 'green' if improvement > 0 else 'red'
        arrow = '↑' if improvement > 0 else '↓'
        ax1.text(0.02, 0.98, f'Total Change: {arrow} {improvement:.2f}',
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', color=color,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 图2: Reward范围
    min_rewards = np.min(results, axis=1)
    max_rewards = np.max(results, axis=1)
    
    ax2.plot(timesteps, mean_rewards, linewidth=2, label='Mean', color='blue')
    ax2.plot(timesteps, max_rewards, linewidth=1.5, label='Max', color='green', linestyle='--')
    ax2.plot(timesteps, min_rewards, linewidth=1.5, label='Min', color='red', linestyle='--')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Range', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 图3: 统计摘要
    ax3.axis('off')
    final_mean = mean_rewards[-1]
    final_std = std_rewards[-1]
    best_mean = np.max(mean_rewards)
    
    stats_text = f"""
{stage_name} Training Summary

📊 Final Performance:
   Mean Reward: {final_mean:.2f} ± {final_std:.2f}

📈 Best Performance:
   Max Mean Reward: {best_mean:.2f}
   (at step {timesteps[np.argmax(mean_rewards)]:,})

📉 Initial Performance:
   Starting Reward: {mean_rewards[0]:.2f}

🎯 Total Improvement:
   {improvement:+.2f}

📝 Evaluations: {len(timesteps)}
   (every 5,000 steps)
    """
    
    ax3.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle(f'{stage_name} Training Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # 保存
    filename = f'{save_path}/{stage_name.replace(" ", "_")}_reward_curves.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Reward曲线已保存: {filename}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='可视化训练过程')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='可视化阶段: 1/2/3')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"📊 生成Stage {args.stage}可视化（训练曲线）")
    print("="*60)
    
    plot_training_curves(stage=args.stage)
    
    print("\n✓ 可视化完成！")


if __name__ == "__main__":
    main()
