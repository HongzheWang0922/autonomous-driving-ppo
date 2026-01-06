#!/usr/bin/env python3
"""
规则策略测试 - 验证 Stage 2 任务是否可行
用简单规则（绿灯走、红灯停）测试，看能否在1500步内完成

Usage:
    python test_rule_based.py                    # 带GUI，5个episode
    python test_rule_based.py --no-gui -n 20    # 无GUI，20个episode
    python test_rule_based.py --delay 0.2       # 慢放，每步0.2秒
"""

import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "envs").exists() else SCRIPT_DIR

# 尝试多个可能的路径
for path in [REPO_DIR, SCRIPT_DIR, SCRIPT_DIR / ".."]:
    if (path / "envs").exists():
        sys.path.insert(0, str(path))
        break

from envs.sumo_env import make_sumo_env


def rule_based_action(obs, verbose=False):
    """
    简单规则策略：
    - 观测空间第96-100维是红绿灯状态
    - [距离(归一化), 红, 黄, 绿, 剩余时间]
    """
    # 提取红绿灯信息
    tls_distance_norm = obs[96]  # 归一化距离
    is_red = obs[97] > 0.5
    is_yellow = obs[98] > 0.5
    is_green = obs[99] > 0.5
    
    # 提取自车速度 (obs[0] 是归一化速度)
    speed_norm = obs[0]
    speed = speed_norm * 15.0  # 反归一化，假设最大速度15m/s
    
    # 反归一化红绿灯距离
    if tls_distance_norm <= 0.25:
        distance = tls_distance_norm / 0.25 * 50
    elif tls_distance_norm <= 0.5:
        distance = 50 + (tls_distance_norm - 0.25) / 0.25 * 50
    else:
        distance = 100 + (tls_distance_norm - 0.5) / 0.5 * 100
    
    # 规则决策
    if is_red or is_yellow:
        if distance < 10:
            # 很近了，强力刹车
            accel = -1.0
        elif distance < 30:
            # 中等距离，中等刹车
            accel = -0.6
        elif distance < 60:
            # 较远，轻微减速
            accel = -0.3
        else:
            # 远处，正常行驶
            accel = 0.5
    else:
        # 绿灯或无红绿灯，正常加速
        if speed < 8:
            accel = 0.8  # 加速到目标速度
        else:
            accel = 0.3  # 保持速度
    
    steer = 0.0  # 不主动转向，依赖SUMO的车道保持
    
    if verbose:
        light = "🔴红" if is_red else ("🟡黄" if is_yellow else ("🟢绿" if is_green else "⚫无"))
        print(f"    {light} 距离:{distance:.0f}m 速度:{speed:.1f}m/s → 加速:{accel:.1f}")
    
    return np.array([accel, steer], dtype=np.float32)


def run_test(n_episodes=5, use_gui=True, delay=0.05, verbose=False):
    """运行规则策略测试"""
    
    print("=" * 60)
    print("规则策略测试 - 验证 Stage 2 任务可行性")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"GUI: {'开启' if use_gui else '关闭'}")
    print(f"延时: {delay}秒/步")
    print("=" * 60)
    print()
    print("规则策略：")
    print("  - 红/黄灯 + 距离<60m → 刹车")
    print("  - 绿灯/无灯 → 加速到8m/s")
    print()
    
    # 创建环境
    print("创建环境...")
    env = make_sumo_env(stage=2, use_gui=use_gui)
    
    # 统计
    results = []
    
    print("\n" + "=" * 60)
    print("开始测试")
    print("=" * 60)
    
    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            route_length = info.get('route_length', 0)
            route_tls = info.get('route_traffic_lights', 0)
            
            print(f"\n{'─' * 40}")
            print(f"Episode {ep+1}/{n_episodes}")
            print(f"  路线长度: {route_length:.0f}m")
            print(f"  红绿灯数(统计): {route_tls}")
            print(f"{'─' * 40}")
            
            while not done:
                action = rule_based_action(obs, verbose=verbose)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                if use_gui:
                    time.sleep(delay)
                
                # 每500步打印一次进度
                if steps % 500 == 0:
                    progress = info.get('route_progress', 0) * 100
                    dist_to_goal = info.get('distance_to_goal', 0)
                    print(f"  [步数 {steps}] 进度:{progress:.1f}% 距终点:{dist_to_goal:.0f}m")
            
            # Episode 结果
            success = info.get('goal_reached', False)
            collision = info.get('collision', False)
            red_light_violations = info.get('red_light_violations', 0)
            progress = info.get('route_progress', 0) * 100
            
            result = {
                'success': success,
                'steps': steps,
                'reward': total_reward,
                'collision': collision,
                'red_light_violations': red_light_violations,
                'progress': progress,
                'route_length': route_length,
            }
            results.append(result)
            
            status = "✅ 成功" if success else ("💥 碰撞" if collision else "❌ 失败")
            print(f"\n  结果: {status}")
            print(f"  步数: {steps}/1500")
            print(f"  进度: {progress:.1f}%")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  闯红灯: {red_light_violations}次")
            
            if use_gui and ep < n_episodes - 1:
                input("  按Enter继续下一个episode...")
    
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    
    finally:
        env.close()
    
    # 汇总统计
    if results:
        print("\n" + "=" * 60)
        print("测试汇总")
        print("=" * 60)
        
        n = len(results)
        success_count = sum(1 for r in results if r['success'])
        collision_count = sum(1 for r in results if r['collision'])
        timeout_count = n - success_count - collision_count
        
        avg_steps = np.mean([r['steps'] for r in results])
        avg_progress = np.mean([r['progress'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        total_red_light = sum(r['red_light_violations'] for r in results)
        
        print(f"\n完成 {n} 个 episodes:")
        print(f"  ✅ 成功: {success_count}/{n} ({success_count/n*100:.1f}%)")
        print(f"  💥 碰撞: {collision_count}/{n} ({collision_count/n*100:.1f}%)")
        print(f"  ⏰ 超时: {timeout_count}/{n} ({timeout_count/n*100:.1f}%)")
        print()
        print(f"平均统计:")
        print(f"  平均步数: {avg_steps:.0f}/1500")
        print(f"  平均进度: {avg_progress:.1f}%")
        print(f"  平均Reward: {avg_reward:.2f}")
        print(f"  总闯红灯: {total_red_light}次")
        
        print("\n" + "=" * 60)
        print("结论")
        print("=" * 60)
        
        if success_count / n >= 0.5:
            print("✅ 规则策略成功率 >= 50%")
            print("   任务可行，问题在于 RL 模型的奖励设计")
        elif avg_progress >= 70:
            print("⚠️ 规则策略成功率低，但平均进度 >= 70%")
            print("   任务边缘可行，建议增加 max_episode_steps 到 2000")
        else:
            print("❌ 规则策略成功率低，平均进度也低")
            print("   任务设置可能有问题，建议：")
            print("   1. 增加 max_episode_steps 到 2500-3000")
            print("   2. 或缩短路线长度")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='规则策略测试')
    parser.add_argument('-n', '--n-episodes', type=int, default=5,
                        help='测试episode数 (默认5)')
    parser.add_argument('--no-gui', action='store_true',
                        help='不使用GUI (批量测试用)')
    parser.add_argument('--delay', type=float, default=0.05,
                        help='每步延时秒数 (默认0.05，越大越慢)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='打印每步的决策细节')
    
    args = parser.parse_args()
    
    # 检查 SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        print("错误: 未设置环境变量 SUMO_HOME")
        sys.exit(1)
    
    run_test(
        n_episodes=args.n_episodes,
        use_gui=not args.no_gui,
        delay=args.delay,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
