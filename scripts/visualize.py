"""
可视化脚本 - 用SUMO-GUI观看模型驾驶
"""
import sys
import time
import argparse
from pathlib import Path

# 添加项目路径
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

import traci
from stable_baselines3 import PPO
from envs.sumo_env import make_sumo_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化模型驾驶")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--stage", type=int, default=2, help="训练阶段")
    parser.add_argument("--n-episodes", type=int, default=5, help="运行几个episode")
    parser.add_argument("--delay", type=float, default=0.1, help="每步延时(秒)，越大越慢，默认0.1")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，相同seed走相同路线")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"可视化 Stage {args.stage}")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"Episodes: {args.n_episodes}")
    print(f"延时: {args.delay}秒/步")
    if args.seed:
        print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 加载模型
    print("📥 加载模型...")
    model = PPO.load(args.model)
    
    # 创建带GUI的环境
    print("📦 创建可视化环境...")
    env = make_sumo_env(stage=args.stage, use_gui=True)
    
    print("\n" + "=" * 60)
    print("开始可视化，关闭SUMO窗口或按Ctrl+C退出")
    print("=" * 60)
    print("\n提示:")
    print("   - 绿色车辆 = 你的AI车")
    print("   - 黄色车辆 = 背景车")
    print("   - 可以用鼠标拖动/缩放地图")
    print("   - 右键点击车辆查看详情")
    print()
    
    try:
        for ep in range(args.n_episodes):
            # 用seed控制路线
            seed = args.seed if args.seed else None
            obs, info = env.reset(seed=seed)
            done = False
            ep_reward = 0
            ep_steps = 0
            
            print(f"\nEpisode {ep + 1}/{args.n_episodes}")
            print(f"   路线长度: {info.get('route_length', 0):.0f}m")
            print(f"   红绿灯数: {info.get('route_traffic_lights', 0)}")
            
            # 让视角跟随自车
            try:
                traci.gui.trackVehicle("View #0", "ego")
                traci.gui.setZoom("View #0", 800)
            except:
                pass
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1
                
                # 延时让可视化变慢
                time.sleep(args.delay)
            
            # 结果
            success = "成功" if info.get('goal_reached', False) else "失败"
            collision = "💥 碰撞" if info.get('collision', False) else ""
            red_light = f"🚦 闯红灯:{info.get('red_light_violations', 0)}"
            
            print(f"   结果: {success} {collision}")
            print(f"   Reward: {ep_reward:.2f}, 步数: {ep_steps}")
            print(f"   {red_light}")
            
            input("   按Enter继续下一个episode...")
    
    except KeyboardInterrupt:
        print("\n\n用户终止")
    
    finally:
        env.close()
        print("可视化结束")
