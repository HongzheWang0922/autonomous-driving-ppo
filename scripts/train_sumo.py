#!/usr/bin/env python3
"""
基于SUMO的多阶段训练脚本
支持四阶段课程学习和可选的LLM训练顾问

Usage:
    # Stage 1: 空路导航
    python train_sumo.py --stage 1 --timesteps 500000
    
    # Stage 2: 红绿灯遵守 (启用LLM顾问)
    python train_sumo.py --stage 2 --timesteps 800000 --llm --llm-api-key YOUR_KEY
    
    # Stage 3: 动态避障
    python train_sumo.py --stage 3 --timesteps 1000000 --llm --llm-api-key YOUR_KEY
    
    # Stage 4: 综合场景
    python train_sumo.py --stage 4 --timesteps 1500000 --llm --llm-api-key YOUR_KEY
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# 添加项目路径
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs.sumo_env import make_sumo_env
from utils.llm_advisor import create_llm_advisor


class EpisodeStatCallback(BaseCallback):
    """
    统计Episode信息的Callback
    用于LLM训练顾问和Tensorboard记录
    """
    
    def __init__(
        self,
        llm_advisor=None,
        eval_freq: int = 5000,
        log_freq: int = 100,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.llm_advisor = llm_advisor
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        
        # 统计
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_collisions = []
        
        # 最近100个episode的统计
        self.recent_rewards = []
        self.recent_successes = []
        self.recent_collisions = []
    
    def _on_step(self) -> bool:
        # 检查是否有episode结束
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals.get("infos", [])
            
            for i, done in enumerate(dones):
                if done and i < len(infos):
                    info = infos[i]
                    
                    # 提取episode信息
                    episode_reward = info.get("total_reward", 0.0)
                    episode_length = info.get("step", 0)
                    success = float(info.get("goal_reached", False))
                    collision = float(info.get("collision", False))
                    
                    # 记录
                    self.episode_count += 1
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_successes.append(success)
                    self.episode_collisions.append(collision)
                    
                    self.recent_rewards.append(episode_reward)
                    self.recent_successes.append(success)
                    self.recent_collisions.append(collision)
                    
                    # 保持最近100个
                    if len(self.recent_rewards) > 100:
                        self.recent_rewards = self.recent_rewards[-100:]
                        self.recent_successes = self.recent_successes[-100:]
                        self.recent_collisions = self.recent_collisions[-100:]
                    
                    # 记录到LLM顾问
                    if self.llm_advisor:
                        self.llm_advisor.record_episode(info)
                    
                    # 定期打印统计
                    if self.episode_count % self.log_freq == 0 and len(self.recent_rewards) > 0:
                        mean_reward = np.mean(self.recent_rewards)
                        success_rate = np.mean(self.recent_successes) * 100
                        collision_rate = np.mean(self.recent_collisions) * 100
                        
                        print(f"\n{'='*60}")
                        print(f"Episode {self.episode_count} | 步数 {self.num_timesteps:,}")
                        print(f"   最近100个episode:")
                        print(f"   - 平均Reward: {mean_reward:.2f}")
                        print(f"   - 成功率: {success_rate:.1f}%")
                        print(f"   - 碰撞率: {collision_rate:.1f}%")
                        print(f"{'='*60}\n")
                    
                    # 记录到Tensorboard
                    self.logger.record("episode/reward", episode_reward)
                    self.logger.record("episode/length", episode_length)
                    self.logger.record("episode/success", success)
                    self.logger.record("episode/collision", collision)
                    
                    if len(self.recent_rewards) > 0:
                        self.logger.record("episode/mean_reward_100", np.mean(self.recent_rewards))
                        self.logger.record("episode/success_rate_100", np.mean(self.recent_successes))
                        self.logger.record("episode/collision_rate_100", np.mean(self.recent_collisions))
        
        # 调用LLM顾问
        if self.llm_advisor and self.episode_count > 0:
            advice = self.llm_advisor.analyze_and_advise(
                current_episode=self.episode_count,
                training_steps=self.num_timesteps
            )
            
            if advice:
                # 可以在这里根据建议自动调整参数（高级功能）
                pass
        
        return True


class EvalCallback(BaseCallback):
    """
    评估Callback
    """
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0
        self.eval_count = 0
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self.eval_count += 1
            
            print(f"\n{'='*30}")
            print(f"评估 #{self.eval_count} (步数: {self.num_timesteps:,})")
            print(f"{'='*30}\n")
            
            episode_rewards = []
            episode_successes = []
            episode_collisions = []
            episode_red_lights = []
            
            for ep in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0
                ep_steps = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_steps += 1
                episode_rewards.append(ep_reward)
                episode_successes.append(float(info.get("goal_reached", False)))
                episode_collisions.append(float(info.get("collision", False)))
                episode_red_lights.append(info.get("red_light_violations", 0))
                print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Steps={ep_steps}/{info.get('max_steps', '?')}, "
                      f"Success={info.get('goal_reached', False)}, "
                      f"Collision={info.get('collision', False)}, "
                      f"RedLight={info.get('red_light_violations', 0)}/{info.get('route_traffic_lights', '?')}, "
                      f"BgVehicles={info.get('avg_bg_vehicles', 0):.1f}")
            
            mean_reward = np.mean(episode_rewards)
            success_rate = np.mean(episode_successes) * 100
            collision_rate = np.mean(episode_collisions) * 100
            total_red_lights = sum(episode_red_lights)
            episodes_with_violations = sum(1 for r in episode_red_lights if r > 0)
            
            print(f"\n评估结果:")
            print(f"   - 平均Reward: {mean_reward:.2f}")
            print(f"   - 成功率: {success_rate:.1f}%")
            print(f"   - 碰撞率: {collision_rate:.1f}%")
            print(f"   - 闯红灯: {total_red_lights}次 ({episodes_with_violations}/{self.n_eval_episodes}个episode违规)")
            print(f"{'='*60}\n")
            
            # 记录到Tensorboard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/collision_rate", collision_rate)
        
        return True


def make_vec_env(stage: int, map_name: str, n_envs: int, start_method: str = 'spawn'):
    """
    创建向量化环境
    
    Args:
        stage: 训练阶段
        map_name: 地图名称
        n_envs: 并行环境数
        start_method: 多进程启动方法
    
    Returns:
        向量化环境
    """
    def make_env(rank: int):
        def _init():
            env = make_sumo_env(
                stage=stage,
                map_name=map_name,
                use_gui=False,
                seed=42 + rank
            )
            env = Monitor(env)
            return env
        return _init
    
    # 尝试使用SubprocVecEnv，如果失败则降级到DummyVecEnv
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    print(f"创建了 {n_envs} 个环境 (DummyVecEnv)")
    
    return env


def create_or_load_model(
    env,
    stage: int,
    from_checkpoint: Optional[str] = None,
    device: str = 'cpu'
) -> PPO:
    """
    创建或加载PPO模型
    
    Args:
        env: 训练环境
        stage: 当前阶段
        from_checkpoint: checkpoint路径（可选）
        device: 设备
    
    Returns:
        PPO模型
    """
    tensorboard_log = str(REPO_DIR / "outputs" / "logs" / f"stage{stage}")
    
    if from_checkpoint:
        # 从指定checkpoint加载
        print(f"📥 从checkpoint加载: {from_checkpoint}")
        model = PPO.load(from_checkpoint, env=env, device=device)
        model.tensorboard_log = tensorboard_log
        reset_timesteps = False
    
    elif stage == 1:
        # Stage 1: 创建新模型
        print(f"🆕 创建新的PPO模型")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,  # 恢复原始学习率
            n_steps=2048,
            batch_size=128,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device=device,
            tensorboard_log=tensorboard_log
        )
        reset_timesteps = True
    
    else:
        # Stage 2-4: 从前一阶段加载
        prev_model_path = str(REPO_DIR / "outputs" / "models" / f"best_stage{stage-1}" / "ppo_final.zip")
        
        if not Path(prev_model_path).exists():
            print(f"找不到前一阶段模型: {prev_model_path}")
            print(f"   创建新模型...")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,  # 恢复原始学习率
                n_steps=2048,
                batch_size=128,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                device=device,
                tensorboard_log=tensorboard_log
            )
            reset_timesteps = True
        else:
            print(f"📥 从前一阶段加载: {prev_model_path}")
            model = PPO.load(prev_model_path, env=env, device=device)
            model.tensorboard_log = tensorboard_log
            reset_timesteps = False
    
    return model, reset_timesteps


def main():
    parser = argparse.ArgumentParser(description='SUMO多阶段训练')
    
    # 基础参数
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4],
                        help='训练阶段 (1=空路, 2=红绿灯, 3=避障, 4=综合)')
    parser.add_argument('--map', type=str, default='sf_mission',
                        help='地图名称 (默认: sf_mission)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='训练步数 (默认: Stage1=500k, Stage2=800k, Stage3=1M, Stage4=1.5M)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='并行环境数 (默认: 8-16根据硬件自动选择)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='训练设备')
    parser.add_argument('--from-checkpoint', type=str, default=None,
                        help='从指定checkpoint继续训练')
    
    # LLM训练顾问参数
    parser.add_argument('--llm', '--enable-llm', action='store_true',
                        help='启用LLM训练顾问 (Stage 2+)')
    parser.add_argument('--llm-api-key', type=str, default=None,
                        help='Gemini API Key')
    
    # 其他参数
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='评估频率 (默认: 10000步)')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                        help='Checkpoint保存频率 (默认: 50000步)')
    parser.add_argument('--gui', action='store_true',
                        help='使用SUMO-GUI (仅用于调试)')
    
    args = parser.parse_args()
    
    # 设置默认参数
    if args.timesteps is None:
        default_timesteps = {1: 500000, 2: 1000000, 3: 1500000, 4: 2000000}
        args.timesteps = default_timesteps[args.stage]
    
    if args.n_envs is None:
        # 根据stage调整并行环境数
        # Stage 1-2: 16个环境（无背景车辆，较快）
        # Stage 3-4: 8个环境（有背景车辆，较慢）
        args.n_envs = 16 if args.stage <= 2 else 8
    
    # 打印配置
    print(f"\n{'='*60}")
    print(f"SUMO多阶段训练")
    print(f"{'='*60}")
    print(f"阶段: Stage {args.stage}")
    print(f"地图: {args.map}")
    print(f"训练步数: {args.timesteps:,}")
    print(f"并行环境: {args.n_envs}")
    print(f"设备: {args.device}")
    print(f"LLM顾问: {'启用' if args.llm else '未启用'}")
    print(f"{'='*60}\n")
    
    # 检查SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        print("错误: 未设置环境变量 SUMO_HOME")
        print("   请设置SUMO安装路径，例如:")
        print("   Windows: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")
        print("   Linux: export SUMO_HOME=/usr/share/sumo")
        sys.exit(1)
    
    # 检查地图文件
    map_file = REPO_DIR / "maps" / f"{args.map}.net.xml"
    if not map_file.exists():
        print(f"错误: 找不到地图文件 {map_file}")
        print(f"   请先运行: python scripts/download_map.py --region {args.map}")
        sys.exit(1)
    
    # 创建输出目录
    output_dirs = [
        REPO_DIR / "outputs" / "models" / f"best_stage{args.stage}",
        REPO_DIR / "outputs" / "logs" / f"stage{args.stage}",
        REPO_DIR / "outputs" / "llm_logs",
    ]
    for d in output_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # 创建LLM训练顾问
    llm_advisor = None
    if args.llm:
        llm_advisor = create_llm_advisor(
            stage=args.stage,
            api_key=args.llm_api_key,
            enabled=True
        )
    
    # 创建环境
    print(f"📦 创建训练环境...")
    train_env = make_vec_env(args.stage, args.map, args.n_envs)
    
    print(f"📦 创建评估环境...")
    eval_env = make_sumo_env(args.stage, args.map, use_gui=args.gui, seed=999)
    eval_env = Monitor(eval_env)
    
    # 创建或加载模型
    print(f"\n准备模型...")
    model, reset_timesteps = create_or_load_model(
        train_env,
        args.stage,
        args.from_checkpoint,
        args.device
    )
    
    # 创建Callbacks
    print(f"\n🔧 设置Callbacks...")
    
    episode_callback = EpisodeStatCallback(
        llm_advisor=llm_advisor,
        log_freq=100,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),  # 修复：需要除以环境数
        save_path=str(REPO_DIR / "outputs" / "models" / f"best_stage{args.stage}"),
        name_prefix=f"ppo_stage{args.stage}",
        verbose=1
    )
    
    callbacks = CallbackList([episode_callback, eval_callback, checkpoint_callback])
    
    # 打印提示
    print(f"\n监控训练:")
    print(f"   tensorboard --logdir {REPO_DIR}/outputs/logs --reload_interval 5")
    
    if llm_advisor:
        print(f"\nLLM训练顾问:")
        print(f"   - 每10000 episode分析一次")
        print(f"   - 日志保存到: {REPO_DIR}/outputs/llm_logs/")
    
    print(f"\n{'='*60}")
    print(f"开始训练 Stage {args.stage}...")
    print(f"{'='*60}\n")
    
    # 开始训练
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps
        )
    except KeyboardInterrupt:
        print(f"\n训练被中断")
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存最终模型
    final_model_path = str(REPO_DIR / "outputs" / "models" / f"best_stage{args.stage}" / "ppo_final.zip")
    model.save(final_model_path)
    print(f"\n训练完成！")
    print(f"最终模型已保存: {final_model_path}")
    
    # LLM顾问摘要
    if llm_advisor:
        print(llm_advisor.get_summary())
    
    # 关闭环境
    train_env.close()
    eval_env.close()
    
    # 检查成功率，决定是否可以进入下一阶段
    print(f"\n{'='*60}")
    print(f"Stage {args.stage} 训练完成")
    print(f"{'='*60}")
    print(f"请运行评估脚本检查成功率:")
    print(f"  python scripts/evaluate_sumo.py --stage {args.stage} --n-episodes 100")
    print(f"\n如果成功率 >= 80%，可以进入下一阶段:")
    if args.stage < 4:
        print(f"  python scripts/train_sumo.py --stage {args.stage + 1}")
    else:
        print(f"  恭喜！已完成所有训练阶段！")
    print(f"\n")


if __name__ == "__main__":
    # 设置多进程启动方法
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except:
        pass
    
    main()


