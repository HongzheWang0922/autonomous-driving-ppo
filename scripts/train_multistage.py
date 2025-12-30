#!/usr/bin/env python3
"""
多阶段训练脚本 - 完整修复版
支持Stage 1/2/3的训练和续训
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import argparse
import numpy as np

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    REPO_DIR = Path(__file__).parent.parent
    sys.path.insert(0, str(REPO_DIR))
    
    from envs.intersection_env import IntersectionEnvWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback


    class DebugEvalCallback(BaseCallback):
        """修复版评估Callback"""
        def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=5, 
                     log_path="../outputs/logs/', verbose=1):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.log_path = Path(log_path)
            self.evaluations_rewards = []
            self.evaluations_timesteps = []
            self.eval_count = 0
            self.last_eval_step = 0
            
            print(f"\n{'='*60}")
            print(f"🔧 评估Callback已初始化")
            print(f"   评估频率: 每{eval_freq}步")
            print(f"   评估episodes: {n_eval_episodes}")
            print(f"{'='*60}\n")
        
        def _on_step(self) -> bool:
            if self.num_timesteps % 1000 == 0 and self.num_timesteps != self.last_eval_step:
                print(f"[Callback] 步数: {self.num_timesteps:,}, n_calls: {self.n_calls}")
            
            if self.num_timesteps - self.last_eval_step >= self.eval_freq:
                self.last_eval_step = self.num_timesteps
                self.eval_count += 1
                
                print(f"\n{'🔵'*30}")
                print(f"🎯 第{self.eval_count}次评估 (步数: {self.num_timesteps:,})")
                print(f"{'🔵'*30}\n")
                
                episode_rewards = []
                
                try:
                    for ep in range(self.n_eval_episodes):
                        print(f"  Episode {ep+1}/{self.n_eval_episodes}...", end=" ")
                        obs = self.eval_env.reset()
                        done = False
                        ep_reward = 0
                        ep_length = 0
                        
                        while not done and ep_length < 600:
                            action, _ = self.model.predict(obs, deterministic=True)
                            obs, reward, done, info = self.eval_env.step(action)
                            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                            ep_length += 1
                        
                        episode_rewards.append(ep_reward)
                        print(f"Reward: {ep_reward:.2f}, Length: {ep_length}")
                    
                    mean_reward = np.mean(episode_rewards)
                    std_reward = np.std(episode_rewards)
                    
                    print(f"\n  💾 写入Tensorboard...")
                    self.logger.record("eval/mean_reward", mean_reward)
                    self.logger.record("eval/std_reward", std_reward)
                    
                    self.evaluations_rewards.append(episode_rewards)
                    self.evaluations_timesteps.append(self.num_timesteps)
                    
                    self._save_npz()
                    
                    print(f"\n{'='*60}")
                    print(f"📊 评估结果 (步数: {self.num_timesteps:,})")
                    print(f"   平均Reward: {mean_reward:.2f} ± {std_reward:.2f}")
                    print(f"   已评估: {self.eval_count}次")
                    print(f"{'='*60}\n")
                    
                except Exception as e:
                    print(f"\n❌ 评估失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            return True
        
        def _save_npz(self):
            try:
                self.log_path.mkdir(parents=True, exist_ok=True)
                save_path = self.log_path / "evaluations.npz"
                
                np.savez(
                    save_path,
                    timesteps=np.array(self.evaluations_timesteps),
                    results=np.array(self.evaluations_rewards)
                )
                print(f"  ✅ 已保存: {save_path}")
            except Exception as e:
                print(f"  ❌ 保存失败: {e}")
        
        def _on_training_end(self) -> None:
            self._save_npz()
            print(f"\n✅ 训练结束，共评估{self.eval_count}次")


    def create_output_dirs():
        dirs = ["../outputs/models', "../outputs/logs/stage1', "../outputs/logs/stage2', 
                "../outputs/logs/stage3', "../outputs/figures']
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


    def make_env(difficulty, rank, seed=42):
        def _init():
            env = IntersectionEnvWrapper(difficulty=difficulty)
            env.reset(seed=seed + rank)
            return env
        return _init


    # 解析参数
    parser = argparse.ArgumentParser(description='多阶段训练')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--n-envs', type=int, default=16)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--from-checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    create_output_dirs()
    
    # 根据stage设置参数
    difficulty_map = {1: 'easy', 2: 'medium', 3: 'hard'}
    prefix_map = {1: 'ppo_stage1', 2: 'ppo_stage2', 3: 'ppo_stage3'}
    default_timesteps = {1: 200000, 2: 400000, 3: 400000}
    
    difficulty = difficulty_map[args.stage]
    prefix = prefix_map[args.stage]
    
    if args.timesteps is None:
        args.timesteps = default_timesteps[args.stage]
    
    print("="*60)
    print(f"🚀 Stage {args.stage} 训练")
    print(f"难度: {difficulty}")
    print(f"训练步数: {args.timesteps:,}")
    print("="*60)
    
    # 创建环境
    print(f"\n📦 创建环境 (难度: {difficulty})...")
    try:
        env = SubprocVecEnv([make_env(difficulty, i) for i in range(args.n_envs)], start_method='spawn')
        eval_env = DummyVecEnv([lambda d=difficulty: IntersectionEnvWrapper(difficulty=d)])
        print(f"✅ {args.n_envs}个并行环境")
    except Exception as e:
        print(f"⚠️ 降级为8个串行环境: {e}")
        env = DummyVecEnv([make_env(difficulty, i) for i in range(8)])
        eval_env = DummyVecEnv([lambda d=difficulty: IntersectionEnvWrapper(difficulty=d)])
    
    # 创建或加载模型
    if args.from_checkpoint:
        # 从指定checkpoint加载
        print(f"\n📥 从checkpoint加载: {args.from_checkpoint}")
        model = PPO.load(args.from_checkpoint, env=env, device='cpu')
        model.tensorboard_log = f"../outputs/logs/stage{args.stage}/"
        reset_timesteps = False
    elif args.stage == 1:
        # Stage 1: 创建新模型
        print(f"\n🆕 创建新模型")
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
            verbose=1, device='cpu',
            tensorboard_log=f"../outputs/logs/stage{args.stage}/"
        )
        reset_timesteps = True
    else:
        # Stage 2/3: 从前一个stage加载
        prev_model = f"../outputs/models/ppo_stage{args.stage-1}_final.zip'
        print(f"\n📥 从{prev_model}加载模型")
        
        if not Path(prev_model).exists():
            print(f"❌ 错误: 找不到{prev_model}")
            print(f"请先完成Stage {args.stage-1}的训练！")
            sys.exit(1)
        
        model = PPO.load(prev_model, env=env, device='cpu')
        model.tensorboard_log = f"../outputs/logs/stage{args.stage}/"
        reset_timesteps = False
    
    # Callbacks
    print(f"\n🔧 设置Callbacks...")
    eval_callback = DebugEvalCallback(
        eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        log_path=f"../outputs/logs/stage{args.stage}/',
        verbose=1
    )
    
    checkpoint = CheckpointCallback(
        save_freq=10000,
        save_path="../outputs/models/',
        name_prefix=prefix
    )
    
    callbacks = CallbackList([eval_callback, checkpoint])
    
    print(f"\n💡 Tensorboard: tensorboard --logdir ./outputs/logs --reload_interval 5")
    print(f"\n{'='*60}")
    print(f"开始训练...")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_timesteps
    )
    
    # 保存最终模型
    final_name = f'ppo_stage{args.stage}_final' if args.stage < 3 else 'ppo_final'
    model.save(f"../outputs/models/{final_name}")
    print(f"\n✅ Stage {args.stage} 训练完成！")
    print(f"✅ 模型已保存: outputs/models/{final_name}.zip")
    
    env.close()
    eval_env.close()
