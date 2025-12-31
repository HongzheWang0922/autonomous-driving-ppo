
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time

class RealtimeMonitorCallback(BaseCallback):
    """
    实时监控训练过程：每N步显示loss和reward
    """
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_collisions = []

        self.last_mean_reward = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # 收集episode信息
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])

                        # 统计成功和碰撞
                        if 'arrived' in info:
                            self.episode_successes.append(1 if info['arrived'] else 0)
                        if 'crashed' in info:
                            self.episode_collisions.append(1 if info['crashed'] else 0)

        # 每check_freq步打印一次
        if self.n_calls % self.check_freq == 0:
            self._print_status()

        return True

    def _print_status(self):
        """打印当前训练状态"""
        if self.verbose == 0:
            return

        elapsed_time = time.time() - self.start_time
        fps = self.n_calls / elapsed_time if elapsed_time > 0 else 0

        print(f"\n{'='*70}")
        print(f"训练进度: {self.n_calls:,} steps | 耗时: {elapsed_time/60:.1f} min | FPS: {fps:.1f}")
        print(f"{'='*70}")

        # Loss信息
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                if hasattr(self.model.logger, 'name_to_value'):
                    metrics = self.model.logger.name_to_value

                    print("Loss指标:")
                    if 'train/loss' in metrics:
                        print(f"   Total Loss: {metrics['train/loss']:.4f}")
                    if 'train/policy_gradient_loss' in metrics:
                        print(f"   Policy Loss: {metrics['train/policy_gradient_loss']:.4f}")
                    if 'train/value_loss' in metrics:
                        print(f"   Value Loss: {metrics['train/value_loss']:.4f}")
                    if 'train/entropy_loss' in metrics:
                        print(f"   Entropy: {metrics['train/entropy_loss']:.4f}")
                    if 'train/explained_variance' in metrics:
                        print(f"   Explained Var: {metrics['train/explained_variance']:.4f}")
            except Exception:
                pass

        # Reward信息
        if len(self.episode_rewards) > 0:
            recent_n = min(20, len(self.episode_rewards))
            recent_rewards = self.episode_rewards[-recent_n:]
            recent_lengths = self.episode_lengths[-recent_n:]

            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            mean_length = np.mean(recent_lengths)

            print(f"\n性能指标 (最近{recent_n}个episodes):")
            print(f"   平均Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"   最高Reward: {max(self.episode_rewards[-recent_n:]):.2f}")
            print(f"   平均长度: {mean_length:.1f} steps")

            if len(self.episode_successes) > 0:
                recent_success = self.episode_successes[-recent_n:]
                success_rate = np.mean(recent_success) * 100
                print(f"   成功率: {success_rate:.1f}%")

            if len(self.episode_collisions) > 0:
                recent_collision = self.episode_collisions[-recent_n:]
                collision_rate = np.mean(recent_collision) * 100
                print(f"   碰撞率: {collision_rate:.1f}%")

            if self.last_mean_reward > 0:
                change = mean_reward - self.last_mean_reward
                arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                print(f"   趋势: {arrow} {change:+.2f}")

            self.last_mean_reward = mean_reward
            print(f"   已完成episodes: {len(self.episode_rewards)}")
        else:
            print("\n⏳ 等待第一个episode完成...")

        print(f"{'='*70}\n")
