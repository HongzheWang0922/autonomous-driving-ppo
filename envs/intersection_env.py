
import gymnasium as gym
import highway_env
import numpy as np

class IntersectionEnvWrapper(gym.Wrapper):
    """
    自定义交叉路口环境
    支持课程学习：easy(4-6车) -> medium(6-10车) -> hard(8-12车)
    """
    def __init__(self, difficulty='easy'):
        # 先创建基础环境
        base_env = gym.make('intersection-v1', render_mode='rgb_array')
        
        # 根据难度设置车辆生成概率
        spawn_prob = {
            'easy': 0.4,
            'medium': 0.6,
            'hard': 0.8
        }
        
        # 配置环境（关键修改：增加duration）
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "normalize": True,
                "observe_intentions": False
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "dynamical": True
            },
            "duration": 40,  # ← 改！从13改到40秒
            "spawn_probability": spawn_prob.get(difficulty, 0.6),
            "collision_reward": -5.0,  # ← 改大惩罚
            "high_speed_reward": 0.4,
            "arrived_reward": 1.0,
            "normalize_reward": False  # ← 关闭归一化，更容易看到差异
        }
        
        # 应用配置
        base_env.unwrapped.configure(config)
        super().__init__(base_env)
        
        self.difficulty = difficulty
        self.episode_count = 0
        
    def reset(self, **kwargs):
        self.episode_count += 1
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 自定义奖励塑形
        shaped_reward = reward
        
        # 惩罚急转弯
        if isinstance(action, np.ndarray) and len(action) >= 2:
            steering_penalty = 0.1 * abs(action[1])
            shaped_reward -= steering_penalty
        
        # 额外奖励/惩罚（基于info）
        if info.get('crashed', False):
            shaped_reward -= 10.0  # 额外碰撞惩罚
        
        # highway-env的arrived信息在rewards字典里
        if 'rewards' in info and info['rewards'].get('arrived_reward', 0) > 0:
            shaped_reward += 10.0  # 额外到达奖励
        
        return obs, shaped_reward, terminated, truncated, info
    
    def get_stats(self):
        return {
            'difficulty': self.difficulty,
            'episode_count': self.episode_count
        }

print("✓ envs/intersection_env.py 已更新（duration=40）")
