
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
            'easy': 0.4,    # 约4-6辆车
            'medium': 0.6,  # 约6-10辆车
            'hard': 0.8     # 约8-12辆车
        }
        
        # 配置环境（关键：保持observation和action一致）
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,  # 最多观测15辆车
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
            "duration": 13,
            "spawn_probability": spawn_prob.get(difficulty, 0.6),
            "collision_reward": -1.0,
            "high_speed_reward": 0.4,
            "arrived_reward": 1.0,
            "normalize_reward": True
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
        
        # 惩罚急转弯（鼓励平滑驾驶）
        if isinstance(action, np.ndarray) and len(action) >= 2:
            steering_penalty = 0.1 * abs(action[1])
            shaped_reward -= steering_penalty
        
        # 额外奖励到达目标
        if info.get('arrived', False):
            shaped_reward += 10.0
        
        # 额外惩罚碰撞
        if info.get('crashed', False):
            shaped_reward -= 10.0
        
        return obs, shaped_reward, terminated, truncated, info
    
    def get_stats(self):
        """返回环境统计信息"""
        return {
            'difficulty': self.difficulty,
            'episode_count': self.episode_count
        }

print("✓ envs/intersection_env.py 创建成功")
