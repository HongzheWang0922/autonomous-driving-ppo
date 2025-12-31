"""
LLM训练顾问模块 - 使用Gemini API分析训练过程并提供建议
支持从Stage 2开始使用，可选启用

功能：
- 每10000 episode分析训练统计数据
- 识别训练问题（如闯红灯、碰撞频繁等）
- 提供奖励函数和超参数调整建议
- 日志保存到outputs/llm_logs/
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class LLMTrainingAdvisor:
    """
    LLM训练顾问
    使用Gemini API分析训练数据并提供建议
    """
    
    def __init__(
        self,
        api_key: str,
        stage: int,
        output_dir: str = "../outputs/llm_logs",
        call_frequency: int = 10000,  # 每10000 episode调用一次
        max_calls_per_day: int = 200,
        enabled: bool = True,
    ):
        """
        Args:
            api_key: Gemini API Key
            stage: 训练阶段 (1-4)
            output_dir: 日志输出目录
            call_frequency: 调用频率（episode数）
            max_calls_per_day: 每天最大调用次数
            enabled: 是否启用
        """
        self.api_key = api_key
        self.stage = stage
        self.output_dir = Path(output_dir)
        self.call_frequency = call_frequency
        self.max_calls_per_day = max_calls_per_day
        self.enabled = enabled
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.total_calls = 0
        self.calls_today = 0
        self.last_call_date = None
        self.last_call_episode = 0
        
        # Episode数据缓存（用于统计）
        self.episode_data = []
        
        # 加载Google Generative AI
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                print(f"LLM训练顾问已启用 (Gemini API)")
                print(f"   - Stage: {stage}")
                print(f"   - 调用频率: 每{call_frequency} episode")
                print(f"   - 日志目录: {output_dir}")
            except ImportError:
                print(f"请安装 google-generativeai: pip install google-generativeai")
                self.enabled = False
            except Exception as e:
                print(f"初始化Gemini失败: {e}")
                self.enabled = False
        else:
            print(f"LLM训练顾问未启用")
    
    def record_episode(self, info: Dict):
        """
        记录一个episode的信息
        
        Args:
            info: episode结束时的info字典
        """
        if not self.enabled:
            return
        
        # 提取关键信息
        episode_data = {
            "episode": info.get("episode", 0),
            "total_reward": info.get("total_reward", 0.0),
            "success": info.get("success", 0.0),
            "collision": int(info.get("collision", False)),
            "goal_reached": int(info.get("goal_reached", False)),
            "red_light_violations": info.get("red_light_violations", 0),
            "off_route_count": info.get("off_route_count", 0),
            "route_progress": info.get("route_progress", 0.0),
            "step": info.get("step", 0),
        }
        
        self.episode_data.append(episode_data)
        
        # 保持最近1000个episode
        if len(self.episode_data) > 1000:
            self.episode_data = self.episode_data[-1000:]
    
    def should_call_llm(self, current_episode: int) -> bool:
        """
        判断是否应该调用LLM
        
        Args:
            current_episode: 当前episode数
        
        Returns:
            是否应该调用
        """
        if not self.enabled:
            return False
        
        # 检查调用频率
        if current_episode - self.last_call_episode < self.call_frequency:
            return False
        
        # 检查每日限额
        today = datetime.now().date()
        if self.last_call_date != today:
            self.calls_today = 0
            self.last_call_date = today
        
        if self.calls_today >= self.max_calls_per_day:
            print(f"已达到今日LLM调用上限 ({self.max_calls_per_day})")
            return False
        
        # 至少需要100个episode的数据
        if len(self.episode_data) < 100:
            return False
        
        return True
    
    def analyze_and_advise(self, current_episode: int, training_steps: int) -> Optional[Dict]:
        """
        分析训练数据并获取LLM建议
        
        Args:
            current_episode: 当前episode数
            training_steps: 当前训练步数
        
        Returns:
            建议字典，如果未调用则返回None
        """
        if not self.should_call_llm(current_episode):
            return None
        
        print(f"\n{'='*60}")
        print(f"🤖 调用LLM训练顾问 (Episode {current_episode})")
        print(f"{'='*60}\n")
        
        # 准备统计数据
        stats = self._compute_statistics()
        
        # 生成提示词
        prompt = self._create_prompt(stats, current_episode, training_steps)
        
        # 调用LLM
        try:
            response = self._call_gemini(prompt)
            
            # 解析响应
            advice = {
                "episode": current_episode,
                "training_steps": training_steps,
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "llm_response": response,
            }
            
            # 保存日志
            self._save_log(advice)
            
            # 更新计数
            self.total_calls += 1
            self.calls_today += 1
            self.last_call_episode = current_episode
            
            # 打印建议
            self._print_advice(advice)
            
            return advice
        
        except Exception as e:
            print(f"LLM调用失败: {e}")
            return None
    
    def _compute_statistics(self) -> Dict:
        """计算最近1000个episode的统计数据"""
        if not self.episode_data:
            return {}
        
        data = self.episode_data[-1000:]  # 最近1000个
        
        rewards = [d["total_reward"] for d in data]
        successes = [d["success"] for d in data]
        
        stats = {
            "num_episodes": len(data),
            "success_rate": np.mean([d["goal_reached"] for d in data]) * 100,
            "collision_rate": np.mean([d["collision"] for d in data]) * 100,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_progress": np.mean([d["route_progress"] for d in data]) * 100,
            "mean_steps": np.mean([d["step"] for d in data]),
        }
        
        # Stage特定统计
        if self.stage >= 2:
            total_violations = sum([d["red_light_violations"] for d in data])
            stats["red_light_violations_total"] = total_violations
            stats["red_light_violations_per_episode"] = total_violations / len(data)
        
        if self.stage >= 3:
            stats["off_route_rate"] = np.mean([d["off_route_count"] > 0 for d in data]) * 100
        
        return stats
    
    def _create_prompt(self, stats: Dict, episode: int, training_steps: int) -> str:
        """创建发送给LLM的提示词"""
        
        # Stage描述
        stage_descriptions = {
            1: "Stage 1: 空路导航 - 学习不偏离车道、成功到达终点（无其他车辆）",
            2: "Stage 2: 红绿灯遵守 - 学习遵守交通信号（无其他车辆）",
            3: "Stage 3: 动态避障 - 学习与其他车辆交互和避障",
            4: "Stage 4: 综合场景 - 行人 + 长距离导航 + 复杂交通",
        }
        
        prompt = f"""你是一位强化学习训练专家，正在帮助训练一个自动驾驶PPO Agent。

## 当前训练阶段
{stage_descriptions.get(self.stage, "未知阶段")}

## 训练进度
- 当前Episode: {episode:,}
- 训练步数: {training_steps:,}
- 统计窗口: 最近{stats.get('num_episodes', 0)}个episode

## 训练统计 (最近{stats.get('num_episodes', 0)}个episode)

### 成功率指标
- 成功率 (到达终点): {stats.get('success_rate', 0):.1f}%
- 碰撞率: {stats.get('collision_rate', 0):.1f}%
- 平均路由完成度: {stats.get('mean_progress', 0):.1f}%

### 奖励指标
- 平均Reward: {stats.get('mean_reward', 0):.2f}
- Reward标准差: {stats.get('std_reward', 0):.2f}
- Reward范围: [{stats.get('min_reward', 0):.2f}, {stats.get('max_reward', 0):.2f}]

### 行为指标
- 平均Episode长度: {stats.get('mean_steps', 0):.1f} 步
"""
        
        # 添加Stage特定指标
        if self.stage >= 2:
            prompt += f"""
### 红绿灯遵守
- 总闯红灯次数: {stats.get('red_light_violations_total', 0)}
- 平均每episode闯红灯: {stats.get('red_light_violations_per_episode', 0):.2f}次
"""
        
        if self.stage >= 3:
            prompt += f"""
### 路由遵守
- 偏离路由比例: {stats.get('off_route_rate', 0):.1f}%
"""
        
        prompt += """

## 请提供分析和建议

请分析以上数据，并提供：

1. **问题诊断**: 当前训练的主要问题是什么？（如：成功率低、闯红灯频繁、碰撞率高等）

2. **奖励函数调整建议**: 
   - 哪些奖励权重需要调整？
   - 是否需要增加新的奖励项？
   - 建议的具体数值（如：将红绿灯违规惩罚从-5调整到-10）

3. **超参数调整建议**:
   - 学习率是否需要调整？
   - 其他PPO超参数建议（如entropy coefficient, clip range等）

4. **训练策略建议**:
   - 是否需要增加/减少episode长度？
   - 是否需要调整环境难度？

请直接给出具体、可操作的建议，避免模糊的描述。
"""
        
        return prompt
    
    def _call_gemini(self, prompt: str) -> str:
        """调用Gemini API"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API调用失败: {e}")
    
    def _save_log(self, advice: Dict):
        """保存建议日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"stage{self.stage}_episode{advice['episode']}_{timestamp}.json"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(advice, f, indent=2, ensure_ascii=False)
            print(f"日志已保存: {log_file}")
        except Exception as e:
            print(f"保存日志失败: {e}")
    
    def _print_advice(self, advice: Dict):
        """打印建议到控制台"""
        print(f"\n{'🤖'*30}")
        print(f"LLM训练顾问建议 (Episode {advice['episode']})")
        print(f"{'🤖'*30}\n")
        print(advice['llm_response'])
        print(f"\n{'='*60}\n")
    
    def get_summary(self) -> str:
        """获取使用摘要"""
        return f"""
LLM训练顾问使用摘要:
- 总调用次数: {self.total_calls}
- 今日调用次数: {self.calls_today}
- 最后调用episode: {self.last_call_episode}
- 日志目录: {self.output_dir}
"""


def create_llm_advisor(
    stage: int,
    api_key: Optional[str] = None,
    enabled: bool = False,
    **kwargs
) -> Optional[LLMTrainingAdvisor]:
    """
    便捷函数：创建LLM训练顾问
    
    Args:
        stage: 训练阶段
        api_key: Gemini API Key
        enabled: 是否启用
        **kwargs: 其他参数
    
    Returns:
        LLMTrainingAdvisor实例，如果未启用则返回None
    """
    # Stage 1不使用LLM顾问
    if stage == 1:
        print("Stage 1 不使用LLM训练顾问")
        return None
    
    # 如果未启用，返回None
    if not enabled:
        print("LLM训练顾问未启用")
        return None
    
    # 检查API Key
    if not api_key:
        print("未提供Gemini API Key，LLM训练顾问未启用")
        print("   使用 --llm --llm-api-key YOUR_KEY 启用")
        return None
    
    try:
        advisor = LLMTrainingAdvisor(
            api_key=api_key,
            stage=stage,
            enabled=True,
            **kwargs
        )
        return advisor
    except Exception as e:
        print(f"创建LLM训练顾问失败: {e}")
        return None


# 用于测试
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试LLM训练顾问')
    parser.add_argument('--api-key', type=str, required=True, help='Gemini API Key')
    parser.add_argument('--stage', type=int, default=2, help='训练阶段')
    args = parser.parse_args()
    
    print("测试LLM训练顾问...\n")
    
    # 创建顾问
    advisor = LLMTrainingAdvisor(
        api_key=args.api_key,
        stage=args.stage,
        enabled=True
    )
    
    # 模拟一些episode数据
    print("生成模拟数据...")
    for i in range(100):
        fake_info = {
            "episode": i,
            "total_reward": np.random.normal(50, 20),
            "success": np.random.random(),
            "collision": np.random.random() < 0.2,
            "goal_reached": np.random.random() < 0.6,
            "red_light_violations": np.random.randint(0, 3),
            "off_route_count": np.random.randint(0, 5),
            "route_progress": np.random.uniform(0.5, 1.0),
            "step": np.random.randint(100, 500),
        }
        advisor.record_episode(fake_info)
    
    # 强制调用LLM
    print("\n调用LLM分析...")
    advisor.last_call_episode = -10000  # 强制触发
    advice = advisor.analyze_and_advise(current_episode=100, training_steps=50000)
    
    if advice:
        print("\n测试成功！")
        print(advisor.get_summary())
    else:
        print("\n测试失败")


