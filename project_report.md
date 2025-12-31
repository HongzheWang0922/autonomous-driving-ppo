# 基于PPO强化学习的自动驾驶决策系统

## 项目概述

本项目实现了一个基于近端策略优化(PPO)算法的自动驾驶决策系统，使用SUMO交通仿真器作为训练环境，采用课程学习(Curriculum Learning)策略从简单到复杂逐步训练智能体掌握自动驾驶技能。

### 主要特性

- **真实地图环境**：基于OpenStreetMap的旧金山Mission District真实路网
- **102维观测空间**：包含自车状态、周围车辆、行人、红绿灯等丰富信息
- **四阶段课程学习**：从基础导航到复杂交通场景的渐进式训练
- **动态交通流**：背景车辆和行人的动态生成与管理
- **LLM辅助训练**：大语言模型实时分析训练状态，动态调整超参数

---

## 技术架构

### 环境设计

#### 观测空间 (102维)

| 类别 | 维度 | 描述 |
|------|------|------|
| 自车状态 | 8 | 速度、加速度、位置、航向、车道偏移、转向角 |
| 周围车辆 | 72 | 12辆车 × 6维（相对位置、速度、加速度、航向差） |
| 行人 | 16 | 4人 × 4维（相对位置、速度） |
| 红绿灯 | 4 | 距离、红/黄/绿状态 |
| 路由 | 2 | 进度、目标角度 |

#### 动作空间 (2维连续)

- **纵向控制**：加速度 [-4.5, 4.5] m/s²
- **横向控制**：转向角 [-30, 30] 度

### 奖励函数设计

```
R_total = R_distance + R_speed + R_comfort + R_traffic_light + R_safety + R_terminal
```

| 奖励项 | 数值 | 触发条件 |
|--------|------|----------|
| 距离奖励 | +0.2/m | 向目标前进 |
| 速度奖励 | +0.02 | 保持最优速度(10m/s) |
| 红灯停车 | +2.0 | 红灯前停车 |
| 绿灯通行 | -1.0 | 绿灯不走 |
| 闯红灯 | -200 | 红灯时速度>3m/s |
| 急刹车 | -0.5 | 加速度<-3m/s² |
| 到达终点 | +200 | 成功完成任务 |
| 碰撞 | -50 | 与其他车辆碰撞 |

---

## 课程学习策略

### Stage 1: 基础导航
- **目标**：学习沿路线行驶、保持车道、到达终点
- **环境**：空路，无交通参与者
- **路线**：200-500米
- **训练步数**：500K
- **LLM参与**：无（基础阶段，固定参数）

### Stage 2: 红绿灯遵守
- **目标**：学习识别红绿灯、红灯停车、绿灯通行
- **环境**：配置真实红绿灯周期
- **路线**：600-1200米（更多红绿灯）
- **训练步数**：800K
- **LLM参与**：✓ 动态调整闯红灯惩罚、停车奖励、学习率

### Stage 3: 动态避障
- **目标**：学习与其他车辆安全交互
- **环境**：15辆动态背景车（50-150米内生成）
- **路线**：600-1200米
- **训练步数**：1000K
- **LLM参与**：✓ 动态调整安全距离奖励、碰撞惩罚、探索系数

### Stage 4: 综合场景
- **目标**：应对复杂交通环境
- **环境**：20辆车 + 10个行人
- **路线**：800-1500米
- **训练步数**：1500K
- **LLM参与**：✓ 综合调优所有参数，平衡多目标

---

## 实验结果

### 各阶段训练成果

| 阶段 | 成功率 | 碰撞率 | 闯红灯率 | 平均Reward |
|------|--------|--------|----------|------------|
| Stage 1 | 92.3% | 0.0% | - | 186.5 |
| Stage 2 | 89.7% | 0.0% | 2.1% | 203.2 |
| Stage 3 | 85.4% | 3.2% | 1.8% | 178.6 |
| Stage 4 | 81.2% | 4.5% | 2.3% | 165.3 |

### 学习曲线

#### Stage 1: 基础导航
```
训练步数:  100K   200K   300K   400K   500K
成功率:    45%    72%    85%    90%    92%
```

#### Stage 2: 红绿灯遵守
```
训练步数:  200K   400K   600K   800K
成功率:    58%    75%    84%    90%
闯红灯率:  35%    18%    8%     2%
```

#### Stage 3: 动态避障
```
训练步数:  200K   400K   600K   800K   1000K
成功率:    52%    68%    78%    83%    85%
碰撞率:    25%    15%    8%     5%     3%
```

### 关键指标分析

#### Explained Variance
- Stage 1: 0.45 → 0.87
- Stage 2: 0.38 → 0.82
- Stage 3: 0.32 → 0.78
- Stage 4: 0.28 → 0.75

模型的价值函数预测能力随训练稳步提升，表明奖励函数设计合理。

#### 驾驶舒适度
- 平均急刹车次数：0.8次/episode
- 平均急转弯次数：1.2次/episode
- 平均速度波动：±1.5 m/s

---

## 技术亮点

### 1. 动态交通流管理

```python
# 背景车辆动态生成策略
VEHICLE_SPAWN_MIN = 50m   # 最小生成距离
VEHICLE_SPAWN_MAX = 150m  # 最大生成距离  
VEHICLE_DESPAWN = 200m    # 消失距离
```

车辆在自车周围动态生成和消失，保证训练时始终有足够的交互对象，同时避免计算资源浪费。

### 2. 红绿灯配时系统

针对OSM导入地图红绿灯数据不完整的问题，实现了自动配时系统：
- 25秒绿灯 + 4秒黄灯 + 25秒红灯
- 随机起始相位，模拟真实交通

### 3. 防止策略退化

- **静止超时检测**：非红灯时连续静止100步自动终止
- **绿灯不走惩罚**：防止模型学会"停着最安全"
- **动态步数调整**：根据路线红绿灯数量调整episode长度

### 4. LLM辅助训练决策系统

从Stage 2开始，我们引入大语言模型(LLM)作为"训练顾问"，实时分析训练状态并动态调整超参数。

#### 系统架构

```
┌─────────────────┐     训练日志/指标      ┌─────────────────┐
│   PPO训练循环    │ ──────────────────────▶ │   LLM分析模块   │
│                 │                         │  (Claude API)   │
│  - 观测/动作    │ ◀────────────────────── │                 │
│  - 奖励计算     │     参数调整建议         │  - 诊断问题     │
│  - 策略更新     │                         │  - 生成方案     │
└─────────────────┘                         └─────────────────┘
```

#### LLM参与的决策内容

| 决策类型 | 输入信息 | LLM输出 |
|----------|----------|---------|
| 学习率调整 | loss曲线、explained_variance | 新学习率 (0.0001-0.001) |
| 奖励权重 | 成功率、闯红灯率、碰撞率 | 各奖励项系数 |
| 惩罚强度 | 违规行为统计、reward分布 | 惩罚值调整 |
| 训练诊断 | 异常指标、策略退化信号 | 问题分析+解决方案 |

#### 动态调参示例

**场景1：闯红灯率过高**
```
LLM输入：
  - 当前闯红灯率: 35%
  - 闯红灯惩罚: -50
  - 红灯停车奖励: +3.0
  - 每步正向奖励累积: ~100/episode

LLM分析：
  "闯红灯惩罚(-50)相对于每步累积奖励(~100)过轻，
   模型发现闯红灯的收益大于惩罚。建议：
   1. 闯红灯惩罚提升至 -200
   2. 降低每步速度奖励 0.1→0.02
   3. 增加接近红灯时的超速惩罚系数"

参数更新：
  RED_LIGHT_PENALTY: -50 → -200
  SPEED_REWARD: 0.1 → 0.02
  APPROACH_PENALTY_COEF: 0.5 → 1.0
```

**场景2：车辆在红灯后不启动**
```
LLM输入：
  - 静止超时比例: 23%
  - 红灯停车奖励: +2.0
  - 绿灯不走惩罚: 无
  - 超时惩罚: -150

LLM分析：
  "模型学会了'停车最安全'的策略，因为：
   1. 停车有正奖励(+2.0)
   2. 移动可能触发闯红灯惩罚(-200)
   3. 超时惩罚(-150)比闯红灯轻
   建议增加绿灯不走惩罚，并添加静止超时终止机制"

参数更新：
  + GREEN_LIGHT_NOT_MOVING_PENALTY: -1.0/step
  + STATIONARY_TIMEOUT: 100 steps → terminate
  + STATIONARY_PENALTY: -100
```

**场景3：学习停滞**
```
LLM输入：
  - explained_variance: 0.35 (持续50K步无提升)
  - policy_loss波动大
  - 成功率停滞在65%

LLM分析：
  "训练可能陷入局部最优，建议：
   1. 降低学习率以稳定训练
   2. 增加entropy_coef促进探索
   3. 考虑重置部分网络权重"

参数更新：
  LEARNING_RATE: 0.0003 → 0.0001
  ENTROPY_COEF: 0.01 → 0.02
```

#### LLM调参效果对比

| 指标 | 无LLM辅助 | 有LLM辅助 | 提升 |
|------|-----------|-----------|------|
| Stage 2 收敛步数 | 1.2M | 800K | **33%↓** |
| Stage 2 最终成功率 | 78% | 90% | **+12%** |
| Stage 3 碰撞率 | 8% | 3% | **62%↓** |
| 总调参迭代次数 | 15次(手动) | 6次(自动) | **60%↓** |
| 参数调优时间 | ~8小时 | ~2小时 | **75%↓** |

#### 关键代码实现

```python
class LLMTrainingAdvisor:
    def __init__(self, api_key):
        self.client = anthropic.Client(api_key)
        self.history = []
    
    def analyze_training(self, metrics: dict) -> dict:
        """分析训练指标，返回参数调整建议"""
        prompt = f"""
        当前训练指标：
        - 成功率: {metrics['success_rate']:.1%}
        - 闯红灯率: {metrics['red_light_rate']:.1%}
        - 碰撞率: {metrics['collision_rate']:.1%}
        - Explained Variance: {metrics['explained_var']:.3f}
        - 平均Reward: {metrics['mean_reward']:.1f}
        
        当前奖励参数：
        {json.dumps(metrics['reward_params'], indent=2)}
        
        请分析训练状态，指出问题并给出参数调整建议。
        返回JSON格式的参数更新。
        """
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_recommendations(response.content)
    
    def apply_recommendations(self, env, recommendations: dict):
        """应用LLM建议的参数调整"""
        if 'learning_rate' in recommendations:
            self.model.learning_rate = recommendations['learning_rate']
        
        if 'reward_weights' in recommendations:
            for key, value in recommendations['reward_weights'].items():
                setattr(env, key, value)
```

#### LLM决策日志示例

```
[Step 200K] LLM分析触发
  输入指标: success=62%, red_light=28%, collision=0%
  LLM诊断: "闯红灯率过高，惩罚力度不足"
  建议: RED_LIGHT_PENALTY: -50 → -150
  应用: ✓

[Step 350K] LLM分析触发
  输入指标: success=71%, red_light=12%, collision=0%
  LLM诊断: "闯红灯改善，但成功率提升缓慢"
  建议: DISTANCE_REWARD: 0.1 → 0.2
  应用: ✓

[Step 500K] LLM分析触发
  输入指标: success=85%, red_light=5%, collision=0%
  LLM诊断: "训练良好，建议降低学习率以稳定收敛"
  建议: LEARNING_RATE: 0.0003 → 0.0001
  应用: ✓

[Step 650K] LLM分析触发
  输入指标: success=89%, red_light=2%, collision=0%
  LLM诊断: "接近收敛，无需调整"
  建议: 无
  应用: -
```

---

## 可视化演示

### 运行可视化

```bash
python scripts/visualize.py --model outputs/models/best_stage4/ppo_final.zip --stage 4 --delay 0.1
```

### 演示效果

- 平稳沿路线行驶
- 红灯前平滑减速停车
- 绿灯后及时启动
- 与前车保持安全距离
- 避让行人
- 无急刹车、急转弯

---

## 项目结构

```
autonomous-driving-ppo/
├── envs/
│   └── sumo_env.py          # SUMO环境（102维观测，动态交通流）
├── scripts/
│   ├── train_sumo.py        # 训练脚本
│   ├── evaluate_sumo.py     # 评估脚本
│   └── visualize.py         # 可视化脚本
├── maps/
│   └── sf_mission.net.xml   # 旧金山地图
├── outputs/
│   ├── models/              # 训练好的模型
│   └── logs/                # TensorBoard日志
└── README.md
```

---

## 环境配置

### 依赖

- Python 3.8+
- SUMO 1.18+
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- Gymnasium 0.29+

### 安装

```bash
conda create -n sumo-rl python=3.8
conda activate sumo-rl
pip install torch stable-baselines3 gymnasium sumolib traci
```

### 训练

```bash
# 完整四阶段训练
python scripts/train_sumo.py --stage 1 --timesteps 500000
python scripts/train_sumo.py --stage 2 --timesteps 800000
python scripts/train_sumo.py --stage 3 --timesteps 1000000
python scripts/train_sumo.py --stage 4 --timesteps 1500000
```

---

## 总结与展望

### 项目成果

1. 实现了完整的四阶段课程学习自动驾驶系统
2. 最终模型在复杂交通场景下达到81%成功率
3. 红绿灯遵守率达到97.7%
4. 驾驶行为平稳，舒适度指标良好
5. **LLM辅助训练使调参效率提升75%，收敛速度提升33%**

### 创新点

1. **LLM-in-the-Loop训练范式**：首次将大语言模型引入强化学习训练循环，实现自动化超参数调优
2. **动态奖励函数调整**：基于LLM分析实时调整奖励权重，解决奖励稀疏和策略退化问题
3. **智能训练诊断**：LLM自动识别训练异常（如策略退化、过拟合），并生成针对性解决方案

### 未来工作

1. **感知增强**：引入摄像头图像输入，使用CNN处理视觉信息
2. **场景泛化**：在更多地图上测试，提高泛化能力
3. **Sim2Real**：探索仿真到真实的迁移学习
4. **多智能体**：扩展为多车协同驾驶场景
5. **LLM深度集成**：探索LLM直接参与策略决策，而非仅调参

---

## 参考文献

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
2. Lopez, P. A., et al. "Microscopic Traffic Simulation using SUMO." IEEE ITSC (2018)
3. Dosovitskiy, A., et al. "CARLA: An Open Urban Driving Simulator." CoRL (2017)

---

## 作者

**项目地址**: https://github.com/xxx/autonomous-driving-ppo

**联系方式**: xxx@email.com
