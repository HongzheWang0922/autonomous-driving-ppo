# 基于SUMO的自动驾驶强化学习项目

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SUMO 1.18+](https://img.shields.io/badge/SUMO-1.18+-green.svg)](https://sumo.dlr.de/)
[![PPO](https://img.shields.io/badge/RL-PPO-orange.svg)](https://github.com/DLR-RM/stable-baselines3)

使用 **Proximal Policy Optimization (PPO)** 算法和 **SUMO仿真器**，在真实美国街道地图上训练自动驾驶Agent。采用**四阶段课程学习**策略，从简单到复杂渐进式训练，并可选集成 **Gemini LLM训练顾问**进行智能训练优化。

---

## 项目概述

本项目重构自highway-env环境，迁移到SUMO仿真器，使用真实的OpenStreetMap数据，实现更接近实际的自动驾驶训练环境。

### 主要特性

- **真实地图**: 使用San Francisco或Manhattan真实街道网络（OpenStreetMap）
- 📚 **四阶段课程学习**: 从空路导航到复杂场景渐进式训练
- 🤖 **LLM训练顾问**: 可选的Gemini AI自动分析训练问题并提供建议
- **SUMO仿真**: 高精度的交通仿真，支持红绿灯、行人等
- ⚡ **并行训练**: 支持8-16个并行环境，充分利用CPU资源
- **完整监控**: Tensorboard实时监控 + LLM定期诊断

---

## 📋 四阶段训练设计

采用课程学习(Curriculum Learning)策略，每个阶段在前一阶段基础上增加难度。**成功率达到80%后进入下一阶段**。

### Stage 1: 空路导航
**目标**: 学习基础驾驶技能 - 不偏离车道、成功到达终点

- 无其他车辆
- 无红绿灯干扰
- 路由长度: 200-500米
- **通过标准**: 成功率 ≥ 80%

**训练命令**:
```bash
python scripts/train_sumo.py --stage 1 --timesteps 500000
```

---

### Stage 2: 红绿灯遵守 🚦
**目标**: 学习遵守交通信号

- 引入红绿灯系统
- 仍无其他车辆（专注学习红绿灯规则）
- 路由长度: 200-500米
- **通过标准**: 成功率 ≥ 80% + 闯红灯率 < 5%

**训练命令** (推荐启用LLM顾问):
```bash
python scripts/train_sumo.py --stage 2 --timesteps 800000 \
    --llm --llm-api-key YOUR_GEMINI_API_KEY
```

---

### Stage 3: 动态避障
**目标**: 学习与其他车辆交互和避障

- 引入15辆背景车辆
- 红绿灯系统
- 复杂的交通场景
- 路由长度: 200-500米
- **通过标准**: 成功率 ≥ 80% + 碰撞率 < 10%

**训练命令**:
```bash
python scripts/train_sumo.py --stage 3 --timesteps 1000000 \
    --llm --llm-api-key YOUR_GEMINI_API_KEY
```

---

### Stage 4: 综合场景
**目标**: 掌握完整的城市驾驶能力

- 20辆背景车辆
- 10个行人
- 红绿灯系统
- **长距离路由**: 500-1500米
- **通过标准**: 成功率 ≥ 80% + 综合安全性

**训练命令**:
```bash
python scripts/train_sumo.py --stage 4 --timesteps 1500000 \
    --llm --llm-api-key YOUR_GEMINI_API_KEY
```

---

## 🤖 LLM训练顾问

从Stage 2开始，可选启用**Gemini LLM训练顾问**，自动分析训练数据并提供优化建议。

### 功能特性

- **自动分析**: 每10000 episode分析训练统计数据
- **问题诊断**: 识别训练问题（闯红灯、碰撞频繁、探索不足等）
- **智能建议**: 提供奖励函数和超参数调整建议
- **日志保存**: 所有建议保存到 `outputs/llm_logs/`
- **用量控制**: 每天最多200次调用，防止超额

### 启用方法

```bash
# 方法1: 命令行参数
python scripts/train_sumo.py --stage 2 \
    --llm \
    --llm-api-key YOUR_GEMINI_API_KEY

# 方法2: 环境变量
export GEMINI_API_KEY=YOUR_GEMINI_API_KEY
python scripts/train_sumo.py --stage 2 --llm
```

### 获取Gemini API Key

1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建API Key
3. 使用 `--llm-api-key` 参数传入

### LLM顾问输入/输出

**输入数据** (最近1000个episode):
- 成功率、碰撞率
- 平均reward、reward方差
- 违规事件统计（闯红灯、偏离路由等）
- 当前训练步数和episode数

**输出建议**:
- 问题诊断
- 奖励权重调整建议（具体数值）
- 超参数调整建议
- 训练策略建议

### 示例LLM建议日志

```json
{
  "episode": 10000,
  "training_steps": 250000,
  "timestamp": "2024-01-15T10:30:00",
  "statistics": {
    "success_rate": 65.5,
    "collision_rate": 12.3,
    "red_light_violations_per_episode": 0.8
  },
  "llm_response": "问题诊断：闯红灯频率较高...\n建议：将红绿灯违规惩罚从-5调整到-10..."
}
```

---

## 快速开始

### 1. 环境准备

#### 安装SUMO

**Windows**:
1. 下载 [SUMO 1.18+](https://sumo.dlr.de/docs/Downloads.php)
2. 安装后设置环境变量:
```cmd
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

**Linux**:
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo
```

**macOS**:
```bash
brew install sumo
export SUMO_HOME=/usr/local/share/sumo
```

#### 安装Python依赖

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate rl-driving-sumo

# 或使用pip
pip install -e .
```

---

### 2. 下载地图

从OpenStreetMap下载真实街道地图并转换为SUMO格式：

```bash
cd scripts

# San Francisco Mission District (推荐)
python download_map.py --region sf_mission

# Manhattan SoHo
python download_map.py --region manhattan_soho

# Manhattan Midtown
python download_map.py --region manhattan_midtown
```

**输出文件**:
- `maps/sf_mission.osm` - OpenStreetMap原始数据
- `maps/sf_mission.net.xml` - SUMO网络文件
- `maps/sf_mission_stage1-4.rou.xml` - 各阶段路由模板

---

### 3. 开始训练

#### Stage 1: 空路导航

```bash
python scripts/train_sumo.py --stage 1 --timesteps 500000 --n-envs 16
```

#### Stage 2-4: 启用LLM顾问

```bash
# Stage 2: 红绿灯
python scripts/train_sumo.py --stage 2 --timesteps 800000 \
    --n-envs 16 \
    --llm --llm-api-key YOUR_KEY

# Stage 3: 动态避障
python scripts/train_sumo.py --stage 3 --timesteps 1000000 \
    --n-envs 8 \
    --llm --llm-api-key YOUR_KEY

# Stage 4: 综合场景
python scripts/train_sumo.py --stage 4 --timesteps 1500000 \
    --n-envs 8 \
    --llm --llm-api-key YOUR_KEY
```

---

### 4. 评估模型

```bash
# 评估Stage 1
python scripts/evaluate_sumo.py --stage 1 --n-episodes 100

# 评估Stage 2 (使用GUI查看)
python scripts/evaluate_sumo.py --stage 2 --n-episodes 20 --render

# 使用自定义模型
python scripts/evaluate_sumo.py --stage 3 \
    --model outputs/models/best_stage3/ppo_500000_steps.zip \
    --n-episodes 100
```

**评估输出**:
- 成功率 (到达终点)
- 碰撞率
- 平均reward
- 红绿灯违规统计
- 路由完成度

---

### 5. 监控训练

#### Tensorboard

```bash
tensorboard --logdir outputs/logs --reload_interval 5
```

访问 `http://localhost:6006` 查看：
- Episode reward曲线
- 成功率/碰撞率
- 红绿灯违规率
- 各项奖励分量

#### LLM训练日志

```bash
# 查看LLM建议历史
ls outputs/llm_logs/

# 查看最新建议
cat outputs/llm_logs/stage2_episode10000_*.json
```

---

## 📁 项目结构

```
autonomous-driving-ppo/
├── envs/                           # 环境定义
│   ├── sumo_env.py                # SUMO环境核心 (新)
│   ├── intersection_env.py        # Highway-env环境 (旧，已弃用)
│   └── __init__.py
│
├── agents/                         # Agent相关 (预留)
│   └── __init__.py
│
├── utils/                          # 工具模块
│   ├── llm_advisor.py             # LLM训练顾问 (新)
│   ├── callbacks.py               # 训练回调
│   ├── reward_logger.py           # 奖励日志
│   └── __init__.py
│
├── scripts/                        # 脚本
│   ├── download_map.py            # 地图下载工具 (新)
│   ├── train_sumo.py              # SUMO训练脚本 (新)
│   ├── evaluate_sumo.py           # SUMO评估脚本 (新)
│   ├── train_multistage.py       # Highway-env训练 (旧)
│   └── evaluate_local.py          # Highway-env评估 (旧)
│
├── maps/                           # 地图文件 (新)
│   ├── sf_mission.osm
│   ├── sf_mission.net.xml
│   └── sf_mission_stage*.rou.xml
│
├── outputs/                        # 输出文件
│   ├── models/                    # 训练模型
│   │   ├── best_stage1/
│   │   ├── best_stage2/
│   │   ├── best_stage3/
│   │   └── best_stage4/
│   ├── logs/                      # Tensorboard日志
│   │   ├── stage1/
│   │   ├── stage2/
│   │   ├── stage3/
│   │   └── stage4/
│   └── llm_logs/                  # LLM顾问日志 (新)
│
├── environment.yml                 # Conda环境
├── pyproject.toml                  # 项目配置
└── README.md                       # 本文档
```

---

## 🔧 高级配置

### 训练参数

```bash
python scripts/train_sumo.py \
    --stage 2 \
    --map sf_mission \
    --timesteps 1000000 \
    --n-envs 16 \
    --device cpu \
    --eval-freq 10000 \
    --checkpoint-freq 50000 \
    --llm --llm-api-key YOUR_KEY
```

**参数说明**:
- `--stage`: 训练阶段 (1-4)
- `--map`: 地图名称
- `--timesteps`: 训练步数
- `--n-envs`: 并行环境数 (推荐: Stage1-2=16, Stage3-4=8)
- `--device`: 训练设备 (cpu/cuda)
- `--eval-freq`: 评估频率
- `--checkpoint-freq`: Checkpoint保存频率
- `--llm`: 启用LLM顾问
- `--llm-api-key`: Gemini API Key
- `--from-checkpoint`: 从checkpoint继续训练

### 硬件要求

**推荐配置** (本项目优化):
- CPU: Ryzen 5600 或更好
- RAM: 32GB
- 并行环境: 8-16个

**最低配置**:
- CPU: 4核心
- RAM: 16GB
- 并行环境: 4个

### PPO超参数

默认超参数 (可在代码中修改):
```python
learning_rate=3e-4
n_steps=2048
batch_size=64
n_epochs=10
gamma=0.99
gae_lambda=0.95
clip_range=0.2
ent_coef=0.01
```

---

## 奖励函数设计

### Stage 1-2: 基础奖励

```python
reward = (
    + 50.0  if goal_reached           # 到达目标
    - 20.0  if collision              # 碰撞
    + 0.05 * distance_progress        # 前进奖励
    + 0.5   if optimal_speed          # 速度奖励
    + 0.1   if on_route               # 保持路由
    - 0.01  per_step                  # 时间惩罚
)
```

### Stage 2+: 红绿灯奖励

```python
if red_light:
    + 0.5  if stopped                 # 停车等待
    - 5.0  if moving (violation)      # 闯红灯惩罚
```

### Stage 3-4: 安全奖励

```python
# 根据与其他车辆距离
distance_to_vehicles > 15m: +0.5
distance_to_vehicles < 5m:  -2.0

# 碰撞风险
time_to_collision < 2s: -1.0
```

---

## 🎓 课程学习策略

### 为什么使用课程学习？

1. **降低训练难度**: 从简单任务开始，逐步增加复杂度
2. **提高样本效率**: 避免在复杂场景中浪费探索时间
3. **稳定性**: 每个阶段的知识迁移到下一阶段
4. **可解释性**: 容易定位问题阶段

### 阶段进入标准

每个阶段达到 **80%成功率** 后进入下一阶段：

```bash
# 评估当前阶段
python scripts/evaluate_sumo.py --stage N --n-episodes 100

# 如果成功率 >= 80%，开始下一阶段
python scripts/train_sumo.py --stage N+1
```

### 知识迁移

- Stage 1 模型 → Stage 2 初始化
- Stage 2 模型 → Stage 3 初始化
- Stage 3 模型 → Stage 4 初始化

---

## 🐛 常见问题

### 1. SUMO相关

**Q: 找不到SUMO_HOME**
```bash
# Windows
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo

# Linux/macOS
export SUMO_HOME=/usr/share/sumo
```

**Q: netconvert命令不存在**

确保SUMO安装完整，netconvert应该在 `$SUMO_HOME/bin/`

### 2. 训练相关

**Q: 并行环境启动失败**

降低并行环境数：
```bash
python scripts/train_sumo.py --stage 1 --n-envs 4
```

**Q: 成功率一直很低**

1. 检查奖励函数是否合理
2. 增加训练时间
3. 启用LLM顾问获取建议
4. 降低环境难度（减少背景车辆）

### 3. LLM相关

**Q: LLM调用失败**

1. 检查API Key是否正确
2. 检查网络连接
3. 查看API配额是否用完

**Q: LLM建议不准确**

LLM建议仅供参考，需要结合实际情况判断。

---

## 📈 预期训练时间

基于 Ryzen 5600 + 32GB RAM + 16并行环境：

| Stage | 训练步数 | 预计时间 | 成功率目标 |
|-------|---------|---------|-----------|
| Stage 1 | 500K | ~8小时 | ≥80% |
| Stage 2 | 800K | ~12小时 | ≥80% |
| Stage 3 | 1M | ~20小时 | ≥80% |
| Stage 4 | 1.5M | ~30小时 | ≥80% |

**总计**: ~70小时 (约3天)

---

## 📚 参考资料

### 论文
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Curriculum Learning for Reinforcement Learning](https://arxiv.org/abs/2003.04960)

### 工具
- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [Gemini API](https://ai.google.dev/)

---

## 🤝 贡献

欢迎提出Issue和Pull Request！

### 开发计划

- [ ] 支持更多地图区域
- [ ] 实现端到端视觉输入
- [ ] 添加更多评估指标
- [ ] 优化SUMO仿真性能
- [ ] 实现自动超参数调优

---

## 📄 License

MIT License

---

## 🙏 致谢

- **SUMO**: 优秀的交通仿真平台
- **Stable-Baselines3**: 强大的强化学习库
- **OpenStreetMap**: 免费的地图数据
- **Google Gemini**: LLM训练顾问支持

---

**作者**: [Your Name]  
**联系**: [Your Email]  
**最后更新**: 2024-01

---

## 开始你的自动驾驶之旅！

```bash
# 1. 安装依赖
conda env create -f environment.yml
conda activate rl-driving-sumo

# 2. 下载地图
python scripts/download_map.py --region sf_mission

# 3. 开始训练
python scripts/train_sumo.py --stage 1

# 4. 启用LLM顾问 (Stage 2+)
python scripts/train_sumo.py --stage 2 --llm --llm-api-key YOUR_KEY
```

Happy Training!
