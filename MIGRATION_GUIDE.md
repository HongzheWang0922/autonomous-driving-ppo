# 项目重构迁移指南

## 从Highway-env到SUMO的迁移说明

本文档说明了项目从highway-env环境迁移到SUMO的主要变化。

---

## 📋 重构概述

### 主要变化

| 方面 | 旧版 (Highway-env) | 新版 (SUMO) |
|------|-------------------|-------------|
| **仿真器** | Highway-env | SUMO 1.18+ |
| **地图** | 简单几何交叉路口 | OpenStreetMap真实街道 |
| **训练阶段** | 3阶段 (easy/medium/hard) | 4阶段课程学习 |
| **环境数量** | 24个并行 | 8-16个并行 |
| **LLM辅助** | 无 | Gemini训练顾问（可选）|
| **红绿灯** | 无红绿灯观测 | 完整红绿灯系统 |
| **行人** | 无 | Stage 4支持 |

---

## 🆕 新增文件

### 核心环境
- `envs/sumo_env.py` - SUMO环境实现
- `envs/__init__.py` - 更新导出

### 训练脚本
- `scripts/train_sumo.py` - 新的训练脚本
- `scripts/evaluate_sumo.py` - 新的评估脚本
- `scripts/download_map.py` - 地图下载工具

### LLM顾问
- `utils/llm_advisor.py` - LLM训练顾问模块

### 文档
- `README.md` - 完全重写
- `QUICKSTART.md` - 快速开始指南
- `INSTALL.md` - 详细安装指南
- `MIGRATION_GUIDE.md` - 本文档

### 配置
- `environment.yml` - 更新依赖
- `pyproject.toml` - 更新项目配置
- `requirements.txt` - 新增pip依赖文件
- `.gitignore` - 更新忽略规则

---

## 弃用文件

以下文件保留但不再推荐使用：

- `envs/intersection_env.py` - Highway-env环境（旧）
- `scripts/train_multistage.py` - Highway-env训练脚本（旧）
- `scripts/evaluate_local.py` - Highway-env评估脚本（旧）
- `scripts/visualize_local.py` - Highway-env可视化脚本（旧）

**注意**: 这些文件保留是为了向后兼容，但新项目应使用SUMO版本。

---

## API变化

### 环境创建

**旧版**:
```python
from envs.intersection_env import IntersectionEnvWrapper
env = IntersectionEnvWrapper(difficulty="easy")
```

**新版**:
```python
from envs.sumo_env import make_sumo_env
env = make_sumo_env(stage=1, map_name="sf_mission")
```

### 训练命令

**旧版**:
```bash
python scripts/train_multistage.py --stage 1 --n-envs 24 --timesteps 200000
```

**新版**:
```bash
python scripts/train_sumo.py --stage 1 --n-envs 16 --timesteps 500000
```

### 评估命令

**旧版**:
```bash
python scripts/evaluate_local.py --stage 1 --n-episodes 100
```

**新版**:
```bash
python scripts/evaluate_sumo.py --stage 1 --n-episodes 100
```

---

## 训练阶段映射

### 旧版 → 新版

| 旧版 | 新版 | 说明 |
|------|------|------|
| Stage 1 (easy) | Stage 1 (空路导航) | 相似，但使用真实地图 |
| Stage 2 (medium) | Stage 2 (红绿灯) + Stage 3 (避障) | 拆分为两个阶段 |
| Stage 3 (hard) | Stage 4 (综合场景) | 新增行人和长距离 |

### 迁移策略

如果你有旧版训练好的模型：

1. **不建议直接迁移**: 环境差异太大，模型不兼容
2. **建议重新训练**: 使用新的SUMO环境从头训练
3. **可以参考奖励**: 旧版的奖励函数设计可以参考

---

## 🔧 配置变化

### 观测空间

**旧版** (Highway-env):
- 20辆车 × 7特征 = 140维
- 相对坐标系
- Kinematics观测

**新版** (SUMO):
- 36维观测向量:
  - Ego状态: 6维
  - 周围车辆: 8×3 = 24维
  - 红绿灯: 4维
  - 路由信息: 2维

### 动作空间

**旧版**:
- 离散动作 (5个动作)

**新版**:
- 连续动作 (2维)
  - 加速度: [-1, 1] → [-4.5, 2.6] m/s²
  - 转向角: [-1, 1] → [-30, 30] 度

### 奖励函数

基本结构相似，但新增：
- 红绿灯遵守奖励/惩罚
- 更精细的距离感知奖励
- 行人避让奖励 (Stage 4)

---

## 迁移步骤

### 1. 环境准备

```bash
# 1. 安装SUMO
# 参考 INSTALL.md

# 2. 更新Python依赖
conda env update -f environment.yml
# 或
pip install -r requirements.txt
```

### 2. 下载地图

```bash
cd scripts
python download_map.py --region sf_mission
cd ..
```

### 3. 测试新环境

```python
# test_sumo_env.py
from envs.sumo_env import make_sumo_env

env = make_sumo_env(stage=1, map_name="sf_mission", use_gui=True)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
print("SUMO环境测试成功")
```

### 4. 开始新训练

```bash
# Stage 1
python scripts/train_sumo.py --stage 1 --timesteps 500000

# 监控
tensorboard --logdir outputs/logs
```

---

## 🆕 新功能使用

### LLM训练顾问

```bash
# 获取Gemini API Key
# https://makersuite.google.com/app/apikey

# 启用LLM顾问 (Stage 2+)
python scripts/train_sumo.py --stage 2 \
    --llm --llm-api-key YOUR_KEY
```

### 真实地图

```bash
# 使用不同地图
python scripts/download_map.py --region manhattan_soho
python scripts/train_sumo.py --stage 1 --map manhattan_soho
```

### 可视化训练

```bash
# 使用SUMO-GUI查看训练过程（会变慢）
python scripts/train_sumo.py --stage 1 --gui --n-envs 1
```

---

## 注意事项

### 性能差异

- **SUMO更慢**: 比Highway-env慢约2-3倍
- **建议**: 
  - Stage 1-2: 16个并行环境
  - Stage 3-4: 8个并行环境（有背景车辆）

### 成功率标准

- 旧版: 未明确定义
- 新版: 每个Stage要求 ≥80% 成功率

### 训练时间

- 旧版 Stage 1: ~4小时
- 新版 Stage 1: ~8小时（更复杂的环境）

---

## 🐛 常见问题

### Q: 旧模型能否在新环境使用？

**A**: 不能。观测和动作空间完全不同，需要重新训练。

### Q: 为什么要迁移到SUMO？

**A**: 
1. 真实地图，更接近实际场景
2. 完整的交通仿真（红绿灯、行人等）
3. 更好的可扩展性
4. 学术界广泛使用

### Q: 旧版还能用吗？

**A**: 可以，旧文件仍保留。但建议新项目使用SUMO版本。

### Q: LLM顾问必须吗？

**A**: 不必须。LLM顾问是可选功能，不启用也能正常训练。

---

## 性能对比

基于相同硬件 (Ryzen 5600 + 32GB RAM):

| 指标 | Highway-env | SUMO |
|------|-------------|------|
| Stage 1训练时间 | ~4小时 | ~8小时 |
| 每秒steps | ~3000 | ~1200 |
| 并行环境数 | 24 | 8-16 |
| 总训练时间 | ~20小时 | ~70小时 |
| 环境真实度 | 2星 | 5星 |
| 可扩展性 | 2星 | 5星 |

---

## 延伸阅读

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [SUMO-RL](https://github.com/LucasAlegre/sumo-rl)
- [Curriculum Learning Paper](https://arxiv.org/abs/2003.04960)

---

## 🤝 需要帮助？

- 查看 [README.md](README.md) 获取完整文档
- 查看 [QUICKSTART.md](QUICKSTART.md) 快速开始
- 查看 [INSTALL.md](INSTALL.md) 解决安装问题

---

祝迁移顺利！


