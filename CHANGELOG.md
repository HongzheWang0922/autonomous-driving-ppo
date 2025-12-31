# 更新日志

## [2.0.0] - 2024-01-15

### 重大更新：从Highway-env迁移到SUMO

本版本完全重构了项目，从highway-env环境迁移到SUMO仿真器，使用真实街道地图进行训练。

---

### 新增功能

#### 核心环境
- **SUMO环境支持** (`envs/sumo_env.py`)
  - 基于SUMO 1.18+的自动驾驶环境
  - 支持真实OpenStreetMap地图
  - 36维观测空间（ego状态、周围车辆、红绿灯、路由）
  - 连续动作空间（加速度、转向）
  - 完整的奖励函数系统

#### 四阶段课程学习
- **Stage 1: 空路导航**
  - 无其他车辆，学习基础驾驶
  - 目标：80%成功率
  
- **Stage 2: 红绿灯遵守**
  - 引入红绿灯系统
  - 学习交通规则
  - 目标：80%成功率 + 低闯红灯率
  
- **Stage 3: 动态避障**
  - 15辆背景车辆
  - 学习车辆交互和避障
  - 目标：80%成功率 + 低碰撞率
  
- **Stage 4: 综合场景**
  - 20辆车辆 + 10个行人
  - 长距离路由（500-1500米）
  - 完整的城市驾驶场景
  - 目标：80%成功率 + 综合安全性

#### LLM训练顾问
- **Gemini AI集成** (`utils/llm_advisor.py`)
  - 每10000 episode自动分析训练数据
  - 问题诊断和优化建议
  - 奖励函数和超参数调整建议
  - 日志保存到 `outputs/llm_logs/`
  - 每天最多200次调用限制
  - 可选启用（`--llm --llm-api-key`）

#### 地图工具
- **地图下载脚本** (`scripts/download_map.py`)
  - 从OpenStreetMap下载真实街道数据
  - 自动转换为SUMO格式（netconvert）
  - 支持3个预定义区域：
    - San Francisco Mission District
    - Manhattan SoHo
    - Manhattan Midtown
  - 自动生成Stage 1-4路由模板

#### 训练和评估
- **新训练脚本** (`scripts/train_sumo.py`)
  - 支持四阶段课程学习
  - 自动从前一阶段加载模型
  - 集成LLM训练顾问
  - Episode统计和Tensorboard记录
  - 定期评估和Checkpoint保存
  
- **新评估脚本** (`scripts/evaluate_sumo.py`)
  - 详细的评估指标
  - 成功率、碰撞率统计
  - 红绿灯违规统计
  - JSON格式结果导出
  - 可选SUMO-GUI可视化

#### 文档
- **完整重写README** (`README.md`)
  - 四阶段训练详细说明
  - LLM训练顾问使用指南
  - 完整的安装和使用说明
  - 常见问题解答
  
- **快速开始指南** (`QUICKSTART.md`)
  - 30分钟上手指南
  - 简化的安装步骤
  
- **详细安装指南** (`INSTALL.md`)
  - 跨平台安装说明
  - 故障排除
  
- **迁移指南** (`MIGRATION_GUIDE.md`)
  - Highway-env到SUMO的迁移说明
  - API变化对比
  - 旧版兼容性说明

---

### 更改

#### 依赖更新
- 添加 `sumolib >= 1.18.0`
- 添加 `traci >= 1.18.0`
- 添加 `sumo-rl >= 1.4.0`
- 添加 `google-generativeai >= 0.3.0` (LLM顾问)
- 更新 `environment.yml`
- 更新 `pyproject.toml`
- 新增 `requirements.txt`

#### 配置文件
- 更新 `.gitignore`
  - 忽略SUMO临时文件
  - 忽略地图文件（需单独下载）
  - 忽略LLM日志

---

### 弃用

以下文件保留但标记为弃用：
- `envs/intersection_env.py` - Highway-env环境
- `scripts/train_multistage.py` - Highway-env训练
- `scripts/evaluate_local.py` - Highway-env评估
- `scripts/visualize_local.py` - Highway-env可视化

**注意**: 这些文件仍可使用，但新项目应使用SUMO版本。

---

### 性能对比

| 指标 | v1.0 (Highway-env) | v2.0 (SUMO) |
|------|-------------------|-------------|
| 环境 | Highway-env | SUMO |
| 地图 | 简单几何 | 真实街道 |
| 训练阶段 | 3 | 4 |
| 并行环境 | 24 | 8-16 |
| Stage 1时间 | ~4小时 | ~8小时 |
| 总训练时间 | ~20小时 | ~70小时 |
| LLM辅助 | 否 | 是 |
| 红绿灯 | 否 | 是 |
| 行人 | 否 | 是 |
| 真实度 | 2星 | 5星 |

---

### 修复

- 解决Highway-env的Length=1 bug（旧环境）
- 改进碰撞检测逻辑
- 优化奖励函数平衡性

---

### 文件结构

```
新增文件:
+ envs/sumo_env.py                 # SUMO环境
+ scripts/train_sumo.py            # SUMO训练
+ scripts/evaluate_sumo.py         # SUMO评估
+ scripts/download_map.py          # 地图下载
+ utils/llm_advisor.py             # LLM顾问
+ maps/                            # 地图目录
+ outputs/llm_logs/                # LLM日志目录
+ README.md                        # 重写
+ QUICKSTART.md                    # 新增
+ INSTALL.md                       # 新增
+ MIGRATION_GUIDE.md               # 新增
+ CHANGELOG.md                     # 本文件
+ requirements.txt                 # 新增
+ .gitignore                       # 更新

保留（弃用）:
~ envs/intersection_env.py         # Highway-env环境
~ scripts/train_multistage.py      # 旧训练脚本
~ scripts/evaluate_local.py        # 旧评估脚本
~ scripts/visualize_local.py       # 旧可视化脚本
```

---

### 升级指南

从v1.0升级到v2.0：

1. **安装SUMO**
   ```bash
   # 参考 INSTALL.md
   ```

2. **更新Python依赖**
   ```bash
   conda env update -f environment.yml
   # 或
   pip install -r requirements.txt
   ```

3. **下载地图**
   ```bash
   python scripts/download_map.py --region sf_mission
   ```

4. **开始新训练**
   ```bash
   python scripts/train_sumo.py --stage 1
   ```

详细说明请查看 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

### 使用建议

1. **首次使用**: 参考 [QUICKSTART.md](QUICKSTART.md)
2. **安装问题**: 参考 [INSTALL.md](INSTALL.md)
3. **从旧版迁移**: 参考 [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
4. **完整文档**: 参考 [README.md](README.md)

---

### 致谢

- SUMO团队提供优秀的交通仿真平台
- OpenStreetMap提供免费地图数据
- Google Gemini提供LLM API
- Stable-Baselines3团队维护强化学习库

---

## [1.0.0] - 2024-01-01

### 初始版本
- Highway-env环境
- PPO算法训练
- 3阶段训练（easy/medium/hard）
- 基础评估和可视化

---

**完整更新内容请查看**: [README.md](README.md)


