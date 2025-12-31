# 快速开始指南

这是一个简化的快速开始指南，帮助你在30分钟内开始训练。

## 步骤1: 安装SUMO (5分钟)

### Windows
1. 下载 https://sumo.dlr.de/releases/1.18.0/sumo-win64-1.18.0.msi
2. 安装到 `C:\Program Files (x86)\Eclipse\Sumo`
3. 设置环境变量:
```cmd
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
```

### Linux (Ubuntu/Debian)
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

### macOS
```bash
brew install sumo
echo 'export SUMO_HOME=/usr/local/share/sumo' >> ~/.zshrc
source ~/.zshrc
```

## 步骤2: 安装Python依赖 (5分钟)

```bash
# 克隆/进入项目目录
cd autonomous-driving-ppo

# 创建conda环境 (推荐)
conda env create -f environment.yml
conda activate rl-driving-sumo

# 或使用pip
pip install -e .
```

## 步骤3: 下载地图 (2分钟)

```bash
cd scripts
python download_map.py --region sf_mission
cd ..
```

## 步骤4: 开始训练 Stage 1 (立即开始)

```bash
python scripts/train_sumo.py --stage 1 --timesteps 500000 --n-envs 16
```

**训练时间**: ~8小时 (Ryzen 5600)

## 步骤5: 监控训练

打开新终端：
```bash
tensorboard --logdir outputs/logs
```

访问 http://localhost:6006

## 步骤6: 评估模型

训练完成后：
```bash
python scripts/evaluate_sumo.py --stage 1 --n-episodes 100
```

如果成功率 ≥ 80%，进入Stage 2：
```bash
python scripts/train_sumo.py --stage 2 --timesteps 800000 --n-envs 16
```

## 启用LLM训练顾问 (可选)

1. 获取Gemini API Key: https://makersuite.google.com/app/apikey
2. 训练时添加参数:
```bash
python scripts/train_sumo.py --stage 2 \
    --llm --llm-api-key YOUR_API_KEY
```

## 常见问题

### SUMO未找到
```bash
# 检查SUMO_HOME
echo $SUMO_HOME  # Linux/macOS
echo %SUMO_HOME%  # Windows

# 测试SUMO
sumo --version
```

### 地图下载失败
使用备选方法：
1. 访问 https://www.openstreetmap.org/export
2. 选择区域：San Francisco Mission District
3. 导出为 .osm 文件
4. 保存到 `maps/sf_mission.osm`
5. 运行: `python scripts/download_map.py --skip-download --region sf_mission`

### 内存不足
减少并行环境数：
```bash
python scripts/train_sumo.py --stage 1 --n-envs 4
```

---

就这么简单！开始你的自动驾驶之旅吧！

完整文档请查看 [README.md](README.md)


