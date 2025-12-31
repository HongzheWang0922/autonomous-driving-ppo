# 📦 详细安装指南

## 系统要求

### 最低配置
- CPU: 4核心
- RAM: 16GB
- 磁盘: 10GB
- 操作系统: Windows 10+, Ubuntu 20.04+, macOS 11+

### 推荐配置
- CPU: Ryzen 5600 或更好 (6核心+)
- RAM: 32GB
- 磁盘: 20GB SSD
- 操作系统: Windows 11, Ubuntu 22.04, macOS 13+

## 1. SUMO安装

### Windows

#### 方法1: 安装包 (推荐)
1. 下载最新版本: https://sumo.dlr.de/docs/Downloads.php
2. 运行安装程序，安装到默认路径
3. 设置环境变量:
   - 右键"此电脑" → "属性" → "高级系统设置" → "环境变量"
   - 新建系统变量: 
     - 变量名: `SUMO_HOME`
     - 变量值: `C:\Program Files (x86)\Eclipse\Sumo`
   - 添加到PATH: `%SUMO_HOME%\bin`
4. 重启命令行，验证:
```cmd
sumo --version
netconvert --version
```

#### 方法2: Chocolatey
```powershell
choco install sumo
```

### Linux (Ubuntu/Debian)

#### 方法1: PPA (推荐)
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# 设置环境变量
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
echo 'export PATH=$PATH:$SUMO_HOME/tools' >> ~/.bashrc
source ~/.bashrc

# 验证
sumo --version
```

#### 方法2: 源码编译
```bash
sudo apt-get install cmake python3 g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev

git clone --recursive https://github.com/eclipse/sumo
cd sumo
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

export SUMO_HOME=/usr/local/share/sumo
```

### macOS

#### Homebrew
```bash
brew install sumo

# 设置环境变量
echo 'export SUMO_HOME=/usr/local/share/sumo' >> ~/.zshrc
echo 'export PATH=$PATH:$SUMO_HOME/tools' >> ~/.zshrc
source ~/.zshrc

# 验证
sumo --version
```

## 2. Python环境

### 方法1: Conda (推荐)

```bash
# 安装Miniconda (如果未安装)
# https://docs.conda.io/en/latest/miniconda.html

# 创建环境
cd autonomous-driving-ppo
conda env create -f environment.yml

# 激活环境
conda activate rl-driving-sumo

# 验证
python -c "import traci; print('SUMO Python API OK')"
python -c "import stable_baselines3; print('SB3 OK')"
```

### 方法2: pip + venv

```bash
# 创建虚拟环境
python3.10 -m venv venv

# 激活环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -e .

# 或直接从requirements
pip install -r requirements.txt
```

### 方法3: 手动安装

```bash
pip install stable-baselines3[extra]>=2.0.0
pip install gymnasium>=0.28.0
pip install sumolib>=1.18.0
pip install traci>=1.18.0
pip install tensorboard>=2.13.0
pip install torch>=2.0.0
pip install google-generativeai>=0.3.0
pip install numpy matplotlib opencv-python requests tqdm
```

## 3. 验证安装

运行测试脚本:

```bash
cd autonomous-driving-ppo

# 测试Python环境
python -c "
import sys
print(f'Python: {sys.version}')

import stable_baselines3
print(f'SB3: {stable_baselines3.__version__}')

import gymnasium
print(f'Gymnasium: {gymnasium.__version__}')

import traci
print('TraCI: OK')

import torch
print(f'PyTorch: {torch.__version__}')

print('\\nPython环境正常！')
"

# 测试SUMO
python -c "
import os
import sys
print(f'SUMO_HOME: {os.environ.get(\"SUMO_HOME\", \"未设置\")}')

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
    print(f'sumolib: OK')
    print('\\nSUMO环境正常！')
else:
    print('\\n请设置SUMO_HOME环境变量')
"
```

## 4. 可选组件

### LLM训练顾问

```bash
pip install google-generativeai>=0.3.0
```

获取API Key: https://makersuite.google.com/app/apikey

### GPU支持 (可选)

如果有NVIDIA GPU，安装CUDA版PyTorch:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证:
```python
import torch
print(torch.cuda.is_available())  # 应该返回True
```

## 5. 常见安装问题

### SUMO_HOME未设置

**Windows**:
```cmd
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
# 重启命令行
```

**Linux/macOS**:
```bash
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
source ~/.bashrc
```

### TraCI导入失败

确保 `$SUMO_HOME/tools` 在Python路径中:
```python
import os, sys
sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
```

### PyTorch安装慢

使用清华镜像:
```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 依赖冲突

使用conda可以避免大部分冲突:
```bash
conda env create -f environment.yml --force
```

## 6. 卸载

### 卸载Python环境

```bash
# Conda
conda deactivate
conda env remove -n rl-driving-sumo

# venv
deactivate
rm -rf venv
```

### 卸载SUMO

**Windows**: 
- 控制面板 → 程序和功能 → 卸载SUMO

**Linux**:
```bash
sudo apt-get remove sumo sumo-tools
```

**macOS**:
```bash
brew uninstall sumo
```

## 7. 升级

### 升级Python包

```bash
pip install --upgrade stable-baselines3 gymnasium torch
```

### 升级SUMO

按照上述安装步骤重新安装最新版本。

---

安装完成后，继续查看 [快速开始指南](QUICKSTART.md)


