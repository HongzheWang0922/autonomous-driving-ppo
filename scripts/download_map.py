#!/usr/bin/env python3
"""
从OpenStreetMap下载真实地图并转换为SUMO格式
支持San Francisco Mission District和Manhattan区域
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import requests


# 预定义的地图区域
MAP_REGIONS = {
    "sf_mission": {
        "name": "San Francisco - Mission District",
        "bbox": "-122.4241,37.7490,-122.4090,37.7630",  # 西,南,东,北
        "description": "旧金山Mission区核心街区，包含Valencia St和Mission St"
    },
    "manhattan_soho": {
        "name": "Manhattan - SoHo",
        "bbox": "-74.0050,40.7200,-73.9950,40.7280",
        "description": "曼哈顿SoHo区，包含Houston St和Broadway"
    },
    "manhattan_midtown": {
        "name": "Manhattan - Midtown (Small)",
        "bbox": "-73.9850,40.7550,-73.9750,40.7620",
        "description": "曼哈顿中城小区域，包含时代广场周边"
    }
}


def download_osm_map(region_key, output_dir):
    """
    从OpenStreetMap下载地图
    
    Args:
        region_key: 地图区域键（如'sf_mission'）
        output_dir: 输出目录
    """
    if region_key not in MAP_REGIONS:
        print(f"❌ 未知的地图区域: {region_key}")
        print(f"可用区域: {', '.join(MAP_REGIONS.keys())}")
        sys.exit(1)
    
    region = MAP_REGIONS[region_key]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    osm_file = output_dir / f"{region_key}.osm"
    
    print(f"{'='*60}")
    print(f"📍 下载地图: {region['name']}")
    print(f"📦 边界框: {region['bbox']}")
    print(f"💾 保存到: {osm_file}")
    print(f"{'='*60}\n")
    
    # 使用Overpass API下载OSM数据
    bbox = region['bbox']
    overpass_url = "https://overpass-api.de/api/map"
    params = {"bbox": bbox}
    
    try:
        print("⏬ 正在下载OSM数据...")
        response = requests.get(overpass_url, params=params, timeout=120)
        response.raise_for_status()
        
        with open(osm_file, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ 下载成功: {osm_file} ({len(response.content) / 1024:.1f} KB)")
        return osm_file
    
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print(f"\n💡 备选方案：手动下载")
        print(f"1. 访问 https://www.openstreetmap.org/export")
        print(f"2. 输入边界框: {bbox}")
        print(f"3. 导出为 .osm 文件")
        print(f"4. 保存到 {osm_file}")
        sys.exit(1)


def convert_osm_to_sumo(osm_file, output_dir):
    """
    使用netconvert将OSM文件转换为SUMO网络
    
    Args:
        osm_file: OSM文件路径
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    net_file = output_dir / f"{osm_file.stem}.net.xml"
    
    print(f"\n{'='*60}")
    print(f"🔄 转换OSM到SUMO格式")
    print(f"{'='*60}\n")
    
    # netconvert命令
    cmd = [
        "netconvert",
        "--osm-files", str(osm_file),
        "--output-file", str(net_file),
        "--geometry.remove",  # 简化几何形状
        "--ramps.guess",  # 自动识别匝道
        "--junctions.join",  # 合并相近的交叉口
        "--tls.guess-signals",  # 自动添加红绿灯
        "--tls.default-type", "actuated",  # 使用感应式红绿灯
        "--keep-edges.by-vclass", "passenger",  # 只保留汽车道路
        "--remove-edges.isolated",  # 移除孤立边
    ]
    
    try:
        print(f"🔧 执行命令:")
        print(f"   {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ 转换成功: {net_file}")
            print(f"\n📊 SUMO网络统计:")
            # 简单统计节点和边的数量
            with open(net_file, 'r', encoding='utf-8') as f:
                content = f.read()
                edge_count = content.count('<edge ')
                junction_count = content.count('<junction ')
                tls_count = content.count('<tlLogic ')
            print(f"   - 路段数: {edge_count}")
            print(f"   - 交叉口数: {junction_count}")
            print(f"   - 红绿灯数: {tls_count}")
            return net_file
        else:
            print(f"❌ 转换失败:")
            print(result.stderr)
            sys.exit(1)
    
    except FileNotFoundError:
        print(f"❌ 错误: 找不到netconvert命令")
        print(f"\n💡 请确保已安装SUMO:")
        print(f"   - Windows: 从 https://sumo.dlr.de/docs/Downloads.php 下载安装")
        print(f"   - Linux: sudo apt install sumo sumo-tools")
        print(f"   - macOS: brew install sumo")
        print(f"\n   安装后请确保netconvert在PATH中")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        sys.exit(1)


def create_route_files(net_file, output_dir):
    """
    创建基础的路由文件（用于各个训练阶段）
    
    Args:
        net_file: SUMO网络文件
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    
    print(f"\n{'='*60}")
    print(f"📝 创建路由配置模板")
    print(f"{'='*60}\n")
    
    # Stage 1: 空路导航（无其他车辆）
    stage1_rou = output_dir / f"{net_file.stem}_stage1.rou.xml"
    with open(stage1_rou, 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!-- Stage 1: 空路导航 - 只有ego车辆，无其他交通参与者 -->
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- 车辆类型定义 -->
    <vType id="ego_vehicle" accel="2.6" decel="4.5" sigma="0.0" length="5.0" maxSpeed="15.0" color="0,255,0"/>
    
    <!-- Ego车辆 - 路由将在运行时动态生成 -->
    <!-- <vehicle id="ego" type="ego_vehicle" depart="0" color="0,255,0"/> -->
</routes>
''')
    print(f"✅ Stage 1 路由文件: {stage1_rou.name}")
    
    # Stage 2: 加入红绿灯（无其他车辆，但有红绿灯）
    stage2_rou = output_dir / f"{net_file.stem}_stage2.rou.xml"
    with open(stage2_rou, 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!-- Stage 2: 红绿灯遵守 - ego车辆 + 红绿灯，无其他车辆 -->
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="ego_vehicle" accel="2.6" decel="4.5" sigma="0.0" length="5.0" maxSpeed="15.0" color="0,255,0"/>
</routes>
''')
    print(f"✅ Stage 2 路由文件: {stage2_rou.name}")
    
    # Stage 3: 加入其他车辆
    stage3_rou = output_dir / f"{net_file.stem}_stage3.rou.xml"
    with open(stage3_rou, 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!-- Stage 3: 动态避障 - ego车辆 + 其他车辆 + 红绿灯 -->
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="ego_vehicle" accel="2.6" decel="4.5" sigma="0.0" length="5.0" maxSpeed="15.0" color="0,255,0"/>
    <vType id="background_vehicle" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89" color="255,255,0"/>
    
    <!-- 背景车辆将在运行时动态生成 -->
</routes>
''')
    print(f"✅ Stage 3 路由文件: {stage3_rou.name}")
    
    # Stage 4: 加入行人 + 增加距离
    stage4_rou = output_dir / f"{net_file.stem}_stage4.rou.xml"
    with open(stage4_rou, 'w', encoding='utf-8') as f:
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!-- Stage 4: 综合场景 - ego车辆 + 其他车辆 + 行人 + 红绿灯 + 长距离 -->
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="ego_vehicle" accel="2.6" decel="4.5" sigma="0.0" length="5.0" maxSpeed="15.0" color="0,255,0"/>
    <vType id="background_vehicle" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.89" color="255,255,0"/>
    <vType id="pedestrian" vClass="pedestrian" width="0.8" length="0.8" maxSpeed="1.5" color="255,0,0"/>
    
    <!-- 背景车辆和行人将在运行时动态生成 -->
</routes>
''')
    print(f"✅ Stage 4 路由文件: {stage4_rou.name}")
    
    print(f"\n💡 提示: 路由文件是模板，实际的起点和终点将在训练时动态生成")


def main():
    parser = argparse.ArgumentParser(description='下载并转换真实地图为SUMO格式')
    parser.add_argument('--region', type=str, default='sf_mission',
                        choices=list(MAP_REGIONS.keys()),
                        help='地图区域')
    parser.add_argument('--output-dir', type=str, default='../maps',
                        help='输出目录')
    parser.add_argument('--skip-download', action='store_true',
                        help='跳过下载，只转换现有OSM文件')
    
    args = parser.parse_args()
    
    print(f"\n{'🗺️ '*20}")
    print(f"SUMO地图下载和转换工具")
    print(f"{'🗺️ '*20}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 显示可用地图
    print(f"可用地图区域:\n")
    for key, info in MAP_REGIONS.items():
        marker = "👉" if key == args.region else "  "
        print(f"{marker} {key}: {info['name']}")
        print(f"     {info['description']}\n")
    
    # 下载OSM文件
    if not args.skip_download:
        osm_file = download_osm_map(args.region, output_dir)
    else:
        osm_file = output_dir / f"{args.region}.osm"
        if not osm_file.exists():
            print(f"❌ 找不到OSM文件: {osm_file}")
            sys.exit(1)
        print(f"📂 使用现有OSM文件: {osm_file}")
    
    # 转换为SUMO格式
    net_file = convert_osm_to_sumo(osm_file, output_dir)
    
    # 创建路由文件
    create_route_files(net_file, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ 地图准备完成！")
    print(f"{'='*60}")
    print(f"\n📁 输出文件:")
    print(f"   - OSM原始文件: {osm_file.name}")
    print(f"   - SUMO网络文件: {net_file.name}")
    print(f"   - Stage 1-4 路由模板")
    print(f"\n🚀 下一步:")
    print(f"   1. 使用 sumo-gui {net_file.name} 查看网络")
    print(f"   2. 运行训练: python train_multistage.py --stage 1 --map {args.region}")
    print(f"\n")


if __name__ == "__main__":
    main()

