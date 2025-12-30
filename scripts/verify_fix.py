#!/usr/bin/env python3
"""
验证Length=1 Bug是否已修复
测试100个episode，统计Length=1的比例
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("🔍 Length=1 Bug修复验证")
print("="*70)

try:
    from envs.intersection_env import IntersectionEnvWrapper
    print("✅ 成功导入环境")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 创建环境
env = IntersectionEnvWrapper(difficulty='easy')

print("\n正在测试100个episode...")
print("如果修复成功，应该几乎没有Length=1的episode\n")

length_1_count = 0
length_distribution = {}
reward_20_length_1 = 0  # 第1步到达
reward_minus10_length_1 = 0  # 第1步碰撞
total_episodes = 100

print("Episode详情:")
print("-" * 70)

for i in range(total_episodes):
    obs, info = env.reset()
    
    # 运行一个episode
    episode_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        if done or truncated:
            episode_length = step + 1
            
            # 统计长度分布
            if episode_length not in length_distribution:
                length_distribution[episode_length] = 0
            length_distribution[episode_length] += 1
            
            # 检查Length=1的情况
            if episode_length == 1:
                length_1_count += 1
                
                if abs(reward - 20.0) < 0.1:
                    reward_20_length_1 += 1
                    reason = "第1步到达❌"
                elif abs(reward - (-10.0)) < 0.1:
                    reward_minus10_length_1 += 1
                    reason = "第1步碰撞❌"
                else:
                    reason = "其他原因❓"
                
                print(f"  {i+1:3d}. Length={episode_length}, Reward={reward:7.2f}, "
                      f"Crashed={info.get('crashed')}, Arrived={info.get('arrived')}, "
                      f"原因:{reason}")
            
            break

env.close()

# 显示统计结果
print("\n" + "="*70)
print("统计结果")
print("="*70)

print(f"\nLength分布:")
for length in sorted(length_distribution.keys()):
    count = length_distribution[length]
    percentage = count / total_episodes * 100
    bar = "█" * int(percentage / 2)
    print(f"  Length {length:2d}: {count:3d} ({percentage:5.1f}%) {bar}")

print(f"\nLength=1 详细统计:")
print(f"  总数: {length_1_count}/{total_episodes} = {length_1_count/total_episodes*100:.1f}%")
print(f"  - 第1步到达: {reward_20_length_1} ({reward_20_length_1/total_episodes*100:.1f}%)")
print(f"  - 第1步碰撞: {reward_minus10_length_1} ({reward_minus10_length_1/total_episodes*100:.1f}%)")

# 判断修复是否成功
print("\n" + "="*70)
if length_1_count < 5:
    print("🎉 修复成功！")
    print("="*70)
    print(f"✅ Length=1的比例 = {length_1_count/total_episodes*100:.1f}% < 5%")
    print(f"✅ 95%以上的episode正常运行")
    print(f"✅ 可以开始正式训练了！")
elif length_1_count < 10:
    print("⚠️ 基本修复，但仍有改进空间")
    print("="*70)
    print(f"⚠️ Length=1的比例 = {length_1_count/total_episodes*100:.1f}%")
    print(f"⚠️ 建议进一步检查环境配置")
else:
    print("❌ 修复失败！")
    print("="*70)
    print(f"❌ Length=1的比例 = {length_1_count/total_episodes*100:.1f}% 仍然很高")
    print(f"❌ 请检查是否正确替换了环境文件")
    print(f"❌ 确认是否清除了__pycache__")

print("="*70)

# 额外检查
print("\n额外检查:")
print("-" * 70)

# 检查has_arrived的时间限制
print("检查1: has_arrived是否有时间限制")
try:
    env2 = IntersectionEnvWrapper(difficulty='easy')
    obs, info = env2.reset()
    
    # 检查第0步时的has_arrived
    has_arrived_at_start = env2.unwrapped.has_arrived
    
    if has_arrived_at_start:
        print("  ❌ 第0步就判定为已到达！时间限制可能未生效！")
    else:
        print("  ✅ 第0步未判定为已到达")
    
    env2.close()
except Exception as e:
    print(f"  ⚠️ 检查失败: {e}")

# 检查初始速度
print("\n检查2: 初始速度")
try:
    env3 = IntersectionEnvWrapper(difficulty='easy')
    obs, info = env3.reset()
    
    initial_speed = env3.unwrapped.vehicle.speed
    print(f"  初始速度: {initial_speed:.2f} m/s")
    
    if initial_speed < 1.0:
        print("  ✅ 初始速度接近0，正确")
    else:
        print(f"  ⚠️ 初始速度 = {initial_speed:.2f}，可能太快")
    
    env3.close()
except Exception as e:
    print(f"  ⚠️ 检查失败: {e}")

print("-" * 70)
