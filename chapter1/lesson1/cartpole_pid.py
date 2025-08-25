import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def run_pid_control(episodes=10, render=True, save_data=True):
    """
    使用PID控制器控制CartPole平衡
    
    参数说明:
    - episodes: 运行回合数
    - render: 是否显示可视化界面
    - save_data: 是否保存性能数据
    """
    
    # 创建环境
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    # PID控制器参数
    # 根据参考文章的参数进行调整
    p = 0.1      # 比例参数 - 控制角度偏差的响应强度
    i = 0.0001   # 积分参数 - 控制位置偏差的累积响应  
    d = 0.005    # 微分参数 - 控制角速度的响应
    
    # 记录数据
    episode_rewards = []
    episode_lengths = []
    
    print("🎯 CartPole PID控制演示")
    print(f"📊 PID参数: P={p}, I={i}, D={d}")
    print(f"🎮 运行 {episodes} 个回合")
    print("=" * 60)
    
    for episode in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 记录控制信号的历史
        control_signals = []
        observations_history = []
        
        print(f"\n📍 回合 {episode + 1}/{episodes}")
        
        for step in range(5000):  # 最大步数限制
            if render:
                env.render()
                time.sleep(0.01)  # 控制显示速度
            
            # 提取观测值
            # observation[0]: 小车位置 (cart position)
            # observation[1]: 小车速度 (cart velocity) 
            # observation[2]: 杆子角度 (pole angle)
            # observation[3]: 杆子角速度 (pole angular velocity)
            cart_pos = observation[0]
            cart_vel = observation[1] 
            pole_angle = observation[2]
            pole_angular_vel = observation[3]
            
            # PID控制算法
            # 控制信号 = P*角度 + I*位置 + D*角速度 + 位置反馈
            control_signal = (pole_angle * p + 
                            cart_pos * i + 
                            pole_angular_vel * d + 
                            cart_vel * i)
            
            # 根据控制信号决定动作
            if control_signal > 0:
                action = 1  # 向右推
            else:
                action = 0  # 向左推
            
            # 记录数据用于分析
            control_signals.append(control_signal)
            observations_history.append(observation.copy())
            
            # 执行动作
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # 每100步显示一次状态信息
            if step % 100 == 0 and step > 0:
                print(f"  步骤 {step}: 角度={pole_angle:.4f}, 位置={cart_pos:.4f}, "
                      f"控制信号={control_signal:.4f}, 累积奖励={episode_reward}")
                sys.stdout.flush()
            
            # 检查是否结束
            if terminated or truncated:
                break
        
        # 回合总结
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"  📋 回合结束: 持续{episode_length}步, 总奖励={episode_reward}")
        
        # 分析控制效果
        if len(control_signals) > 0:
            avg_control = np.mean(np.abs(control_signals))
            max_control = np.max(np.abs(control_signals))
            print(f"  📊 控制信号: 平均幅度={avg_control:.4f}, 最大幅度={max_control:.4f}")
    
    env.close()
    
    # 统计结果
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = np.sum(np.array(episode_lengths) >= 500) / episodes * 100  # 500步认为成功
    
    print("\n" + "=" * 60)
    print("🏁 PID控制结果统计:")
    print(f"   • 平均奖励: {avg_reward:.2f}")
    print(f"   • 平均步数: {avg_length:.2f}")
    print(f"   • 成功率: {success_rate:.1f}% (≥500步)")
    print(f"   • 最好成绩: {max(episode_lengths)}步")
    print(f"   • PID参数: P={p}, I={i}, D={d}")
    print("=" * 60)
    
    # 保存和可视化数据
    if save_data:
        save_pid_results(episode_rewards, episode_lengths, p, i, d)
    
    return episode_rewards, episode_lengths

def save_pid_results(rewards, lengths, p, i, d):
    """保存PID控制结果并生成图表"""
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 奖励曲线
    ax1.plot(rewards, 'b-', linewidth=2, label='每回合奖励')
    ax1.axhline(y=np.mean(rewards), color='r', linestyle='--', 
                label=f'平均奖励: {np.mean(rewards):.1f}')
    ax1.set_title(f'CartPole PID控制 - 奖励曲线 (P={p}, I={i}, D={d})', fontsize=14)
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 步数曲线
    ax2.plot(lengths, 'g-', linewidth=2, label='每回合步数')
    ax2.axhline(y=np.mean(lengths), color='r', linestyle='--', 
                label=f'平均步数: {np.mean(lengths):.1f}')
    ax2.axhline(y=500, color='orange', linestyle=':', 
                label='成功阈值: 500步')
    ax2.set_title('持续步数曲线', fontsize=14)
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('步数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cartpole_pid_results.png', dpi=150, bbox_inches='tight')
    print(f"📈 结果图表已保存到 cartpole_pid_results.png")

def tune_pid_parameters():
    """PID参数调优实验"""
    
    print("🔧 PID参数调优实验")
    print("=" * 50)
    
    # 测试不同的PID参数组合
    param_sets = [
        {'p': 0.1, 'i': 0.0001, 'd': 0.005, 'name': '参考参数'},
        {'p': 0.15, 'i': 0.0001, 'd': 0.005, 'name': '增大P'},
        {'p': 0.1, 'i': 0.0005, 'd': 0.005, 'name': '增大I'}, 
        {'p': 0.1, 'i': 0.0001, 'd': 0.01, 'name': '增大D'},
        {'p': 0.05, 'i': 0.0001, 'd': 0.005, 'name': '减小P'},
    ]
    
    results = []
    
    for params in param_sets:
        print(f"\n🧪 测试 {params['name']}: P={params['p']}, I={params['i']}, D={params['d']}")
        
        # 临时修改PID参数
        global p, i, d
        p, i, d = params['p'], params['i'], params['d']
        
        # 运行测试
        rewards, lengths = run_pid_control(episodes=5, render=False, save_data=False)
        
        avg_length = np.mean(lengths)
        success_rate = np.sum(np.array(lengths) >= 500) / len(lengths) * 100
        
        results.append({
            'name': params['name'],
            'params': params,
            'avg_length': avg_length,
            'success_rate': success_rate
        })
        
        print(f"  结果: 平均步数={avg_length:.1f}, 成功率={success_rate:.1f}%")
    
    # 显示最佳参数
    best_result = max(results, key=lambda x: x['avg_length'])
    print(f"\n🏆 最佳参数组合: {best_result['name']}")
    print(f"   参数: P={best_result['params']['p']}, I={best_result['params']['i']}, D={best_result['params']['d']}")
    print(f"   性能: 平均步数={best_result['avg_length']:.1f}, 成功率={best_result['success_rate']:.1f}%")

if __name__ == '__main__':
    print("🚀 CartPole PID控制演示")
    print("💡 提示: PID控制器通过比例、积分、微分参数实现杆子平衡控制")
    print("💡 提示: 相比Q-learning，PID控制不需要训练，直接基于物理反馈")
    print()
    
    # 基本演示
    print("📍 基本PID控制演示:")
    run_pid_control(episodes=5, render=True, save_data=True)
    
    # 参数调优实验
    print("\n" + "="*70)
    print("📍 PID参数调优实验:")
    tune_pid_parameters()
