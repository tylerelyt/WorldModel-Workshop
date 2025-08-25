import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys

def discretize_state(state, bins):
    """将连续状态离散化为离散状态"""
    cart_pos, cart_vel, pole_angle, pole_vel = state
    
    # 定义状态空间的边界
    cart_pos_bins = np.linspace(-2.4, 2.4, bins)
    cart_vel_bins = np.linspace(-3.0, 3.0, bins)
    pole_angle_bins = np.linspace(-0.5, 0.5, bins)
    pole_vel_bins = np.linspace(-2.0, 2.0, bins)
    
    # 离散化各个状态维度
    cart_pos_idx = np.digitize(cart_pos, cart_pos_bins) - 1
    cart_vel_idx = np.digitize(cart_vel, cart_vel_bins) - 1
    pole_angle_idx = np.digitize(pole_angle, pole_angle_bins) - 1
    pole_vel_idx = np.digitize(pole_vel, pole_vel_bins) - 1
    
    # 确保索引在有效范围内
    cart_pos_idx = np.clip(cart_pos_idx, 0, bins-1)
    cart_vel_idx = np.clip(cart_vel_idx, 0, bins-1)
    pole_angle_idx = np.clip(pole_angle_idx, 0, bins-1)
    pole_vel_idx = np.clip(pole_vel_idx, 0, bins-1)
    
    return cart_pos_idx, cart_vel_idx, pole_angle_idx, pole_vel_idx

def run(is_training=True, render=True, log_details=True):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    
    bins = 10  # 每个状态维度的离散化数量
    
    if is_training:
        q = np.zeros((bins, bins, bins, bins, env.action_space.n))
        print("🚀 开始CartPole Q-Learning训练")
        print("="*50)
    else:
        try:
            with open('cartpole.pkl', 'rb') as f:
                q = pickle.load(f)
            print("🎯 加载已训练的Q表，开始测试")
            print("="*50)
        except FileNotFoundError:
            print("❌ 未找到训练好的模型文件 cartpole.pkl")
            print("💡 请先运行训练模式")
            return
    
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0 if is_training else 0.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    episodes = 1000 if is_training else 5
    rewards_per_episode = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        state_discrete = discretize_state(state, bins)
        total_reward = 0
        step = 0
        
        if log_details:
            if is_training:
                print(f"\n📍 训练回合 {episode+1}/{episodes}")
                print(f"🎲 探索率: {epsilon:.3f}")
            else:
                print(f"\n🎯 测试回合 {episode+1}/{episodes}")
            print(f"🏁 初始状态: {state}")
            print(f"🔢 离散状态: {state_discrete}")
            sys.stdout.flush()
        
        while True:
            step += 1
            
            # 选择动作
            if is_training and np.random.random() < epsilon:
                action = env.action_space.sample()
                if log_details:
                    print(f"🎲 步骤 {step}: 随机探索选择动作 {action} ({'向左推车' if action == 0 else '向右推车'})")
            else:
                q_values = q[state_discrete]
                action = np.argmax(q_values)
                if log_details:
                    print(f"🧠 步骤 {step}: 策略选择动作 {action} ({'向左推车' if action == 0 else '向右推车'})")
                    print(f"   Q值: [向左={q_values[0]:.3f}, 向右={q_values[1]:.3f}]")
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            if render:
                try:
                    env.render()
                except:
                    pass
            
            next_state_discrete = discretize_state(next_state, bins)
            total_reward += reward
            
            if log_details:
                print(f"   ⚡ 奖励: {reward}, 累计奖励: {total_reward}")
                print(f"   🔄 新状态: {next_state}")
                print(f"   🔢 新离散状态: {next_state_discrete}")
                sys.stdout.flush()
            
            # Q-Learning更新
            if is_training:
                old_q_value = q[state_discrete][action]
                max_next_q = np.max(q[next_state_discrete])
                
                # 贝尔曼方程更新
                target_q = reward + discount_factor * max_next_q
                td_error = target_q - old_q_value
                new_q_value = old_q_value + learning_rate * td_error
                
                q[state_discrete][action] = new_q_value
                
                if log_details:
                    print(f"   📊 Q值更新:")
                    print(f"      旧Q值: {old_q_value:.3f}")
                    print(f"      最大下一Q值: {max_next_q:.3f}")
                    print(f"      目标Q值: {target_q:.3f}")
                    print(f"      TD误差: {td_error:.3f}")
                    print(f"      新Q值: {new_q_value:.3f}")
                    sys.stdout.flush()
            
            state = next_state
            state_discrete = next_state_discrete
            
            if terminated or truncated:
                break
        
        rewards_per_episode.append(total_reward)
        
        if log_details:
            print(f"✅ 回合结束 - 总步数: {step}, 总奖励: {total_reward}")
            if episode % 100 == 0 and is_training:
                avg_reward = np.mean(rewards_per_episode[-100:])
                print(f"📈 最近100回合平均奖励: {avg_reward:.2f}")
            sys.stdout.flush()
        
        # 更新epsilon
        if is_training:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    env.close()
    
    # 保存训练结果
    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)
        print(f"\n💾 Q表已保存到 cartpole.pkl")
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode)
        plt.title('Training Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # 移动平均
        window_size = 100
        if len(rewards_per_episode) >= window_size:
            moving_avg = []
            for i in range(len(rewards_per_episode) - window_size + 1):
                moving_avg.append(np.mean(rewards_per_episode[i:i+window_size]))
            
            plt.subplot(1, 2, 2)
            plt.plot(moving_avg)
            plt.title(f'Moving Average Reward (window={window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('cartpole_training.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 训练曲线已保存到 cartpole_training.png")
    
    print(f"\n🎉 {'训练' if is_training else '测试'}完成!")
    print(f"📊 平均奖励: {np.mean(rewards_per_episode):.2f}")

if __name__ == '__main__':
    print("🤖 CartPole Q-Learning 演示")
    print("💡 Q-Learning: 通过试错学习最优策略的强化学习算法")
    print()
    
    # 提供不同的运行模式
    print("选择运行模式:")
    print("1. 快速训练模式 (无可视化)")
    print("2. 可视化训练模式 (有可视化)")
    print("3. 仅测试模式 (需要先训练)")
    
    try:
        mode = int(input("请输入模式编号 (1-3): "))
    except:
        mode = 2  # 默认可视化训练
    
    if mode == 1:
        print("🚀 快速训练模式")
        run(is_training=True, render=False, log_details=False)
    elif mode == 2:
        print("🎮 可视化训练模式")
        run(is_training=True, render=True, log_details=True)
    elif mode == 3:
        print("🎯 测试模式")
        run(is_training=False, render=True, log_details=True)
    else:
        print("❌ 无效模式，使用默认可视化训练模式")
        run(is_training=True, render=True, log_details=True)