import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys

def run(episodes, is_training=True, render=False, log_details=True):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    
    # 日志相关变量
    action_names = ["左", "下", "右", "上"]
    log_interval = max(1, episodes // 100)  # 每1%的进度输出一次详细日志
    step_count = 0
    
    if log_details and is_training:
        print("=" * 80)
        print("🎯 Q-Learning 算法训练开始")
        print(f"📊 环境: FrozenLake 8x8, 总回合数: {episodes}")
        print(f"🧠 初始参数: α(学习率)={learning_rate_a}, γ(折扣因子)={discount_factor_g}, ε(探索率)={epsilon}")
        print("=" * 80)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        episode_steps = 0
        episode_reward = 0

        # 记录回合开始信息
        show_episode_detail = log_details and is_training and (i % log_interval == 0 or i < 5)
        
        if show_episode_detail:
            print(f"\n📍 回合 {i+1}/{episodes} (进度: {(i+1)/episodes*100:.1f}%)")
            print(f"🎲 当前探索率 ε = {epsilon:.4f}, 学习率 α = {learning_rate_a:.4f}")
            print(f"🗺️  起始位置: 状态 {state} (位置: 行{state//8}, 列{state%8})")

        while(not terminated and not truncated):
            step_count += 1
            episode_steps += 1
            
            # 详细日志 - 步骤开始
            if show_episode_detail and episode_steps <= 15:  # 增加到15步
                print(f"\n  📍 步骤 {episode_steps}: 当前状态 {state} (行{state//8}, 列{state%8})")
                print(f"     当前状态Q值: {q[state,:]}")
                sys.stdout.flush()
            
            # 动作选择策略
            random_value = rng.random() if is_training else 0
            if is_training and random_value < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                action_type = "🎲 随机探索"
                if show_episode_detail and episode_steps <= 15:
                    print(f"     决策依据: 随机数 {random_value:.4f} < ε({epsilon:.4f}) → 随机选择")
                    sys.stdout.flush()
            else:
                action = np.argmax(q[state,:])
                action_type = "🧠 策略选择"
                if show_episode_detail and episode_steps <= 15:
                    best_q_values = q[state,:]
                    print(f"     决策依据: 选择最大Q值动作")
                    for act_idx, (act_name, q_val) in enumerate(zip(action_names, best_q_values)):
                        marker = " ✅" if act_idx == action else ""
                        print(f"       动作'{act_name}': Q={q_val:.4f}{marker}")
                    sys.stdout.flush()
                
            # 记录动作前的Q值
            old_q_value = q[state, action] if is_training else 0
            
            if show_episode_detail and episode_steps <= 15:
                print(f"     ➡️  选择动作: '{action_names[action]}' ({action_type})")
                sys.stdout.flush()

            new_state,reward,terminated,truncated,_ = env.step(action)
            episode_reward += reward

            if is_training:
                # 🎯 奖励塑造: 为冰洞添加负奖励来改善学习
                if terminated and reward == 0.0:  # 掉入冰洞
                    shaped_reward = -0.1  # 负奖励惩罚掉入冰洞
                else:
                    shaped_reward = reward  # 保持原始奖励
                
                # Q-learning 更新公式详细过程
                max_next_q = np.max(q[new_state,:])
                target = shaped_reward + discount_factor_g * max_next_q
                td_error = target - old_q_value
                new_q_value = old_q_value + learning_rate_a * td_error
                q[state,action] = new_q_value
                
                # 详细日志输出 - Q值更新过程
                if show_episode_detail and episode_steps <= 15:
                    print(f"     🔄 状态转移: {state} → {new_state} (行{new_state//8}, 列{new_state%8})")
                    print(f"     🏆 环境奖励: r = {reward}")
                    if shaped_reward != reward:
                        print(f"     🎯 奖励塑造: shaped_r = {shaped_reward} (掉入冰洞惩罚 -0.1)")
                    else:
                        print(f"     🎯 最终奖励: shaped_r = {shaped_reward}")
                    print(f"     📊 Q-Learning 更新计算详细过程:")
                    print(f"       🔹 步骤1: 获取当前Q值")
                    print(f"         Q(s={state}, a={action}) = {old_q_value:.4f}")
                    print(f"       🔹 步骤2: 查看新状态的所有Q值")
                    print(f"         Q({new_state}, :) = {q[new_state,:]}")
                    print(f"       🔹 步骤3: 找到新状态的最大Q值")
                    print(f"         max Q(s'={new_state}, a') = {max_next_q:.4f}")
                    print(f"       🔹 步骤4: 使用贝尔曼方程计算目标值")
                    print(f"         Target = shaped_r + γ × max Q(s', a')")
                    print(f"         Target = {shaped_reward} + {discount_factor_g} × {max_next_q:.4f}")
                    print(f"         Target = {target:.4f}")
                    print(f"       🔹 步骤5: 计算时序差分(TD)误差")
                    print(f"         TD_error = Target - Q(s,a)")
                    print(f"         TD_error = {target:.4f} - {old_q_value:.4f}")
                    print(f"         TD_error = {td_error:.4f}")
                    print(f"       🔹 步骤6: 使用学习率更新Q值")
                    print(f"         Q_new(s,a) = Q_old(s,a) + α × TD_error")
                    print(f"         Q_new({state},{action}) = {old_q_value:.4f} + {learning_rate_a} × {td_error:.4f}")
                    print(f"         Q_new({state},{action}) = {new_q_value:.4f}")
                    print(f"     📋 更新后的Q表状态:")
                    print(f"       当前状态Q值: Q({state},:) = {q[state,:]}")
                    
                    if terminated:
                        if reward > 0:
                            print(f"     🎉 到达目标! 获得奖励 {reward}")
                        else:
                            print(f"     ❄️ 掉入冰洞! 回合结束")
                    
                    print(f"     {'='*70}")
                    # 每个动作步骤后立即刷新输出
                    sys.stdout.flush()
            else:
                if show_episode_detail and episode_steps <= 15:
                    print(f"     🔄 状态转移: {state} → {new_state} (测试模式，不更新Q值)")
                    sys.stdout.flush()

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1
            
        # 回合总结
        if show_episode_detail:
            success_rate = np.sum(rewards_per_episode[:i+1]) / (i+1) * 100
            print(f"  📋 回合总结: {episode_steps}步, 奖励={episode_reward}, 当前成功率={success_rate:.1f}%")
            
        # 进度更新
        if log_details and is_training and i % (episodes // 20) == 0 and i > 0:
            success_rate = np.sum(rewards_per_episode[:i+1]) / (i+1) * 100
            avg_q = np.mean(np.max(q, axis=1))
            print(f"\n📈 训练进度 {i+1}/{episodes} ({(i+1)/episodes*100:.0f}%): "
                  f"成功率={success_rate:.1f}%, 平均Q值={avg_q:.3f}, ε={epsilon:.4f}")
            sys.stdout.flush()  # 强制刷新输出

    env.close()

    # 训练完成总结
    if log_details and is_training:
        final_success_rate = np.sum(rewards_per_episode) / episodes * 100
        total_steps = step_count
        avg_final_q = np.mean(np.max(q, axis=1))
        
        print("\n" + "=" * 80)
        print("🏁 训练完成!")
        print(f"📊 最终统计:")
        print(f"   • 总回合数: {episodes}")
        print(f"   • 总步数: {total_steps}")
        print(f"   • 成功率: {final_success_rate:.2f}%")
        print(f"   • 平均Q值: {avg_final_q:.4f}")
        print(f"   • 最终探索率: {epsilon:.4f}")
        print(f"   • 最终学习率: {learning_rate_a:.4f}")
        
        # 显示学到的最优策略示例
        print(f"\n🧠 学到的策略示例 (前16个状态的最优动作):")
        for row in range(4):
            actions_row = []
            for col in range(4):
                state = row * 8 + col
                best_action = np.argmax(q[state, :])
                actions_row.append(action_names[best_action])
            print(f"   行{row}: {' '.join(f'{action:^4}' for action in actions_row)}")
        print("=" * 80)

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(sum_rewards)
    plt.title('Q-Learning 训练过程 - 移动平均奖励 (100回合窗口)', fontsize=14)
    plt.xlabel('回合数')
    plt.ylabel('累积成功次数 (过去100回合)')
    plt.grid(True, alpha=0.3)
    
    # 添加成功率图
    plt.subplot(2, 1, 2)
    success_rate_curve = np.zeros(episodes)
    for t in range(episodes):
        success_rate_curve[t] = np.sum(rewards_per_episode[:t+1]) / (t+1) * 100
    plt.plot(success_rate_curve, 'r-', linewidth=2)
    plt.title('累积成功率变化', fontsize=14)
    plt.xlabel('回合数')
    plt.ylabel('成功率 (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frozen_lake8x8.png', dpi=150, bbox_inches='tight')
    if log_details:
        print(f"📈 训练图表已保存到 frozen_lake8x8.png")

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()
        if log_details:
            print(f"💾 训练模型已保存到 frozen_lake8x8.pkl")

if __name__ == '__main__':
    # 训练模式示例 - 带详细日志
    print("🚀 启动 Q-Learning 算法演示")
    print("💡 提示: render=True 显示可视化界面，render=False 加快训练速度")
    print("💡 提示: log_details=False 可以关闭详细日志")
    print()
    
    # 你可以调整这些参数:
    # episodes: 训练回合数
    # is_training: True=训练模式, False=测试模式(需要先训练)
    # render: True=显示图形界面, False=无图形界面(更快)
    # log_details: True=显示详细日志, False=静默运行
    
    # 🎮 方案1: 带可视化界面的训练 (速度较慢但能看到agent移动)
    run(episodes=100, is_training=True, render=True, log_details=True)
    
    # 🚀 方案2: 快速训练 (取消注释下面这行，注释上面那行)
    # run(episodes=1000, is_training=True, render=False, log_details=True)
    
    # 训练完成后的测试运行 (可选)
    print("\n" + "="*50)
    print("🎮 测试训练好的智能体 (运行3回合)")
    run(episodes=3, is_training=False, render=True, log_details=True)
