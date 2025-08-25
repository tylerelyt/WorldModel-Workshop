# Reinforcement Learning Control Algorithms 🤖

This project contains implementations and demonstrations of various control algorithms, covering classical control theory and modern reinforcement learning methods.

## 📚 Project Structure

### Chapter 1: PID Control
- **CartPole PID Control**: Classical control theory approach (`cartpole_pid.py`)

### Chapter 2: Q-Learning
- **FrozenLake Q-Learning**: Detailed learning process visualization (`q_learning_demo.py`)
- **CartPole Q-Learning**: Tabular reinforcement learning algorithm (`cartpole_q_learning.py`)

## 🚀 Quick Start

### Requirements
```bash
pip install gymnasium pygame numpy matplotlib
```

### Run Demos
```bash
# Chapter 1: PID Control
cd chapter1/lesson1
python cartpole_pid.py

# Chapter 2: Q-Learning
cd chapter2/lesson1
python cartpole_q_learning.py
python frozenlake_q_learning.py
```

## 🎯 Algorithm Comparison

| Algorithm | Learning Method | Training Time | Stability | Interpretability |
|-----------|----------------|---------------|-----------|------------------|
| PID Control | No training | Instant | Immediate | High |
| Q-Learning | Trial & Error | Long | Gradual improvement | Medium |

## 📈 Key Features

- 🎮 **Real-time Visualization**: Observe agent learning process
- 📊 **Detailed Logging**: Understand algorithm decision logic  
- 🔧 **Parameter Tuning**: Experiment with different algorithm parameters
- 📋 **Performance Analysis**: Automatic training curve generation

## 🔬 Learning Focus

1. **Control Theory Basics**: PID control physical intuition
2. **Reinforcement Learning Fundamentals**: States, actions, rewards, policies
3. **Exploration vs Exploitation**: ε-greedy strategy balance
4. **Q-value Updates**: Practical application of Bellman equation

## 🌍 Future Development

### World Model Development
- **Latent World Models**: Learn compressed representations of environment dynamics
- **Transformer-based World Models**: Sequence modeling for long-horizon planning
- **Multi-modal World Models**: Vision, language, and action integration

### Meta Learning & Transfer
- **Meta-RL**: Few-shot adaptation to new environments and tasks
- **Real2Sim**: Transfer real-world data to simulation environments
- **Sim2Real**: Bridge simulation-to-reality gap for robust deployment

### Embodied Intelligence
- **Embodied AI**: Integration with robotic simulation environments
- **Spatial Reasoning**: 3D environment understanding and navigation
- **Manipulation Skills**: Object interaction and tool use

## 📖 Educational Features

This project emphasizes educational value:
- Step-by-step detailed algorithm process logs
- Visualized learning process demonstrations
- Comparative analysis of different algorithms
- Clear documentation and comments

## 🤝 Contributing

Contributions are welcome! Please see contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.