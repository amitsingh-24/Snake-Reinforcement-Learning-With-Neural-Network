# Snake-Reinforcement-Learning-With-Neural-Network

# Q-Learning with Neural Network for Snake Game

This project implements a Snake game powered by a reinforcement learning agent using Q-Learning with a neural network to approximate the Q-values. The agent learns to play Snake by interacting with the game environment and improving its strategy over time.

## File Structure
- **`QLearning_Neural_Network.py`**: Implements the Q-Learning algorithm with a neural network, training logic, and visualization.
- **`Snake.py`**: Contains the Snake game environment, including game logic, board representation, and state transitions.
- **`README.md`**: Project documentation.

## Features
- **Custom Snake Environment**: A self-contained implementation of the Snake game, allowing integration with reinforcement learning agents.
- **Q-Learning with Neural Network**: The agent uses a neural network to predict Q-values, enabling efficient decision-making.
- **Replay Buffer**: Experience replay is used to improve the stability and efficiency of the training process.
- **Visualization**: Animations of the agent's performance during training are generated for evaluation purposes.

---

## Usage

### Train the Agent
Run the following command to train the Q-Learning agent:

```bash
python QLearning_Neural_Network.py
