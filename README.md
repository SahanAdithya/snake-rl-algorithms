# Algorithmic-Enhanced Snake RL

This project combines Reinforcement Learning (DQN) with classical Graph Algorithms (A*, DFS) to create an intelligent and efficient Snake agent.

## Architecture

- **RL Agent**: A Deep Q-Network (DQN) built with PyTorch that learns to navigate the Snake game.
- **A* Pathfinding**: Used to provide "reward shaping" bonuses. If the agent follows the A* path to the food, it receives an extra reward.
- **DFS Survival Check**: Used to penalize the agent if it enters a "dead end" where it might trap itself.
- **Snake Environment**: A custom Pygame-based environment following a Gymnasium-like wrapper.

## Repository Structure

- `env/`: Contains the Snake game logic and agent wrapper.
- `src/`: Contains the DQN agent, classical algorithms, and the training script.
- `requirements.txt`: Project dependencies.

## How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start Training**:
    ```bash
    python3 src/trainer.py
    ```

## Performance Comparison

- **Pure RL**: Often takes hundreds of games to learn simple navigation.
- **Algorithmic-Enhanced RL**: Learns significantly faster by receiving immediate feedback from A* guidance.
