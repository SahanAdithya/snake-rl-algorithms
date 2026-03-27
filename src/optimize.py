import optuna
import torch
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.snake_game import SnakeGame
from env.wrapper import SnakeGameAIWrapper
from src.agent import Agent, QTrainer, LR, MAX_MEMORY

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1000])

    # Initialize agent and game
    agent = Agent()
    agent.trainer.lr = lr
    agent.gamma = gamma
    # We can override agent's internal BATCH_SIZE if needed for the trial
    
    game = SnakeGame()
    wrapper = SnakeGameAIWrapper(game)
    
    total_score = 0
    n_episodes = 20 # Run for 20 episodes to evaluate

    for _ in range(n_episodes):
        done = False
        while not done:
            state_old = wrapper.get_state()
            action = agent.get_action(state_old)
            state_new, reward, done, _, info = wrapper.step(action)
            agent.remember(state_old, action, reward, state_new, done)
            agent.train_short_memory(state_old, action, reward, state_new, done)

        total_score += info["score"]
        game.reset()
        agent.train_long_memory()
        agent.n_games += 1

    return total_score / n_episodes

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)
