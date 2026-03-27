import torch
import numpy as np
import random
import sys
import os
import wandb # Added for pro-level tracking

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.snake_game import SnakeGame, BLOCK_SIZE, Direction, Point
from env.wrapper import SnakeGameAIWrapper
from src.agent import Agent
from src.algorithms import a_star, is_dead_end, get_hamiltonian_cycle, get_next_hamiltonian_step
from src.helper import plot

# Set up WandB (Optional)
USE_WANDB = False # Set to True to enable professional tracking
if USE_WANDB:
    wandb.init(project="snake-rl-pro", config={
        "learning_rate": 0.001,
        "gamma": 0.9,
        "batch_size": 1000,
        "architecture": "Dueling-DDQN",
        "memory": "PER"
    })

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    wrapper = SnakeGameAIWrapper(game)
    
    # Pre-calculate Hamiltonian cycle
    h_cycle = get_hamiltonian_cycle(game.w, game.h, BLOCK_SIZE)

    print("Starting Pro-Level Training (DDQN + PER + Hamiltonian Guidance)...")
    while True:
        # get old state
        state_old = wrapper.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # Calculate algorithms for reward shaping
        head = (game.head.x, game.head.y)
        food = (game.food.x, game.food.y)
        body = [(p.x, p.y) for p in game.snake]
        
        path = a_star(head, food, game.w, game.h, BLOCK_SIZE, body)
        h_step = get_next_hamiltonian_step(head, h_cycle)
        
        # perform move and get new state
        state_new, reward, done, _, info = wrapper.step(final_move)
        score = info["score"]
        
        # --- PRO REWARD SHAPING ---
        # 1. A* Guidance
        if path and len(path) > 0:
            if (game.head.x, game.head.y) == path[0]:
                reward += 1.5 
        
        # 2. Hamiltonian Failsafe (If A* fails, use Hamiltonian)
        elif h_step:
            if (game.head.x, game.head.y) == h_step:
                reward += 0.5
        
        # 3. DFS Survival Check
        if is_dead_end((game.head.x, game.head.y), game.w, game.h, BLOCK_SIZE, body):
            reward -= 10.0 # Heavy penalty for trapping itself
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Update target network every few episodes (DDQN)
            if agent.n_games % agent.target_update_frequency == 0:
                agent.update_target_model()
                
            game.reset()
            agent.n_games += 1
            loss = agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Episode: {agent.n_games} | Score: {score} | Record: {record} | Loss: {loss:.4f}')

            if USE_WANDB:
                wandb.log({"score": score, "record": record, "loss": loss, "episode": agent.n_games})

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
