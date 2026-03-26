import torch
import numpy as np
import random
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.snake_game import SnakeGame, BLOCK_SIZE, Direction, Point
from env.wrapper import SnakeGameAIWrapper
from src.agent import Agent
from src.algorithms import a_star, is_dead_end

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    wrapper = SnakeGameAIWrapper(game)

    print("Starting training...")
    while True:
        # get old state
        state_old = wrapper.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # Calculate A* guidance and DFS survival check for reward shaping
        head = (game.head.x, game.head.y)
        food = (game.food.x, game.food.y)
        body = [(p.x, p.y) for p in game.snake]
        
        path = a_star(head, food, game.w, game.h, BLOCK_SIZE, body)
        
        # perform move and get new state
        # In our refined wrapper, step returns (obs, reward, terminated, truncated, info)
        state_new, reward, done, _, info = wrapper.step(final_move)
        score = info["score"]
        
        # --- REWARD SHAPING (The Algorithmic Twist) ---
        # 1. A* Bonus: reward for moving towards food along the optimal path
        if path and len(path) > 0:
            new_head = (game.head.x, game.head.y)
            if new_head == path[0]:
                reward += 1.0 # Bonus for following A* guidance
            else:
                reward -= 0.5 # Slight penalty for deviating from optimal path
        
        # 2. DFS Survival Check: penalty for entering a dead end
        new_body = [(p.x, p.y) for p in game.snake]
        if is_dead_end((game.head.x, game.head.y), game.w, game.h, BLOCK_SIZE, new_body):
            reward -= 5.0 # Pre-emptive penalty for trapping itself
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Episode: {agent.n_games} | Score: {score} | Record: {record} | Reward: {reward:.2f}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

if __name__ == '__main__':
    train()
