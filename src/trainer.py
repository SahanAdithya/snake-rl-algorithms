import torch
import numpy as np
import random
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
        
        # Determine the "best" move based on A*
        a_star_bonus = 0
        if path:
            next_step = path[0]
            # Convert final_move to the coordinate change it produces
            # and check if it matches next_step
            # For simplicity, we can just check if the action leads to the next_step
            pass # We will apply bonuses in the step logic below

        # perform move and get new state
        state_new, reward, done, score = wrapper.step(final_move)
        
        # --- REWARD SHAPING ---
        # 1. A* Bonus: reward for moving towards food
        if path and len(path) > 0:
            new_head = (game.head.x, game.head.y)
            if new_head == path[0]:
                reward += 1 # Bonus for following A* path
        
        # 2. DFS Penalty: penalty for entering a dead end
        new_body = [(p.x, p.y) for p in game.snake]
        if is_dead_end((game.head.x, game.head.y), game.w, game.h, BLOCK_SIZE, new_body):
            reward -= 5 # Heavy penalty for dead ends
        
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

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # You could add plotting here using matplotlib if desired

if __name__ == '__main__':
    train()
