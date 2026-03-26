import torch
import sys
import os
import time

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.snake_game import SnakeGame
from env.wrapper import SnakeGameAIWrapper
from src.agent import Agent

def test():
    # Load agent
    agent = Agent()
    model_path = './model/model.pth'
    
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()
        print(f"Loaded trained model: {model_path}")
    else:
        print("No trained model found. Testing with an untrained agent.")

    game = SnakeGame()
    wrapper = SnakeGameAIWrapper(game)

    while True:
        # get state
        state = wrapper.get_state()

        # get move (No exploration)
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = agent.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        # perform move
        _, done, score = game.play_step(final_move)
        
        if done:
            print(f'Game Over! Final Score: {score}')
            time.sleep(2) # Show the end state
            game.reset()

if __name__ == '__main__':
    test()
