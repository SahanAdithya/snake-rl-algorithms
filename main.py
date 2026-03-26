import pygame
import random
from env.snake_game import SnakeGame

def test_run():
    game = SnakeGame()
    
    while True:
        # Generate a random action [straight, right, left]
        # For testing, we just pick one randomly
        action = [0, 0, 0]
        action[random.randint(0, 2)] = 1
        
        reward, done, score = game.play_step(action)
        
        if done:
            game.reset()
            print(f'Game Over! Final Score: {score}')

if __name__ == "__main__":
    test_run()