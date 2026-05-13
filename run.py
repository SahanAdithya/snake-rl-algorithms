import sys
import matplotlib.pyplot as plt
from IPython import display
from src.agent import Agent
from env.snake_game import SnakeGame
import torch

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

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
            plot(plot_scores, plot_mean_scores)

def test():
    agent = Agent()
    # Load model if it exists
    try:
        agent.model.load_state_dict(torch.load('./model/model.pth'))
        agent.model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No trained model found. Please train the model first.")
        return

    game = SnakeGame()
    while True:
        state = agent.get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = agent.model(state_tensor)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)
        if done:
            game.reset()
            print('Final Score', score)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 run.py [train|test]")
    elif sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        print("Invalid command. Use 'train' or 'test'.")
