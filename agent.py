import torch
import random
import numpy as np
import matplotlib as plt
import pygame as pg
from collections import deque
from snake_game import SnakeGameAI, Direction, Point
from model import LinearQNetwork, QTrainer
from plotter import plot
from game_recorder import GameRecorder

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
# LR = 0.0005

class Agent():
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        # self.gamma = 0.85
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNetwork(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        """
        Extract the current game state as a NumPy array.
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger forward
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger to the right
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger to the left
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state: np.ndarray, action: list[int],
            reward: float, next_state: np.ndarray, has_game_ended: bool) -> None:
        """
        Stores the experience (state, action, reward, next state, game status) in memory.
        """
        self.memory.append((state, action, reward, next_state, has_game_ended))

    def train_longterm_memory(self) -> None:
        """
        Trains the model using a batch of past experiences stored in memory.
        """
        if len(self.memory) > BATCH_SIZE:
            # last_samples = list(self.memory)[-BATCH_SIZE//2:]
            # rand_samples = random.sample(self.memory, BATCH_SIZE//2)
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, has_games_ended = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, has_games_ended)

    def train_shortterm_memory(self, state: np.ndarray, action: list[int],
            reward: float, next_state: np.ndarray, has_game_ended: bool) -> None:
        """
        Trains the model with a single experience step.
        """
        self.trainer.train_step(state, action, reward, next_state, has_game_ended)

    def get_action(self, state: np.ndarray) -> list[int]:
        """
        Decides the next action.
        """
        # self.epsilon = max(0.01, 0.1 * (0.99 ** self.n_games))
        self.epsilon = 80 - self.n_games
        final_action = [0, 0, 0]
        # if random.uniform(0, 1) < self.epsilon:
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
            final_action[action] = 1
        else:
            current_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(current_state)
            action = torch.argmax(prediction).item()
            final_action[action] = 1
        return final_action


def train() -> None:
    """
    Trains the reinforcement learning agent to play the Snake game.
    """
    plot_scores = []
    plot_mean_scores = []
    record = 0
    total_score = 0
    threshold = 20
    threshold_reached = False
    agent = Agent()
    game = SnakeGameAI()
    recorder = GameRecorder()
    while True:
        old_state = agent.get_state(game)

        final_action = agent.get_action(old_state)

        reward, has_game_ended, score = game.play_step(final_action)
        new_state = agent.get_state(game)

        agent.train_shortterm_memory(old_state, final_action, reward, new_state, has_game_ended)
        agent.remember(old_state, final_action, reward, new_state, has_game_ended)

        recorder.record_frame(pg.display.get_surface())        

        if has_game_ended:
            game.reset()
            agent.n_games += 1
            agent.train_longterm_memory()

            if agent.n_games % 100 == 0 and score <= record:
                recorder.save_game(filename=f"recordings/game{agent.n_games}.gif")

            if score > record:
                record = score
                agent.model.save()
                recorder.save_game(filename=f"recordings/record_game{agent.n_games}_score{score}.gif")
                print(f"New record! Game {agent.n_games} saved with score {score}")

            recorder.drop_game()

            print('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record)

        
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            if plot_mean_scores[-1] >= threshold and not threshold_reached:
                threshold_reached = True
                print(f"Average score exceeded {threshold} at game № {agent.n_games}!")
                plot(plot_scores, plot_mean_scores, threshold_reached=threshold_reached, threshold=threshold)
            else:
                plot(plot_scores, plot_mean_scores, threshold_reached=False, threshold=threshold)


if __name__ == '__main__':
    train()
