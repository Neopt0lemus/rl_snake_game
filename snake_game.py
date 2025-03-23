import pygame as pg
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pg.init()
font = pg.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40000

class SnakeGameAI:
    def _move(self, action: list[int]) -> None:
        """
        Updates the snake's direction and moves the head to the new position.
        """
        # [move_forward, right, left]
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clockwise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clockwise[next_idx]
        
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self) -> None:
        """
        Updates the game screen by drawing the snake, food, and score.
        """
        self.display.fill(BLACK)

        for pt in self.snake:
            pg.draw.rect(self.display, BLUE1, pg.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pg.draw.rect(self.display, BLUE1, pg.Rect(pt.x+4, pt.y+4, 12, 12))
        
        pg.draw.rect(self.display, RED, pg.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pg.display.flip()   

    def _place_food(self) -> None:
        """
        Places food randomly on the game board, ensuring it does not spawn inside the snake.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def __init__(self, w=640, h=480) -> None:
        """
        Initializes the game window.
        """
        self.w = w
        self.h = h
        self.display = pg.display.set_mode((self.w, self.h))
        pg.display.set_caption('Snake')
        self.clock = pg.time.Clock()
        self.reset()    

    def reset(self) -> None:
        """
        Resets the game state.
        """
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                        Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    
    def play_step(self, action: list[int]) -> tuple[float, bool, int]:
        """
        Executes one step of the game, updating the snake's position, checking for collision, 
        and returning reward.
        """
        self.frame_iteration += 1
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            # reward = -10
            reward = -10 * (len(self.snake) // 2)
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            self.frame_iteration = 0
            reward = 10 * len(self.snake)
            # reward = 10
            self._place_food()
        else:
            self.snake.pop()
            # or no punishment
            reward -= (0.9 ** len(self.snake))

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt: Point = None) -> bool:
        """
        Checks if the snake collides with the wall or itself.
        """
        if pt is None:
            pt = self.head
        
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False
