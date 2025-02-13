import imageio
import pygame as pg
import numpy as np
import os

class GameRecorder:
    def __init__(self):
        self.frames = []
        os.makedirs("recordings", exist_ok=True)

    def record_frame(self, game_screen):
        screen_array = pg.surfarray.array3d(game_screen)
        self.frames.append(np.rot90(np.flipud(screen_array), -1)) 

    def drop_game(self):
        self.frames = []  

    def save_game(self, n_game, filename):
        if self.frames:
            imageio.mimsave(filename, self.frames, duration=40)
            print(f"Saved: {filename}")
            self.frames = []
