import imageio
import pygame as pg
import numpy as np
import os

class GameRecorder:

    DIR: str = "recordings"

    def __init__(self, dir_path: str = DIR) -> None:
        """
        Initializes the GameRecorder class. Creates a directory for storing recordings 
        if it does not already exist.
        """
        self.frames = []
        os.makedirs(dir_path, exist_ok=True)

    def record_frame(self, game_screen: any) -> None:
        """
        Records a single frame of the game by capturing the current game screen, 
        converting it to a NumPy array, flipping and rotating it as needed to match 
        the correct orientation, and appending the processed frame to the frames list.
        """
        screen_array = pg.surfarray.array3d(game_screen)
        self.frames.append(np.rot90(np.flipud(screen_array), -1)) 

    def drop_game(self) -> None:
        """
        Clears all recorded frames for the current game.
        """
        self.frames = []  

    def save_game(self, filename: str) -> None:
        """
        Saves the recorded game frames as a GIF file using the specified filename. 
        If there are no frames to save, the method does nothing. Once saved, the frames 
        list is cleared. Prints a confirmation message when the file is successfully saved.
        """
        if self.frames:
            imageio.mimsave(filename, self.frames, duration=40)
            print(f"Saved: {filename}")
            self.frames = []
