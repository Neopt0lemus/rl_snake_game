import matplotlib.pyplot as plt
from IPython import display

threshold_game = None

plt.ion()

def plot(scores: list[int], mean_scores: list[float], threshold_reached: bool, threshold: int) -> None:
    """
    Visualizes the progress of the Snake AI training. It plots the scores of each game and the 
    running average score (mean_scores). If the average score surpasses the specified threshold for the first 
    time, a vertical dashed green line is drawn at that point to mark it. The plot updates dynamically during training.
    """
    global threshold_game
    display.display(plt.gcf())
    plt.clf()
    plt.title('Snake AI Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score per Game')
    plt.plot(mean_scores, label='Average Score', linestyle='dashed')
    if threshold_reached and threshold_game is None:
        # threshold_game is assigned to a value only when the threshold is reached for the 1st time
        threshold_game = len(mean_scores)

    if threshold_game is not None:
        plt.axvline(threshold_game, color="green", linestyle="dashed", label=f"Avg Score > {threshold}")
    plt.legend()
    plt.pause(0.1)