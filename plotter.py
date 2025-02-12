import matplotlib.pyplot as plt
from IPython import display

threshold_game = None

plt.ion()

def plot(scores, mean_scores, threshold_reached, threshold):
    global threshold_game
    display.display(plt.gcf())
    plt.clf()
    plt.title('Snake AI Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score per Game')
    plt.plot(mean_scores, label='Average Score', linestyle='dashed')
    if threshold_reached and threshold_game is None:
        threshold_game = len(mean_scores)

    if threshold_game is not None:
        plt.axvline(threshold_game, color="green", linestyle="dashed", label=f"Avg Score > {threshold}")
    plt.legend()
    plt.pause(0.1)