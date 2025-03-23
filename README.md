# RL-SnakeGame

This project implements a Q-Learning-based AI using PyTorch to play the Snake game. The model has been trained with different strategies for exploration (`epsilon`) and reward systems. Below, you’ll find an overview of the training process, results, and insights.

![Record Game](/recordings/record_game754_score73.gif)

---

## Table of Contents
1. [Overview](#overview)
2. [Training Strategies](#training-strategies)
    - [Linear Epsilon Decay](#1-linear-epsilon-decay)
    - [Exponential Epsilon Decay](#2-exponential-epsilon-decay)
    - [Modified Reward System with Linear Epsilon Decay](#3-modified-reward-system-with-linear-epsilon-decay)
    - [Modified Reward System with Exponential Epsilon Decay](#4-modified-reward-system-with-exponential-epsilon-decay)
    - [Frame Iteration Reset on Food Pickup](#5-frame-iteration-reset-on-food-pickup)
3. [Results](#results)
4. [Conclusion](#conclusion)
5. [How to Run](#how-to-run)

---

## Overview

The goal of this project is to teach an AI how to play the Snake game using the Q-Learning algorithm. By experimenting with different exploration and reward mechanisms, I observed significant variations in the model's performance and its ability to outperform the given average score (for the sake of time I chose 20 points).

---

## Training Strategies

### 1. Linear Epsilon Decay
- **Description**: `self.epsilon = 80 - self.n_games`. This approach starts with a high exploration rate and gradually reduces it linearly as the number of games increases.
- **Performance**: The model outperformed the average score of 20 at **game 266**.
- **Graph**:

    ![Linear Epsilon Decay](/images/epsilon_strong80_266.png)

### 2. Exponential Epsilon Decay
- **Description**: Exponential decay over time. This approach reduces exploration more gradually compared to strict decay.
- **Performance**: The model outperformed the average score at **game 436**.
- **Graph**:

    ![Exponential Epsilon Decay](/images/epsilon_exp_436.png)

### 3. Modified Reward System with Linear Epsilon Decay
- **Description**: In this approach, the reward system was adjusted as follows:
    ```python
    reward = 0
    game_over = False
    if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
        game_over = True
        reward = -10 * (len(self.snake) // 2)
        return reward, game_over, self.score

    if self.head == self.food:
        self.score += 1
        reward = 10 * len(self.snake)
        self._place_food()
    else:
        self.snake.pop()
        reward -= (0.9 ** len(self.snake))
    ``` 
This approach also uses Linear Epsilon Decay.
- **Performance**: The model outperformed the average score of 20 at **game 808**. The graph for this approach shows a more linear improvement compared to the previous strategies.
- **Graph**:

    ![Modified Reward System with Linear Epsilon Decay](/images/fine_for_inactivity_and_fine_for_late_collision_strong80_808.png)

### 4. Modified Reward System with Exponential Epsilon Decay

- **Description**: This strategy combines an **exponential epsilon decay** with the **modified reward system**. The reward system is defined as follows:

  ```python
  reward = 0
  game_over = False
  if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
      game_over = True
      reward = -10 * (len(self.snake) // 2)
      return reward, game_over, self.score

  if self.head == self.food:
      self.score += 1
      reward = 10 * len(self.snake)
      self._place_food()
  else:
      self.snake.pop()
      reward -= (0.9 ** len(self.snake))
    ```
- **Performance**: The model outperformed the average score of 20 points at **game 793**. The graph shows a stable growth in the average score, also it has shown slightly better results compared to the same reward system with Linear Epsilon Decay.
- **Graph**:

    ![Modified Reward System with Exponential Epsilon Decay](/images/fine_for_inactivity_and_fine_for_late_collision_exp_epsilon_793.png)

### 5. Frame Iteration Reset on Food Pickup
- **Description**: Uses Linear decay over time. This approach further refines the reward system by resetting `frame_iteration` when the snake eats food.
  ```python
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
    ```
This change prevents unnecessary penalties for long survival times and allows the model to focus more on learning optimal movement patterns
- **Performance**: The model outperformed the average score at **game 756**. It showed more stable progress compared to previous reward strategies
- **Graph**:

    ![Frame Iteration Reset on Food Pickup](/images/fine_for_inactivity_and_fine_for_late_collision_strong80_fram_iter_756.png)

---

## Results

### Performance Comparison
- **Linear Epsilon Decay**: Fastest improvement, outperforming the average score of 20 at game 266. The graph resembles a square root function.
- **Exponential Epsilon Decay**: Slower improvement, crossing the average score of 20 at game 436. The graph also resembles a square root function.
- **Modified Reward System with Linear Epsilon Decay**: The slowest improvement, reaching the average score of 20 at game 808. However, the graph demonstrates consistent, near-linear improvement.
- **Modified Reward System with Exponential Epsilon Decay**: Slightly faster improvement, compared to the same Reward System with Linear Epsilon Decay. Reaching the average score of 20 at game 793. However, the graph also resembles a square root function rather than near-linear function.
- **Frame Iteration Reset on Food Pickup**: Balanced improvement, outperforming the average score of 20 at game 756. The approach leads to a more stable progression in training.

### Visualized Progress Comparison
- These gif-files were recorded based on the *Frame Iteration Reset on Food Pickup* model, as it has shown the most stable progress:
#### Game №100:
![Game №100](/recordings/game100.gif)
#### Game №500:
![Game №500](/recordings/game500.gif)
#### Game №900:
![Game №900](/recordings/game900.gif)


### Insights
- A stricter epsilon decay resulted in faster learning due to aggressive exploration early on.
- Gradual epsilon decay led to slower learning but still achieved reasonable performance.
- The modified reward system improved the stability of the model’s learning curve, albeit at the cost of slower progress.
- The modified reward system combined with exponential epsilon decay has shown slightly better learning performance than the model with the same reward system with linear epsilon decay, however the graph demonstrates less-linear improvement than the graph of modified reward system with linear epsilon decay.
- Frame iteration reset on food pickup has shown the most gradual and stable progress over time.

---

## Conclusion

Through experimentation with different exploration and reward strategies, I observed significant differences in the model's learning behavior:
- **Linear Epsilon Decay**: Fast learning but potentially riskier due to reduced exploration over time.
- **Exponential Epsilon Decay**: Slower but more stable learning curve.
- **Modified Reward System with Linear Epsilon Decay**: Enhanced stability and better long-term performance, but slower initial progress.
- **Modified Reward System with Exponential Epsilon Decay**: Less stable, but faster initial performance than the model with linear epsilon decay.
- **Frame Iteration Reset on Food Pickup**: The most stable model, but initial performance is quite slow compared to other models.

---

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Neopt0lemus/rl_snake
    ```


2. Run the agent:
    ```bash
    python agent.py
    ```

3. (Optional) View the saved graphs in the `images` folder.
4. NOTE! In the file `snake_game.py` you may find the 
    ```python
        SPEED = 40000 
    ``` 

    variable. It was assigned so in order to make longterm research. The optimal value to see the performance yourself is 40.

    To see the results of different approaches you may want to uncomment this section in `agent.py` file
    ```python
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
    ```
    and this section in `snake_game.py` file
    ```python
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
    ```
    *Important* The repository contains plotter and recorder to do the research. However you may want to run the game without these tools. In order to do so comment the sections with `plot()` function and `recorder` instance in `agent.py` file. Just like this:
    ``` python
    def train() -> None:
        """
        Trains the reinforcement learning agent to play the Snake game.
        """
        # plot_scores = []
        # plot_mean_scores = []
        record = 0
        # total_score = 0
        # threshold = 20
        # threshold_reached = False
        agent = Agent()
        game = SnakeGameAI()
        # recorder = GameRecorder()
        while True:
            old_state = agent.get_state(game)

            final_action = agent.get_action(old_state)

            reward, has_game_ended, score = game.play_step(final_action)
            new_state = agent.get_state(game)

            agent.train_shortterm_memory(old_state, final_action, reward, new_state, has_game_ended)
            agent.remember(old_state, final_action, reward, new_state, has_game_ended)

            # recorder.record_frame(pg.display.get_surface())        

            if has_game_ended:
                game.reset()
                agent.n_games += 1
                agent.train_longterm_memory()

                # if agent.n_games % 100 == 0 and score <= record:
                #     recorder.save_game(filename=f"recordings/game{agent.n_games}.gif")

                if score > record:
                    record = score
                    agent.model.save()
                #     recorder.save_game(filename=f"recordings/record_game{agent.n_games}_score{score}.gif")
                #     print(f"New record! Game {agent.n_games} saved with score {score}")

                # recorder.drop_game()

                print('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record)

            
                # plot_scores.append(score)
                # total_score += score
                # mean_score = total_score / agent.n_games
                # plot_mean_scores.append(mean_score)
                # if plot_mean_scores[-1] >= threshold and not threshold_reached:
                #     threshold_reached = True
                #     print(f"Average score exceeded {threshold} at game № {agent.n_games}!")
                #     plot(plot_scores, plot_mean_scores, threshold_reached=threshold_reached, threshold=threshold)
                # else:
                #     plot(plot_scores, plot_mean_scores, threshold_reached=False, threshold=threshold)
        ```

---

## Author

Developed by Saveliy Maksimau. If you have any questions or feedback, feel free to reach out or create an issue in the repository.