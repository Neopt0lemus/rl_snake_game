import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
import os

class LinearQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.lr = learning_rate
        self.gamma = gamma
        self.optimizer = optimizer.Adam(model.parameters(), learning_rate)
        self.loss_function = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, has_game_ended):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            has_game_ended = (has_game_ended, )

        prediction = self.model(state)
        target = prediction.clone()

        for idx in range(len(has_game_ended)):
            new_Q_value = reward[idx]
            if not has_game_ended[idx]:
                new_Q_value = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = new_Q_value
        
        self.optimizer.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()
        self.optimizer.step()
