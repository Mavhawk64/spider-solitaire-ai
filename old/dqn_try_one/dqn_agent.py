import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from SpiderSolitaireEnv import SpiderSolitaireEnv

# 🚀 Get the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.1,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr  # Learning rate

        # Replay memory
        self.memory = deque(maxlen=10000)

        # 🚀 Move models to GPU
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Copy weights to target model
        self.update_target_model()

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action based on an epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)  # Random action (exploration)

        # 🚀 Convert state to tensor and move to GPU
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)  # Use GPU model
        return torch.argmax(q_values).item()  # Best action (exploitation)

    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # 🚀 Move states to GPU
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            next_state_tensor = torch.tensor(
                next_state, dtype=torch.float32, device=device
            ).unsqueeze(0)

            # Compute Q target
            with torch.no_grad():
                max_next_q = torch.max(self.target_model(next_state_tensor)).item()
                target = reward if done else reward + self.gamma * max_next_q

            q_values = self.model(state_tensor)
            target_q_values = q_values.clone()
            target_q_values[0, action] = target

            # Train the model
            self.optimizer.zero_grad()
            loss = self.loss_fn(q_values, target_q_values)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """Save the trained model."""
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(filename, map_location=device))
        self.update_target_model()
