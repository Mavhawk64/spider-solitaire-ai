import numpy as np
import torch

from dqn_agent import DQNAgent
from SpiderSolitaireEnv import SpiderSolitaireEnv

# Initial Values
NUM_SUITS = 4
INITIAL_SEED = 42

# 🚀 Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPISODES = 1000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 10

# Initialize environment
env = SpiderSolitaireEnv(suits=NUM_SUITS, seed=INITIAL_SEED)
state_size = 10 * 13 + 1  # Tableau (10x13) + Draw Pile
action_size = 100  # Placeholder, will be mapped dynamically

# 🚀 Move DQNAgent to GPU
agent = DQNAgent(state_size, action_size)
agent.model.to(device)  # Move the main model to GPU
agent.target_model.to(device)  # Move the target model to GPU

for episode in range(EPISODES):
    tableau, draw_pile = env.reset() if episode > 10 else env.reset(seed=INITIAL_SEED)
    state = np.append(tableau.flatten(), draw_pile)  # Flatten state
    state = torch.tensor(state, dtype=torch.float32, device=device)  # Move to GPU

    total_reward = 0

    for step in range(500):  # Max moves per episode
        action = agent.act(
            state.cpu().numpy()
        )  # Convert back to CPU for action selection
        next_state, reward, done, _ = env.step(action)

        next_tableau, draw_pile = next_state
        next_state = np.append(next_tableau.flatten(), draw_pile)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=device
        )  # Move to GPU

        agent.remember(
            state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done
        )
        agent.replay(BATCH_SIZE)

        state = next_state
        total_reward += reward

        if done:
            break

    if episode % TARGET_UPDATE_FREQUENCY == 0:
        agent.update_target_model()

    print(
        f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.3f}"
    )

# 🚀 Save the trained model (ensure it's saved on GPU)
torch.save(agent.model.state_dict(), "dqn_spider_solitaire.pth")

# Close environment
env.close()
print("Environment closed.")
