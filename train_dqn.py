import numpy as np
import torch

from dqn_agent import DQNAgent
from SpiderSolitaireEnv import SpiderSolitaireEnv

# Hyperparameters
EPISODES = 1000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 10

# Initialize environment
env = SpiderSolitaireEnv(suits=4, seed=42)
state_size = 10 * 13 + 1  # Tableau (10x13) + Draw Pile
action_size = 100  # Placeholder, will be mapped dynamically

agent = DQNAgent(state_size, action_size)

for episode in range(EPISODES):
    tableau, draw_pile = env.reset()
    state = np.append(tableau.flatten(), draw_pile)  # Flatten state
    total_reward = 0

    for step in range(500):  # Max moves per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_tableau, draw_pile = next_state
        # Flatten next_state
        next_state = np.append(next_tableau.flatten(), draw_pile)

        agent.remember(state, action, reward, next_state, done)
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

# Save the trained model
agent.save("dqn_spider_solitaire.pth")

# Close environment
env.close()
