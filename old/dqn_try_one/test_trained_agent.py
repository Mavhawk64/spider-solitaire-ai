import numpy as np
import torch

from dqn_agent import DQNAgent
from SpiderSolitaireEnv import SpiderSolitaireEnv

# Load the trained model
env = SpiderSolitaireEnv(suits=4, seed=42)
state_size = 10 * 13 + 1  # Tableau (10x13) + Draw Pile
action_size = 100
agent = DQNAgent(state_size, action_size)
agent.load("dqn_spider_solitaire.pth")

# Run the agent on a test game
state, _ = env.reset()
state = np.array(state).flatten()

for step in range(500):
    action = agent.act(state)
    state, _, done, _ = env.step(action)
    state = np.array(state).flatten()

    if done:
        print("Game Over!")
        break

env.close()
