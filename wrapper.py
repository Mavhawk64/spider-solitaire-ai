from typing import List

import gym
import numpy as np

from model import Agent, DeepQNetwork
from Moves import Moves
from SpiderSolitaire import SpiderSolitaire
from utils import get_valid_index_arr, plotLearning


def game_test():
    main_game = SpiderSolitaire(1, 0)
    main_game.display_board()
    val = main_game.get_possible_moves()
    print(val)
    ind = get_valid_index_arr(main_game, val)
    print(ind)
    for i in ind:
        print(main_game.all_moves[i])


if __name__ == "__main__":
    # game_test()
    env = gym.make("SpiderSolitaire-v0")  # ?
    brain = Agent(gamma=0.95, epsilon=1.0, alpha=0.003)
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            valid_action_indices = env.get_valid_moves()
            action = brain.chooseAction(observation, valid_action_indices)
            observation_, reward, done, info = env.step(action)
            if done:
                observation_ = None
            brain.storeTransition(observation, action, reward, observation_)
            observation = observation_
