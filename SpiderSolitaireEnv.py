import gym
import numpy as np
from gym import spaces

from SpiderSolitaire import SpiderSolitaire
from utils import get_valid_index_arr, vectorize_state  # You'll need to define this


class SpiderSolitaireEnv(gym.Env):
    def __init__(self):
        super(SpiderSolitaireEnv, self).__init__()
        self.game = SpiderSolitaire(4, 0)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(805,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1082)  # 1080 move actions + draw + undo

    def reset(self):
        self.game = SpiderSolitaire(1, 0)
        return vectorize_state(self.game)

    def step(self, action_idx):
        move = self.game.all_moves[action_idx]
        reward, done = self.game.execute_move(move)
        next_state = vectorize_state(self.game)
        return next_state, reward, done, {}

    def get_valid_moves(self):
        return get_valid_index_arr(self.game, self.game.get_possible_moves())
