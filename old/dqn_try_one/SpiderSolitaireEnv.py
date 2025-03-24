import gym
import numpy as np
from gym import spaces

from Moves import Move
from SpiderSolitaire import SpiderSolitaire


class SpiderSolitaireEnv(gym.Env):
    """
    OpenAI Gym-like RL environment for Spider Solitaire.
    """

    def __init__(
        self, suits: int = 4, seed: int | float | str | bytes | bytearray | None = None
    ):
        super(SpiderSolitaireEnv, self).__init__()

        self.game = SpiderSolitaire(suits, seed)
        # Placeholder, will map to real moves.
        self.action_space = spaces.Discrete(100)

        # State space: tableau card states (flipped/unflipped) + draw pile count
        self.observation_space = spaces.Box(
            # 10 columns, max 13 visible cards per column
            low=-5000,
            high=5000,
            shape=(10, 13),
            dtype=np.int16,
        )

    def cantor_pair(self, k1: int, k2: int) -> int:
        return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

    def inverse_cantor_pair(self, z: int) -> tuple[int, int]:
        w = int(((8 * z + 1) ** 0.5 - 1) / 2)
        t = (w * (w + 3)) // 2
        y = z - t
        x = w - y
        return x, y

    def _get_observation(self) -> tuple[np.ndarray, int]:
        """
        Convert the game state into an RL-compatible format.
        """
        tableau = np.zeros((10, 13), dtype=np.int16)
        for col_idx, column in enumerate(self.game.tableau):
            # Only track last 13 cards max
            for row_idx, card in enumerate(column[-13:]):
                tableau[col_idx, row_idx] = (
                    self.cantor_pair(card.rank, card.suit)
                    * (-1 if not card.face_up else 1)
                    if card.known
                    else -1
                )

        draw_pile = len(self.game.deck.cards)
        return tableau, draw_pile

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict]:
        """
        Take an action in the environment.
        Returns (observation, reward, done, info)
        """
        move = self._decode_action(action)
        reward = 0

        if move.move_type == "move":
            success = self.game.move_bundle(
                move.source, move.target, move.bundle_length
            )
            if success:
                reward = self._calculate_reward(move)
        elif move.move_type == "draw":
            self.game.draw_cards()
            reward = -10  # Slight penalty to discourage unnecessary draws
        elif move.move_type == "undo":
            self.game.undo()
            reward = -20  # Undo should not be abused

        observation = self._get_observation()
        done = self.game.has_won()
        return observation, reward, done, {}

    def reset(
        self, seed: int | float | str | bytes | bytearray | None = None
    ) -> tuple[np.ndarray, int]:
        """
        Reset the game for a new episode.
        """
        self.game = SpiderSolitaire(suits=self.game.suits, seed=seed)
        return self._get_observation()

    def _calculate_reward(self, move: Move) -> int:
        """
        Assign rewards based on move impact.
        """
        reward = 0
        if self.game.reward_state.exposed_hidden_card:
            reward += 20
        if self.game.reward_state.created_empty_pile:
            reward += 30
        if self.game.reward_state.built_sequence:
            reward += 50
        if self.game.reward_state.built_on_higher_card_out_of_suit:
            reward -= 5
        return reward

    def _decode_action(self, action: int) -> Move:
        """
        Convert action index into a Move object.
        Placeholder: Replace with proper mapping.
        """
        moves = self.game.get_possible_moves()
        return moves[action % len(moves)]  # Map action index to valid moves
