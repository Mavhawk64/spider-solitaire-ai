import random
import numpy as np
from DQNAgent import DQNAgent
from SpiderSolitaire import SpiderSolitaire

class AiTrainer:
    def __init__(self):
        # Initialize Spider Solitaire and DQN Agent with example parameters
        self.game = SpiderSolitaire(4, 0)
        state_size = 100  # Adjust based on your state representation
        action_size = 18541  # Estimate of possible actions
        self.agent = DQNAgent(state_size, action_size, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01)

    def reset_game(self):
        self.game = SpiderSolitaire(4, 0)  # Reset the game for a new simulation

    def get_state(self):
        # Convert the Spider Solitaire game state into a format suitable for the DQN agent
        # This is a placeholder; you'll need to implement the actual conversion
        state = [0] * 100  # Example: a placeholder 100-dimensional state
        return np.array(state).reshape(1, -1)  # Reshape to 2D array

    def get_reward(self, action, success):
        # Define and return the reward based on the action and its outcome
        # Placeholder logic; adjust as needed
        return 1 if success else -1

    def choose_action(self):
        state = self.get_state()
        valid_actions = [self.get_action_index(move) for move in self.get_valid_moves()]

        if not valid_actions:  # Check if there are no valid actions
            return None  # Return None or handle this scenario appropriately

        action_index = self.agent.choose_action(state, valid_actions)
        return self.get_action_tuple(action_index)
    
    def perform_action(self, action):
        # if action is valid, perform it and return True; if action is invalid, check to see if there are cards left in the deck and draw if possible, otherwise return False
        if action and action in self.get_valid_moves():
            self.game.move_bundle(action[0], action[1], action[2])
            return True
        elif len(self.game.deck.cards) > 0:
            self.game.draw_cards()
            return True
        return False
    
    def cantor_pair(self, a, b):
        return (a + b) * (a + b + 1) // 2 + b
    
    def cantor_pair_inverse(self, z):
        w = int((np.sqrt(8 * z + 1) - 1) / 2)
        t = (w ** 2 + w) // 2
        y = z - t
        x = w - y
        return x, y

    def cantor_pair_3(self, a, b, c):
        intermediate = self.cantor_pair(a, b)
        return self.cantor_pair(intermediate, c)
    
    def cantor_pair_3_inverse(self, z):
        intermediate, c = self.cantor_pair_inverse(z)
        a, b = self.cantor_pair_inverse(intermediate)
        return a, b, c

    def get_action_index(self, action_tuple):
        # Convert action tuple to an integer index using the Cantor pairing function
        a, b, c = action_tuple
        return self.cantor_pair_3(a, b, c)

    def get_action_tuple(self, action_index):
        # Convert an action index back to an action tuple
        # You'll need to implement the inverse of the Cantor pairing function
        return self.cantor_pair_3_inverse(action_index)

    def get_valid_moves(self):
        # Return a list of valid moves (source column, target column, bundle length)
        valid_moves = []
        for i, source_column in enumerate(self.game.tableau):
            for bundle in self.game.find_bundles(source_column):
                bundle_length = len(bundle)
                for j, target_column in enumerate(self.game.tableau):
                    if i != j and self.game.can_move_bundle(bundle, target_column):
                        valid_moves.append((i, j, bundle_length))
        print(valid_moves)
        return valid_moves

    def is_game_stuck(self):
        """Check if the game is stuck."""
        return self.game.stuck_moves > 10

    def play_game(self):
        while not self.game.has_won() and not (self.is_game_stuck() and len(self.game.deck.cards) == 0):
            state = self.get_state()
            action = self.choose_action()

            if action is None:
                if len(self.game.deck.cards) > 0:
                    self.game.draw_cards()
                else:
                    break  # No valid moves and no cards to draw, end the game
                continue

            success = self.perform_action(action)
            reward = self.get_reward(action, success)
            next_state = self.get_state()
            done = self.game.has_won() or (self.is_game_stuck() and len(self.game.deck.cards) == 0)
            self.agent.decay_exploration_rate()
            action_index = self.get_action_index(action)
            self.agent.update_model(state, action_index, reward, next_state, done)

            # Optional: Decay exploration rate
            self.agent.decay_exploration_rate()

            # Update game display
            print('\n')
            self.game.display_board()

    def train(self, num_games):
        for _ in range(num_games):
            self.play_game()
            self.reset_game()

# Example usage
trainer = AiTrainer()
trainer.train(1)  # Train the AI with 1 game for testing
