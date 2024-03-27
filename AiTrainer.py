import random
import numpy as np
from DQNAgent import DQNAgent
from SpiderSolitaire import SpiderSolitaire
import os
import matplotlib.pyplot as plt
from bcolors import bcolors

# Define rewards and penalties

# Reward system
REWARD_FOR_BUILDING_SEQUENCE = 5
REWARD_FOR_EXPOSING_HIDDEN_CARD = 10
REWARD_FOR_CREATING_EMPTY_PILE = 15
REWARD_FOR_BUILDING_ON_HIGHER_CARD_OUT_OF_SUIT = 2
REWARD_FOR_MAXIMIZING_CARD_EXPOSURE_BEFORE_NEW_DEAL = 20

# Penalties
PENALTY_FOR_REDUNDANT_MOVE = -10

# Maximum number of cards per column
MAX_CARDS_PER_COLUMN = 104

class AiTrainer:
    def __init__(self, difficulty=4, seed=None, load_model_path=None, display_boards=False):
        self.seed = seed
        self.prev_action = None
        self.prev_actions = []
        self.display_boards = display_boards
        self.difficulty = difficulty
        self.rep_tolerance = 2
        self.removed_moves = []
        # Initialize Spider Solitaire and DQN Agent with example parameters
        self.game = SpiderSolitaire(self.difficulty, seed)        
        state_size = 11+10*MAX_CARDS_PER_COLUMN  # Adjust based on your state representation
        action_size = 18541  # Estimate of possible actions
        self.save_interval = 1  # Save the model every game
        self.model_save_path = "./output/models"  # Define the path to save your model
        self.folder = len(os.listdir(self.model_save_path))
        if self.seed is not None:
            self.folder = self.seed
        self.agent = DQNAgent(state_size, action_size, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.995, min_exploration_rate=0.01)
        if load_model_path:
            print(f"Loading model from {load_model_path}")
            self.agent.load_model(load_model_path)
            self.start_from_game = int(load_model_path.split("_")[-1].split(".")[0])
        else:
            self.start_from_game = 1
        # Initialize lists to store metrics
        self.win_rates = []
        self.total_rewards = []
        self.moves_per_game = []

    def reset_game(self):
        self.game = SpiderSolitaire(self.difficulty, self.seed)  # Reset the game for a new simulation

    def get_state(self):
        state = []

        # Encode each tableau column
        for column in self.game.tableau:
            column_state = []
            # Encode visible cards (simplified example: rank only)
            for card in column:
                if card.face_up:
                    column_state.append(self.cantor_pair(card.suit, card.rank))  # Assuming ranks are numerical
                else:
                    column_state.append(-1)  # -1 represents a hidden card
            # Pad column state to ensure fixed length
            column_state += [0] * (MAX_CARDS_PER_COLUMN - len(column_state))
            state.extend(column_state)

        # Encode empty columns
        empty_columns = [1 if len(column) == 0 else 0 for column in self.game.tableau]
        state.extend(empty_columns)

        # Encode remaining cards in the deck
        state.append(len(self.game.deck.cards))

        # Convert to numpy array and reshape for the neural network
        return np.array(state).reshape(1, -1)


    def get_reward(self, action, success, game_state_before, game_state_after):
            reward = 0
            if not success:
                return PENALTY_FOR_REDUNDANT_MOVE * (1 + len(self.removed_moves))  # Assuming 'success' indicates a valid and non-redundant move

            # Check for specific achievements to add rewards
            if game_state_after.exposed_hidden_card:
                reward += REWARD_FOR_EXPOSING_HIDDEN_CARD
            if game_state_after.created_empty_pile:
                reward += REWARD_FOR_CREATING_EMPTY_PILE
            if game_state_after.built_sequence:
                reward += REWARD_FOR_BUILDING_SEQUENCE * action[2]  # Reward based on the length of the sequence
            if game_state_after.built_on_higher_card_out_of_suit:
                reward += REWARD_FOR_BUILDING_ON_HIGHER_CARD_OUT_OF_SUIT
            if game_state_after.maximized_card_exposure_before_new_deal:
                reward += REWARD_FOR_MAXIMIZING_CARD_EXPOSURE_BEFORE_NEW_DEAL

            # Add more conditions as needed based on game state changes
            return reward


    def choose_action(self, rem=None):
        state = self.get_state()
        valid_actions = [self.get_action_index(move) for move in self.get_valid_moves()]
        # if there are draw cards, add that as a valid action
        if len(self.game.deck.cards) > 0:
            valid_actions.append(-1)
        if rem is not None:
            self.removed_moves.append(rem)
            self.removed_moves.append((rem[1],rem[0],rem[2]))
            for remo in self.removed_moves:
                if self.get_action_index(remo) in valid_actions:
                    valid_actions.remove(self.get_action_index(remo))

        if not valid_actions:  # Check if there are no valid actions
            return None  # Return None or handle this scenario appropriately

        action_index = self.agent.choose_action(state, valid_actions)
        if action_index == -1:
            return None
        return self.get_action_tuple(action_index)
    
    def perform_action(self, action):
        if self.prev_action is None:
            self.prev_actions = [] # reset after a draw
            self.removed_moves = [] # reset after a draw
        # if action is valid, perform it and return True; if action is invalid, check to see if there are cards left in the deck and draw if possible, otherwise return False
        if action is not None and action in self.get_valid_moves(stdout=False) and not (self.prev_action is not None and action is not None and ((action[0] == self.prev_action[1] and action[1] == self.prev_action[0] and action[2] == self.prev_action[2]) or (action in self.prev_actions and self.prev_actions[:min(5,len(self.prev_actions))].count(action) > self.rep_tolerance))):
            ret = self.game.move_bundle(action[0], action[1], action[2])
            if ret:
                self.prev_action = action
                self.prev_actions.append(action)
                self.removed_moves = []
            return ret
        elif self.prev_action is not None and action is not None and action[0] == self.prev_action[1] and action[1] == self.prev_action[0] and action[2] == self.prev_action[2]:
            
            return self.perform_action(self.choose_action(rem=action))
        elif len(self.game.deck.cards) > 0:
            self.game.draw_cards()
            self.prev_action = None
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
        return self.cantor_pair_3_inverse(action_index)

    def get_valid_moves(self, stdout=True):
        # Return a list of valid moves (source column, target column, bundle length)
        valid_moves = []
        for i, source_column in enumerate(self.game.tableau):
            for bundle in self.game.find_bundles(source_column):
                bundle_length = len(bundle)
                for j, target_column in enumerate(self.game.tableau):
                    if i != j and self.game.can_move_bundle(bundle, target_column):
                        valid_moves.append((i, j, bundle_length))
        # add all the sub-bundles as valid moves
        # length = len(valid_moves)
        # for i in range(length):
        #     if valid_moves[i][2] > 1:
        #         for j in range(1, valid_moves[i][2]):
        #             valid_moves.append((valid_moves[i][0], valid_moves[i][1], j))
        if stdout:
            print(valid_moves)
        return valid_moves

    def is_game_stuck(self):
        """Check if the game is stuck."""
        return sum([i.stuck_moves for i in list(self.game.previous_games)]) > 2 or sum([i.draw_count for i in list(self.game.previous_games)]) > 5 # be aggressive with the draw count - it's the fastest way to lose

    def play_game(self):
        total_reward = 0
        moves = 0

        if self.display_boards:
            print('\n')
            self.game.display_board()

        while not self.game.has_won() and not (self.is_game_stuck() and len(self.game.deck.cards) == 0):
            prev_state = self.get_state()
            prev_reward_state = self.game.reward_state
            action = self.choose_action()
            print(bcolors.BLUE_FG + f"Action: {"Draw" if action is None else action}" + bcolors.ENDC)
            moves += 1

            if action is None:
                if len(self.game.deck.cards) > 0:
                    self.game.draw_cards()
                    if self.display_boards:
                        print('\n')
                        self.game.display_board()
                else:
                    break  # No valid moves and no cards to draw, end the game
                continue

            success = self.perform_action(action)
            next_state = self.get_state()
            next_reward_state = self.game.reward_state
            reward = self.get_reward(action, success, prev_reward_state, next_reward_state)
            total_reward += reward
            done = self.game.has_won() or (self.is_game_stuck() and len(self.game.deck.cards) == 0)
            self.agent.decay_exploration_rate()
            action_index = self.get_action_index(action)
            self.agent.update_model(prev_state, action_index, reward, next_state, done, f"{self.model_save_path}/{self.folder}")

            # Optional: Decay exploration rate
            self.agent.decay_exploration_rate()

            # Update game display
            if self.display_boards:
                print('\n')
                self.game.display_board()
        win = 1 if self.game.has_won() else 0
        self.win_rates.append(win)
        self.total_rewards.append(total_reward)
        self.moves_per_game.append(moves)


    def plot_progress(self, display=True):
        plot_save_path = f"{self.model_save_path}/{self.folder}/plots"
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)
        # Plot win rate
        plt.figure(figsize=(10, 7))
        plt.subplot(4, 1, 1)
        plt.plot(self.win_rates, label='Win Rate')
        plt.xlabel('Game')
        plt.ylabel('Win Rate')
        plt.title('Win Rate Over Time')
        plt.legend()

        # Plot average reward
        plt.subplot(4, 1, 2)
        plt.plot(self.total_rewards, label='Reward')
        plt.xlabel('Game')
        plt.ylabel('Reward')
        plt.title('Average Reward Per Game')
        plt.legend()

        # Plot moves per game
        plt.subplot(4, 1, 3)
        plt.plot(self.moves_per_game, label='Moves Per Game')
        plt.xlabel('Game')
        plt.ylabel('Moves')
        plt.title('Moves Per Game Over Time')
        plt.legend()

        # Plot rewards per move
        plt.subplot(4, 1, 4)
        rewards_per_move = [r / m if m > 0 else 0 for r, m in zip(self.total_rewards, self.moves_per_game)]
        plt.plot(rewards_per_move, label='Rewards Per Move')
        plt.xlabel('Game')
        plt.ylabel('Reward')
        plt.title('Average Reward Per Move Over Time')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{plot_save_path}/training_progress.png")
        print(f"Plot saved to {plot_save_path}/training_progress.png")
        plt.show(block=display)

    def train(self, num_games):
        for game_number in range(self.start_from_game,num_games+1):
            self.play_game()
            self.reset_game()
            # Save the model at the specified interval
            if game_number % self.save_interval == 0:
                if not os.path.exists(f"{self.model_save_path}/{self.folder}"):
                    os.makedirs(f"{self.model_save_path}/{self.folder}")
                self.agent.save_model(f"{self.model_save_path}/{self.folder}/model_{game_number}.keras")
                print(f"Model saved after {game_number} games.")

            # Plot the training progress
            self.plot_progress(display=False)
            print(bcolors.BRIGHT_GREEN_FG + f"Game {game_number} completed.\nWin rate: {self.win_rates[-1]}.\nTotal reward: {self.total_rewards[-1]}.\nMoves: {self.moves_per_game[-1]}.\nAverage reward: {self.total_rewards[-1]/self.moves_per_game[-1] if self.moves_per_game[-1] > 0 else 0}."+ bcolors.ENDC)
        self.plot_progress()
        print("Training completed.")
# Example usage
load_model_path = "./output/models/1/model_1.keras"
trainer = AiTrainer(difficulty=1, display_boards=False) # Initialize the AI trainer (seed is set to None for random seed)
trainer.train(10)  # Train the AI
