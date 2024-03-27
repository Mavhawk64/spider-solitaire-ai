import Deck
from bcolors import *

# 4 cols of 5 downs and 1 up; 6 cols of 4 downs and 1 up; 5 * 10 draw cards
class SpiderSolitaire:
    class RewardState:
        def __init__(self, exposed_hidden_card=False, created_empty_pile=False, built_sequence=False, built_on_higher_card_out_of_suit=False, maximized_card_exposure_before_new_deal=False):
            self.exposed_hidden_card = exposed_hidden_card
            self.created_empty_pile = created_empty_pile
            self.built_sequence = built_sequence
            self.built_on_higher_card_out_of_suit = built_on_higher_card_out_of_suit
            self.maximized_card_exposure_before_new_deal = maximized_card_exposure_before_new_deal
    
    
    def __init__(self,suits=4,seed=None):
        self.previous_states = set()
        self.previous_games = set()
        self.stuck_moves = 0
        self.reward_state = self.RewardState()
        self.draw_count = 0
        
        # 104 cards
        # 1 suit -> 13 cards * 8
        # 2 suits -> 2*13 cards * 4
        # 3 suits -> 3*13 cards * 2 = 78; 2*13 cards = 26; 78 + 26 = 104
        # 4 suits -> 4*13 cards * 2 = 104
        # switch case
        if suits == 1:
            self.deck = Deck.Deck(8,1,seed)
        elif suits == 2:
            self.deck = Deck.Deck(4,2,seed)
        elif suits == 3:
            self.deck = Deck.Deck(2,3,seed)
            self.deck.add_suit(0)
            self.deck.add_suit(1)
        else:
            self.deck = Deck.Deck(2,4,seed)  # Using 2 decks for Spider Solitaire
        self.deck.shuffle()
        self.tableau = [[] for _ in range(10)]  # 10 columns in the tableau
        self.deal_initial_cards()

    def deal_initial_cards(self):
        # Deal cards to the tableau with the specific rules of Spider Solitaire
        for i in range(10):
            for j in range(6 if i < 4 else 5):  # First 4 cols get 6 cards, rest get 5
                card = self.deck.cards.pop()  # Take the top card from the deck
                self.tableau[i].append(card)

        # Turn the top card of each column face up
        for column in self.tableau:
            column[-1].face_up = True

    def draw_cards(self):
        """Draws one card for each column if available."""
        if len(self.deck.cards) < 10:
            print("Not enough cards in the deck to draw.")
            self.draw_count += 1
            return

        for column in self.tableau:
            card = self.deck.deal()
            if card:
                card.face_up = True
                column.append(card)
        self.remove_complete_sets()

    def get_game_state(self):
        """Returns a string representation of the current game state."""
        state = ''
        for column in self.tableau:
            state += ''.join([f'{card.display()}' for card in column if card.face_up])
            state += '|'
        return state

    def update_game_state(self):
        """Updates the set of previous game states and checks for repetition."""
        current_state = self.get_game_state()
        if current_state in self.previous_states:
            self.stuck_moves += 1
            # maybe we have maximized our card exposure before a new deal
            # this may not be the best or correct way, but let's just see if current_state was 2 moves ago (move 1 card to a spot and back again)
            # if current_state == self.previous_states[-2]:
            # self.reward_state.maximized_card_exposure_before_new_deal = True
        else:
            self.stuck_moves = 0
            self.previous_states.add(current_state)
            self.previous_games.add(self)

    def find_bundles(self, column):
        """Finds all possible, movable bundles in a given column."""
        bundles = []
        for i in range(len(column)-1, -1, -1):
            if not (column[i].face_up and (i == len(column)-1 or column[i].rank == column[i+1].rank + 1) and (i == len(column)-1 or column[i].suit == column[i+1].suit)):
                break
            bundles.append(column[i:])

        return bundles

    def can_move_bundle(self, bundle, target_column):
        """Checks if a bundle can be moved to the target column."""
        if not bundle:
            return False
        # check if the bundle is all the same suit
        if not all(card.suit == bundle[0].suit for card in bundle):
            return False
        if not target_column or len(target_column) == 0:  # Target column is empty
            return True
        return bundle[0].rank == target_column[-1].rank - 1

    def move_bundle(self, source_column_index, target_column_index, bundle_length):
        """Moves a bundle from the source column to the target column."""
        source_column = self.tableau[source_column_index]
        target_column = self.tableau[target_column_index]

        # Check if the move is valid
        if not source_column or len(source_column) < bundle_length:
            print("Invalid move: No such bundle in the source column.")
            return False

        bundle = source_column[-bundle_length:]
        if self.can_move_bundle(bundle, target_column):
            # Move the bundle
            self.tableau[source_column_index] = source_column[:-bundle_length]
            self.tableau[target_column_index].extend(bundle)
            # check if this created an empty pile
            if len(source_column) == 0:
                self.reward_state.created_empty_pile = True
            # check if this built a sequence by checking to see if the target column's last card is the same suit as the bundle's first card
            if target_column and target_column[-1].suit == bundle[0].suit:
                self.reward_state.built_sequence = True
            # check if this built on a higher card out of suit
            # this is always true unless we move a bundle to an empty column
            if target_column and len(target_column) != 0 and target_column[-1].rank > bundle[0].rank and target_column[-1].suit != bundle[0].suit:
                self.reward_state.built_on_higher_card_out_of_suit = True 
            self.remove_complete_sets()
            return True
        else:
            print("Invalid move: Bundle cannot be moved to the target column.")
            return False

    def remove_complete_sets(self):
        """Removes complete sets (King to Ace of the same suit) from the tableau."""
        column_index = -1
        for column in self.tableau:
            column_index += 1
            if len(column) >= 13:
                # Check if the last 13 cards form a complete set
                if self.is_complete_set(column[-13:]):
                    # Remove the complete set
                    del column[-13:]
            # Flip the next card in the source column, if there is one
            if self.tableau[column_index] and not self.tableau[column_index][-1].face_up:
                self.tableau[column_index][-1].face_up = True
                self.reward_state.exposed_hidden_card = True
            elif len(column) == 0:
                self.reward_state.created_empty_pile = True
        self.update_game_state()

    def is_complete_set(self, cards):
        """Checks if the given cards form a complete set from King to Ace of the same suit."""
        if len(cards) != 13:
            return False
        expected_rank = 13  # Start from King
        for card in cards:
            if card.rank != expected_rank or card.suit != cards[0].suit:
                return False
            expected_rank -= 1
        return expected_rank == 0  # True if the sequence ended with Ace
        
    def show_bundles(self):
        for i in range(10):
            print("Column %d:" % i)
            for j in self.find_bundles(self.tableau[i]):
                print('[', end='')
                for k in j:
                    print(k.display(), end='')
                print(']', end=' ')
            print()
    
    def has_won(self):
        """Checks if the player has won the game."""
        # Check if the player has won
        for column in self.tableau:
            if len(column) > 0:
                return False
        return True

    def display_board(self):
            """Displays the current state of the game board."""
            print("Spider Solitaire Board:")
            max_length = max([len(column) for column in self.tableau])
            for row in range(max_length):
                row_display = []
                for column in self.tableau:
                    if row < len(column):
                        row_display.append(column[row].display())
                    else:
                        row_display.append("   ")
                print(" ".join(row_display))

            print("\nDraw Pile: %s cards" % len(self.deck.cards))