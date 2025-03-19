import copy
from typing import List

from bcolors import *
from Card import Card
from Deck import Deck


class SpiderSolitaire:
    class RewardState:
        def __init__(
            self,
            exposed_hidden_card=False,
            created_empty_pile=False,
            built_sequence=False,
            built_on_higher_card_out_of_suit=False,
            maximized_card_exposure_before_new_deal=False,
        ):
            self.exposed_hidden_card = exposed_hidden_card
            self.created_empty_pile = created_empty_pile
            self.built_sequence = built_sequence
            self.built_on_higher_card_out_of_suit = built_on_higher_card_out_of_suit
            self.maximized_card_exposure_before_new_deal = (
                maximized_card_exposure_before_new_deal
            )

    def __init__(
        self, suits: int = 4, seed: int | float | str | bytes | bytearray | None = None
    ):
        # History stack for undo functionality.
        self.state_history: List[dict] = []
        self.stuck_moves = 0
        self.reward_state = self.RewardState()
        self.draw_count = 0
        self.completed_sets = 0
        self.undo_count = 0
        self.undo_from_stock_count = 0
        self.just_drew = False
        self.move_count = 0

        # Build the deck based on the number of suits.
        if suits == 1:
            self.deck = Deck(8, 1, seed)
        elif suits == 2:
            self.deck = Deck(4, 2, seed)
        elif suits == 3:
            self.deck = Deck(2, 3, seed)
            self.deck.add_suit(0)
            self.deck.add_suit(1)
        else:
            self.deck = Deck(2, 4, seed)  # Standard Spider Solitaire uses 2 decks.
        self.deck.shuffle()
        self.tableau: List[List[Card]] = [[] for _ in range(10)]  # 10 columns.
        self.deal_initial_cards()

    def deal_initial_cards(self):
        """Deal cards to the tableau according to Spider Solitaire rules."""
        for i in range(10):
            for j in range(6 if i < 4 else 5):
                card = self.deck.cards.pop()
                self.tableau[i].append(card)
        # Turn the top card of each column face up.
        for column in self.tableau:
            if column:
                column[-1].face_up = True

    def save_state(self):
        """
        Save a snapshot of the current game state.
        This snapshot includes the tableau, the deck, draw count, stuck moves, and reward state.
        """
        snapshot = {
            "tableau": copy.deepcopy(self.tableau),
            "deck": copy.deepcopy(self.deck.cards),
            "draw_count": self.draw_count,
            "stuck_moves": self.stuck_moves,
            "reward_state": copy.deepcopy(self.reward_state),
            "completed_sets": self.completed_sets,
            "undo_count": self.undo_count,
            "undo_from_stock_count": self.undo_from_stock_count,
        }
        self.state_history.append(snapshot)

    def restore_state(self, snapshot: dict):
        """Restore the game state from a snapshot."""
        self.tableau = snapshot["tableau"]
        self.deck.cards = snapshot["deck"]
        self.draw_count = snapshot["draw_count"]
        self.stuck_moves = snapshot["stuck_moves"]
        self.reward_state = snapshot["reward_state"]

    def can_undo(self) -> bool:
        """Check if there are any saved states to undo."""
        return bool(self.state_history)

    def undo(self) -> bool:
        """
        Undo the player's latest move.
        Returns True if successful, False if no previous state exists.
        """
        if not self.state_history:
            print("No moves to undo.")
            return False
        # Pop the last snapshot and restore it.
        snapshot = self.state_history.pop()
        self.restore_state(snapshot)
        self.undo_count += 1
        if self.just_drew:
            self.undo_from_stock_count += 1
            self.just_drew = False
        self.move_count += 1
        return True

    def draw_cards(self):
        """Draw one card for each column if available.
        Note: This method does not call save_state() internally.
        The calling code should call save_state() before drawing if an undo is desired.
        """
        if len(self.deck.cards) < 10:
            print("Not enough cards in the deck to draw.")
            self.draw_count += 1
            return
        self.save_state()
        for column in self.tableau:
            card = self.deck.deal()
            if card:
                card.face_up = True
                column.append(card)
        self.remove_complete_sets()
        self.just_drew = True
        self.move_count += 1

    def get_game_state(self) -> str:
        """Return a string representation of the current game state."""
        state = ""
        for column in self.tableau:
            state += "".join([f"{card.display()}" for card in column if card.face_up])
            state += "|"
        return state

    def find_bundles(self, column: List[Card]) -> List[List[Card]]:
        """
        Find all possible movable bundles in a given column.
        A bundle is a sequence of face-up cards in descending order and of the same suit.
        """
        bundles: List[List[Card]] = []
        for i in range(len(column) - 1, -1, -1):
            if not (
                column[i].face_up
                and (i == len(column) - 1 or column[i].rank == column[i + 1].rank + 1)
                and (i == len(column) - 1 or column[i].suit == column[i + 1].suit)
            ):
                break
            bundles.append(column[i:])
        return bundles

    def can_move_bundle(self, bundle: List[Card], target_column: List[Card]) -> bool:
        """
        Check if a bundle can be moved to the target column.
        """
        if not bundle or not all(card.suit == bundle[0].suit for card in bundle):
            return False
        if not target_column or len(target_column) == 0:
            return True
        return bundle[0].rank == target_column[-1].rank - 1

    def move_bundle(
        self, source_column_index: int, target_column_index: int, bundle_length: int
    ) -> bool:
        """
        Move a bundle from the source column to the target column.
        Note: Do not call save_state() inside this method. The calling code should
        have saved the state before invoking this move.
        """
        source_column = self.tableau[source_column_index]
        target_column = self.tableau[target_column_index]

        if not source_column or len(source_column) < bundle_length:
            print("Invalid move: No such bundle in the source column.")
            return False

        bundle = source_column[-bundle_length:]
        if self.can_move_bundle(bundle, target_column):
            self.save_state()
            self.tableau[source_column_index] = source_column[:-bundle_length]
            self.tableau[target_column_index].extend(bundle)
            self.remove_complete_sets()
            self.just_drew = False
            self.move_count += 1
            return True
        print("Invalid move: Bundle cannot be moved to the target column.")
        return False

    def remove_complete_sets(self):
        """
        Remove complete sets (King to Ace of the same suit) from the tableau.
        Also, flip the next card in a column if it is face down.
        Note: Automatic flips and removals are not separately saved; they are part of the move.
        """
        for idx, column in enumerate(self.tableau):
            if len(column) >= 13:
                if self.is_complete_set(column[-13:]):
                    del column[-13:]
                    self.reward_state.built_sequence = True
                    self.completed_sets += 1
            if column and not column[-1].face_up:
                column[-1].face_up = True
                self.reward_state.exposed_hidden_card = True
            elif len(column) == 0:
                self.reward_state.created_empty_pile = True

    def is_complete_set(self, cards: List[Card]) -> bool:
        """
        Check if the given 13 cards form a complete set (King to Ace) of the same suit.
        """
        if len(cards) != 13:
            return False
        expected_rank = 13  # Start from King.
        for card in cards:
            if card.rank != expected_rank or card.suit != cards[0].suit:
                return False
            expected_rank -= 1
        return expected_rank == 0

    def show_bundles(self):
        """Display all movable bundles in each column."""
        for i in range(10):
            print("Column %d:" % i)
            for bundle in self.find_bundles(self.tableau[i]):
                print("[", end="")
                for card in bundle:
                    print(card.display(), end=" ")
                print("]", end=" ")
            print()

    def has_won(self) -> bool:
        """Check if the player has won the game (all columns empty)."""
        for column in self.tableau:
            if len(column) > 0:
                return False
        return True

    def display_board(self):
        """Display the current state of the game board."""
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
        print("\nDraw Pile: {} cards".format(len(self.deck.cards)))
