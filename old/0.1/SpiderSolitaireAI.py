import copy
from typing import List

from SpiderSolitaire import SpiderSolitaire


class SpiderSolitaireAI:
    class Move(object):
        def __init__(
            self, move_type: str, source: int, target: int, bundle_length: int
        ):
            self.move_type = move_type
            self.source = source
            self.target = target
            self.bundle_length = bundle_length

        def __str__(self):
            if self.move_type == "draw" or self.move_type == "undo":
                return self.move_type.capitalize()
            return f"{self.move_type.capitalize()} ({self.source}, {self.target}, {self.bundle_length})"

    def __init__(self, game: SpiderSolitaire, depth: int = 2, discount: float = 0.9):
        self.game = game
        self.depth = depth
        self.discount = discount
        self.q_table = {}
        self.previous_states = {}

    def get_state_representation(self, game: SpiderSolitaire) -> str:
        return game.get_game_state()

    def detect_repeating_pattern(self, game: SpiderSolitaire) -> bool:
        state = self.get_state_representation(game)
        self.previous_states[state] = self.previous_states.get(state, 0) + 1
        return self.previous_states[state] > 2

    def evaluate_state(self, game: SpiderSolitaire) -> int:
        revealed_count = sum(1 for col in game.tableau for card in col if card.face_up)
        hidden_count = sum(
            1 for col in game.tableau for card in col if not card.face_up
        )
        empty_columns = sum(1 for col in game.tableau if not col)

        score = revealed_count * 10
        completed_foundations = game.completed_sets
        score += completed_foundations * 130
        score -= game.undo_count * 5
        score -= game.undo_from_stock_count * 100

        if game.has_won():
            min_moves = 100
            avg_moves = 140
            move_efficiency = (
                max(0, (avg_moves - game.move_count) / (avg_moves - min_moves)) * 100
            )
            score += move_efficiency
        else:
            estimated_moves_remaining = hidden_count / 3 - empty_columns * 5
            score += max(0, 100 - estimated_moves_remaining)

        if self.detect_repeating_pattern(game):
            score -= 50

        return score

    def evaluate_move(self, game: SpiderSolitaire, move: Move, depth: int) -> int:
        new_game = self.simulate_move(game, move)
        if new_game is None:
            return -float("inf")

        immediate_reward = self.evaluate_state(new_game)

        if move.move_type == "move":
            source_card = game.tableau[move.source][-move.bundle_length]
            target_card = (
                game.tableau[move.target][-1] if game.tableau[move.target] else None
            )
            if target_card and source_card.suit == target_card.suit:
                immediate_reward += 5  # Bonus for moving within the same suit
            else:
                immediate_reward -= 5  # Penalty for moving across suits

        if depth == 1:
            return immediate_reward

        possible_moves = self.get_possible_moves(new_game)
        if not possible_moves:
            return immediate_reward

        future_reward = max(
            self.evaluate_move(new_game, next_move, depth - 1)
            for next_move in possible_moves
        )
        return immediate_reward + self.discount * future_reward

    def simulate_move(self, game: SpiderSolitaire, move: Move) -> SpiderSolitaire:
        new_game = copy.deepcopy(game)
        if move.move_type == "move":
            valid = new_game.move_bundle(move.source, move.target, move.bundle_length)
            if not valid:
                return None
        elif move.move_type == "draw":
            new_game.draw_cards()
        elif move.move_type == "undo":
            new_game.undo()
        return new_game

    def choose_best_move(self) -> Move | None:
        possible_moves = self.get_possible_moves(self.game)
        if not possible_moves:
            return None

        best_move = None
        best_value = -float("inf")
        for move in possible_moves:
            value = self.evaluate_move(self.game, move, self.depth)
            state_after_move = self.simulate_move(self.game, move)
            if state_after_move and self.detect_repeating_pattern(state_after_move):
                value -= 100
            print(f"{move} has evaluated value {value}")
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def get_possible_moves(self, game: SpiderSolitaire) -> List[Move]:
        moves = []
        for i, column in enumerate(game.tableau):
            bundles = game.find_bundles(column)
            if not bundles:
                continue
            for bundle in bundles:
                for j, target_column in enumerate(game.tableau):
                    if i == j:
                        continue
                    if game.can_move_bundle(bundle, target_column):
                        moves.append(self.Move("move", i, j, len(bundle)))

        if len(game.deck.cards) >= 10:
            moves.append(self.Move("draw", -1, -1, -1))
        if game.can_undo():
            moves.append(self.Move("undo", -1, -1, -1))

        return moves

    def update_q_value(self, state, move, reward, next_state):
        key = (state, move)
        max_next_q = max(
            [
                self.q_table.get((next_state, m), 0)
                for m in self.get_possible_moves(self.game)
            ]
            or [0]
        )
        current_q = self.q_table.get(key, 0)
        learning_rate = 0.01
        self.q_table[key] = current_q + learning_rate * (
            reward + self.discount * max_next_q - current_q
        )


if __name__ == "__main__":
    game = SpiderSolitaire(suits=4, seed=42)
    ai = SpiderSolitaireAI(game, depth=4)

    while not game.has_won():
        game.display_board()
        best_move = ai.choose_best_move()
        if best_move:
            print("AI recommends move:", best_move)
        else:
            print("No valid moves found.")

        if best_move.move_type == "move":
            game.move_bundle(
                best_move.source, best_move.target, best_move.bundle_length
            )
        elif best_move.move_type == "draw":
            game.draw_cards()
        elif best_move.move_type == "undo":
            game.undo()
