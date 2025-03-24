import hashlib
import time
from typing import List
from SpiderSolitaire import SpiderSolitaire
from bcolors import bcolors


class Move:
    def __init__(self, move_type: str, source: int, target: int, bundle_length: int):
        self.move_type = move_type
        self.source = source
        self.target = target
        self.bundle_length = bundle_length

    def __str__(self):
        if self.move_type in {"draw", "undo"}:
            return self.move_type.capitalize()
        return f"{self.move_type.capitalize()} ({self.source}, {self.target}, {self.bundle_length})"


class SpiderSolver:
    def __init__(self, suits=4, seed=None):
        self.game = SpiderSolitaire(suits, seed)
        self.visited_states = set()
        self.move_sequence = []
        self.branch_count = 0
        self.start_time = time.time()

    def hash_state(self) -> str:
        """Generate a unique hash for the current game state."""
        state_str = self.game.get_game_state()
        return hashlib.md5(state_str.encode(), usedforsecurity=False).hexdigest()

    def get_possible_moves(self) -> List[Move]:
        """Find all possible valid moves in the current game state."""
        moves = []

        # Check for valid card movements
        for i, column in enumerate(self.game.tableau):
            bundles = self.game.find_bundles(column)
            if not bundles:
                continue
            for bundle in bundles:
                for j, target_column in enumerate(self.game.tableau):
                    if i == j:
                        continue
                    if self.game.can_move_bundle(bundle, target_column):
                        moves.append(Move("move", i, j, len(bundle)))

        # Check if a draw is possible
        if len(self.game.deck.cards) >= 10:
            moves.append(Move("draw", -1, -1, -1))

        # Check if undo is possible # DEBUG: TRY WITHOUT
        # if self.game.can_undo():
        #     moves.append(Move("undo", -1, -1, -1))

        return moves

    def solve(self, depth=0, max_depth=100):
        """Brute-force search to solve Spider Solitaire using DFS."""
        if depth > max_depth:
            return False

        if self.game.has_won():
            print("\n🎉 Solution Found! 🎉\n")
            for move in self.move_sequence:
                print(move)
            return True

        # Generate hash of current state
        state_hash = self.hash_state()

        # If we have seen this state before, prune the search
        if state_hash in self.visited_states:
            # print(bcolors.FAIL + "# DEBUG Pruned branch (VISITED STATE)"+bcolors.ENDC)
            return False
        self.visited_states.add(state_hash)

        # Get possible moves
        moves = self.get_possible_moves()
        # self.game.display_board()
        # print([str(move) for move in moves])

        for move in moves:
            if move.move_type == "move":
                success = self.game.move_bundle(
                    move.source, move.target, move.bundle_length)
            elif move.move_type == "draw":
                success = self.game.draw_cards()
            elif move.move_type == "undo":
                success = self.game.undo()
            else:
                continue

            if success:
                move_log = f"# DEBUG Branch ({self.branch_count}, {depth}): {move}"
                self.move_sequence.append(move_log)
                # print(move_log)

                # Recursive call
                if self.solve(depth + 1, max_depth):
                    return True  # Solution found!

                self.branch_count += 1
                # Backtrack (Undo last move)
                self.game.undo()
                self.move_sequence.pop()

        return False  # No solution found at this branch


if __name__ == '__main__':
    # Run the solver
    for d in range(1, 11):
        now = time.time()
        solver = SpiderSolver(suits=1, seed=42)
        solver.solve(depth=0, max_depth=d)
        print(f"({d},{time.time() - now})")
