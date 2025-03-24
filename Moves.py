from typing import List


class Move(object):
    def __init__(self, move_type: str, source: int, target: int, bundle_length: int):
        self.move_type = move_type
        self.source = source
        self.target = target
        self.bundle_length = bundle_length

    def __str__(self):
        if self.move_type in {"draw", "undo"}:
            return self.move_type.capitalize()
        return f"{self.move_type.capitalize()} ({self.source}, {self.target}, {self.bundle_length})"


class Moves(object):
    def __init__(self, moves: List[Move] = []):
        self.moves = moves

    def append(self, move: Move):
        self.moves.append(move)

    def __str__(self):
        return ", ".join(str(move) for move in self.moves)

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx: int):
        return self.moves[idx]

    def __iter__(self):
        return iter(self.moves)

    def index(self, move: Move) -> int:
        for i, m in enumerate(self.moves):
            if (
                m.move_type == move.move_type
                and m.source == move.source
                and m.target == move.target
                and m.bundle_length == move.bundle_length
            ):
                return i
        return -1
