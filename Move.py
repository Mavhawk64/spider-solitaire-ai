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
