import random
from typing import List

from Card import Card


class Deck(object):
    """Represents a deck/set of cards."""

    def __init__(
        self,
        decks: int = 1,
        suits: int = 4,
        seed: int | float | str | bytes | bytearray | None = None,
    ):
        random.seed(seed)
        self.cards: List[Card] = []
        for _ in range(decks):
            for suit in range(suits):
                for rank in range(1, 14):
                    self.cards.append(Card(suit, rank))

    def __str__(self) -> str:
        res = []
        for card in self.cards:
            res.append(str(card))
        return "\n".join(res)

    def __len__(self) -> int:
        return len(self.cards)

    def add_suit(self, suit):
        for rank in range(1, 14):
            self.cards.append(Card(suit, rank))

    def shuffle(self):
        random.shuffle(self.cards)

    def sort(self):
        self.cards.sort()

    def deal(self) -> Card | None:
        """Deals and returns the top card from the deck."""
        return self.cards.pop() if self.cards else None
